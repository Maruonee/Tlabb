import sys
import os
import time
import threading
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer

machine_error = 1  # 0 = 정상, 1 = 고장 예측, 2 = 고장

class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)

class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    data_signal = pyqtSignal(np.ndarray)

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels
        self.folder_path = folder_path
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event

    def run(self):
        for i in range(self.repeat_num):
            if self.stop_event.is_set():
                self.log_signal.emit("음향 진단 중지")
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')
            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')
                    self.log_signal.emit(f"{filename} 저장.")
                    return
                self.data_signal.emit(recording[:int((second + 1) * self.samplerate)].flatten())
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)
                time.sleep(1)
            if self.stop_event.is_set():
                break
            sd.wait()
            sf.write(filename, recording, self.samplerate, format='FLAC')
            self.log_signal.emit(f"{filename} 저장.")
            self.data_signal.emit(recording.flatten())
            self.progress_signal.emit(100, 100)
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)
        
        self.log_signal.emit("설정한 기간의 음향 진단 완료")
        self.finished_signal.emit()
        self.stop_event.set()

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration
        self.baud_rate = baud_rate
        self.serial_port = serial_port
        self.folder_path = folder_path
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)
        except serial.SerialException as e:
            self.log_signal.emit(f"직렬 포트를 열 수 없습니다: {e}")
            self.finished_signal.emit()
            return

        txt_file_ref, initial_filename = self.create_new_file()
        txt_file_ref = [txt_file_ref]
        lock = threading.Lock()

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)
        file_refresh_thread_obj.start()

        try:
            while file_refresh_thread_obj.is_alive():
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')
                            txt_file_ref[0].flush()
        except KeyboardInterrupt:
            self.log_signal.emit("진동 진단 중지")
        finally:
            with lock:
                txt_file_ref[0].close()
            ser.close()
            self.finished_signal.emit()
            
    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')
        return open(filename, mode='w'), filename

    def file_refresh_thread(self, txt_file_ref, lock):
        for i in range(self.repeat_num):
            for _ in range(self.duration):
                if self.stop_event.is_set():
                    return
                time.sleep(1)
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100)
            new_file, filename = self.create_new_file()
            with lock:
                txt_file_ref[0].close()
                txt_file_ref[0] = new_file
            self.log_signal.emit(f"{filename} 저장.")
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)
        self.log_signal.emit("설정한 기간의 진동 진단 완료")
        self.finished_signal.emit()

class GraphUpdater(QObject):
    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, graph_data_length):
        super().__init__()
        self.graph_data = np.zeros(graph_data_length)
        self.lock = threading.Lock()

    def update_data(self, data):
        with self.lock:
            flattened_data = data.flatten()
            self.graph_data = np.roll(self.graph_data, -len(flattened_data))
            self.graph_data[-len(flattened_data):] = flattened_data
        self.update_signal.emit(self.graph_data)

def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.initUI()
        self.logger = Logger()
        self.machine_error = machine_error
        self.logger.log_signal.connect(self.update_log)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.status_visible = True

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')
        self.resize(1000, 700)
        
        status_frame = QGroupBox('고장진단상황')
        status_frame.setStyleSheet('background-color: white')
        status_layout = QVBoxLayout()
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 48px')
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        
        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        
        duration_frame = QGroupBox('고장진단 주기 및 기간 설정')
        duration_layout = QVBoxLayout()
        self.duration_label = QLabel('고장진단 주기(초)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        duration_hbox = QHBoxLayout()
        self.minute_checkbox = QCheckBox("1 min")
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)
        duration_hbox.addWidget(self.duration_input)
        duration_hbox.addWidget(self.minute_checkbox)
        self.repeat_num_label = QLabel('고장진단 기간(주기반복횟수)')
        repeat_num_layout = QHBoxLayout()
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        self.hour_checkbox = QCheckBox("1 hour")
        self.hour_checkbox.stateChanged.connect(self.toggle_hour_checkbox)
        self.month_checkbox = QCheckBox("1 month")
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)
        repeat_num_layout.addWidget(self.repeat_num_input)
        repeat_num_layout.addWidget(self.hour_checkbox)
        repeat_num_layout.addWidget(self.month_checkbox)
        duration_layout.addWidget(self.duration_label)
        duration_layout.addLayout(duration_hbox)
        duration_layout.addWidget(self.repeat_num_label)
        duration_layout.addLayout(repeat_num_layout)
        duration_frame.setLayout(duration_layout)
        
        vibration_frame = QGroupBox('진동 진단 설정')
        vibration_layout = QVBoxLayout()
        self.serial_port_label = QLabel('COM Port')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM3')
        self.baud_rate_label = QLabel('Arduino Baud Rate(bps)')
        self.baud_rate_input = QLineEdit(self)
        self.baud_rate_input.setText('19200')
        vibration_layout.addWidget(self.serial_port_label)
        vibration_layout.addWidget(self.serial_port_input)
        vibration_layout.addWidget(self.baud_rate_label)
        vibration_layout.addWidget(self.baud_rate_input)
        vibration_frame.setLayout(vibration_layout)
        
        audio_frame = QGroupBox('음향 진단 설정')
        audio_layout = QVBoxLayout()
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')
        audio_layout.addWidget(self.audio_samplerate_label)
        audio_layout.addWidget(self.audio_samplerate_input)
        audio_frame.setLayout(audio_layout)
        
        exp_frame = QGroupBox('데이터 획득 설정')
        exp_layout = QVBoxLayout()
        self.exp_date_label = QLabel('날짜(YYMMDD)')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        self.exp_num_label = QLabel('번호')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')
        exp_layout.addWidget(self.exp_date_label)
        exp_layout.addWidget(self.exp_date_input)
        exp_layout.addWidget(self.exp_num_label)
        exp_layout.addWidget(self.exp_num_input)
        exp_frame.setLayout(exp_layout)
        
        progress_frame = QGroupBox('진행률')
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel('고장 진단 주기')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.total_progress_label = QLabel('고장 진단 기간')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.total_progress_label)
        progress_layout.addWidget(self.total_progress_bar)
        progress_frame.setLayout(progress_layout)
        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('고장 진단 시작', self)
        self.start_button.clicked.connect(self.start_collection)
        self.stop_button = QPushButton('고장 진단 중단', self)
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        
        # 그래프를 위한 레이아웃
        self.graph_layout = QVBoxLayout()
        self.graph_widget = pg.PlotWidget(title="실시간 음향 데이터")
        self.graph_widget.setBackground('w')
        self.graph_layout.addWidget(self.graph_widget)
        self.graph_plot = self.graph_widget.plot(pen=pg.mkPen(color='orange'))
        
        # 메인 레이아웃 설정
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        control_layout.addWidget(status_frame)
        control_layout.addWidget(self.savedir_label)
        control_layout.addWidget(self.savedir_input)
        control_layout.addWidget(self.savedir_button)
        control_layout.addWidget(duration_frame)
        control_layout.addWidget(vibration_frame)
        control_layout.addWidget(audio_frame)
        control_layout.addWidget(exp_frame)
        control_layout.addLayout(button_layout)
        control_layout.addWidget(progress_frame)
        control_layout.addWidget(self.log_output)
        
        main_layout.addLayout(control_layout, 3)
        main_layout.addLayout(self.graph_layout, 2)
        
        self.setLayout(main_layout)

    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')
            self.duration_input.setEnabled(False)
        else:
            self.duration_input.setEnabled(True)

    def toggle_hour_checkbox(self):
        if self.hour_checkbox.isChecked():
            self.repeat_num_input.setText('60')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)
    
    def start_collection(self):
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.baud_rate = int(self.baud_rate_input.text())
        self.serial_port = self.serial_port_input.text()
        self.repeat_num = int(self.repeat_num_input.text())
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        self.audio_samplerate = int(self.audio_samplerate_input.text())

        # 그래프 데이터를 초기화
        self.graph_data_length = self.duration * self.audio_samplerate * 2  # 스테레오 데이터를 처리하기 위해 2배 크기
        self.graph_updater = GraphUpdater(self.graph_data_length)
        self.graph_updater.update_signal.connect(self.update_graph)
        
        self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')
        
        self.stop_event = threading.Event()
        
        self.logger.log(f"=======ECOTAP 고장 진단 시작=======\n설정 주기: {self.duration}초\n설정 기간: {self.repeat_num}\n진동데이터 저장위치: {self.sensor_recordings_folder_path}\n음향데이터 저장위치: {self.audio_recordings_folder_path}\n")
        
        if self.machine_error == 0:
            self.timer.start(500)
        elif self.machine_error == 1:
            self.timer.start(300)
        elif self.machine_error == 2:
            self.timer.start(100)
        
        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)
        self.recorder_worker.data_signal.connect(self.graph_updater.update_data)
        
        self.data_collector_worker = DataCollectorWorker(self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.data_collector_thread = QThread()
        self.data_collector_worker.moveToThread(self.data_collector_thread)
        self.data_collector_worker.progress_signal.connect(self.update_progress)
        self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
        self.data_collector_worker.log_signal.connect(self.logger.log)
        self.data_collector_worker.finished_signal.connect(self.collection_finished)
        
        self.data_collector_thread.started.connect(self.data_collector_worker.run)
        self.recorder_thread.started.connect(self.recorder_worker.run)
        self.data_collector_thread.start()
        self.recorder_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.total_progress_bar.setMaximum(100)
        self.total_progress_bar.setValue(0)
        
    def stop_collection(self):
        self.stop_event.set()
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.logger.log("고장진단이 중지되었습니다.")
        self.timer.stop()
        self.status_label.setText('')
    
    def collection_finished(self):
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.timer.stop()
        self.status_label.setText('')

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

    def update_graph(self, graph_data):
        self.graph_plot.setData(graph_data)
    
    def update_status(self):
        if self.machine_error == 0:
            if self.status_visible:
                self.status_label.setText('정상')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: green; font-size: 48px')
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_label.setText('고장 예측')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: orange; font-size: 48px')
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_label.setText('고장')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: red; font-size: 48px')
        
        self.status_visible = not self.status_visible

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorApp(machine_error)
    ex.show()
    sys.exit(app.exec_())
