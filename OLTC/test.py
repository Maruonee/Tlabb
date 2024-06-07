import sys
import os
import time
import threading
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import serial
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt

# Logger 클래스: 로그 메시지를 UI로 전달
class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)

# RecorderWorker 클래스: 오디오 녹음을 수행하는 작업자
class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event, unlimited):
        super().__init__()
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels
        self.folder_path = folder_path
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event
        self.unlimited = unlimited

    def run(self):
        i = 0
        while self.unlimited or i < self.repeat_num:
            if self.stop_event.is_set():
                self.log_signal.emit("녹음 중지 요청됨.")
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')

            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')
                    self.log_signal.emit(f"{filename} 저장완료.")
                    return
                self.progress_signal.emit(second + 1, self.duration)
                time.sleep(1)
            if self.stop_event.is_set():
                break
            sd.wait()
            sf.write(filename, recording, self.samplerate, format='FLAC')
            self.log_signal.emit(f"{filename} 저장완료.")
            self.progress_signal.emit(self.duration, self.duration)
            if not self.unlimited:
                self.total_progress_signal.emit(i + 1)
            i += 1

        if not self.unlimited:
            self.log_signal.emit("모든 녹음이 완료되었습니다.")
        self.finished_signal.emit()
        self.stop_event.set()

# DataCollectorWorker 클래스: 센서 데이터를 수집하는 작업자
class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, serial_port, samplerate, folder_path, duration, repeat_num, exp_date, exp_num, stop_event, unlimited):
        super().__init__()
        self.serial_port = serial_port
        self.samplerate = samplerate
        self.folder_path = folder_path
        self.duration = duration
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event
        self.serial_connection = None
        self.unlimited = unlimited

    def run(self):
        try:
            self.serial_connection = serial.Serial(self.serial_port, self.samplerate)
        except serial.SerialException as e:
            self.log_signal.emit(f"직렬 포트를 열 수 없습니다: {e}. 음향 녹음만 진행합니다.")
            self.finished_signal.emit()
            return

        txt_file_ref, initial_filename = self.create_new_file()
        txt_file_ref = [txt_file_ref]
        lock = threading.Lock()

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)
        file_refresh_thread_obj.start()

        i = 0
        try:
            while self.unlimited or i < self.repeat_num:
                if self.stop_event.is_set():
                    break
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.readline().decode('utf-8').strip()
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')
                            txt_file_ref[0].flush()
                if not self.unlimited:
                    i += 1
        except KeyboardInterrupt:
            self.log_signal.emit("데이터 수집 중지 요청됨.")
        finally:
            with lock:
                txt_file_ref[0].close()
            if self.serial_connection is not None:
                self.serial_connection.close()
            self.finished_signal.emit()

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')
        return open(filename, mode='w'), filename

    def file_refresh_thread(self, txt_file_ref, lock):
        i = 0
        while self.unlimited or i < self.repeat_num:
            for _ in range(self.duration):
                if self.stop_event.is_set():
                    return
                time.sleep(1)
                self.progress_signal.emit(_ + 1, self.duration)
            new_file, filename = self.create_new_file()
            with lock:
                txt_file_ref[0].close()
                txt_file_ref[0] = new_file
            self.log_signal.emit(f"{filename} 저장완료.")
            if not self.unlimited:
                self.total_progress_signal.emit(i + 1)
            i += 1
        if not self.unlimited:
            self.log_signal.emit("모든 데이터 획득이 완료하였습니다.")
        self.finished_signal.emit()

# 저장 폴더를 생성하는 함수
def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

# 메인 애플리케이션 클래스
class DataCollectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.logger = Logger()
        self.logger.log_signal.connect(self.update_log)

    def initUI(self):
        self.setWindowTitle('OLTC Audio & Sensor Recorder Tlab')
        self.resize(500, 700)

        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

        # 고장진단상황 프레임
        status_frame = QGroupBox('고장진단상황')
        status_layout = QVBoxLayout()
        status_frame.setLayout(status_layout)
        
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        
        # 파일 교체 주기 및 반복 횟수 프레임
        duration_frame = QGroupBox('파일 교체 주기 및 반복 횟수')
        duration_layout = QVBoxLayout()
        self.duration_label = QLabel('파일 교체 주기(초)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        self.repeat_num_label = QLabel('반복횟수')
        repeat_num_layout = QHBoxLayout()
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        self.unlimited_repeat_checkbox = QCheckBox("무제한")
        self.unlimited_repeat_checkbox.stateChanged.connect(self.toggle_repeat_input)
        repeat_num_layout.addWidget(self.repeat_num_input)
        repeat_num_layout.addWidget(self.unlimited_repeat_checkbox)
        duration_layout.addWidget(self.duration_label)
        duration_layout.addWidget(self.duration_input)
        duration_layout.addWidget(self.repeat_num_label)
        duration_layout.addLayout(repeat_num_layout)
        duration_frame.setLayout(duration_layout)
        
        # 진동데이터 프레임
        vibration_frame = QGroupBox('진동데이터')
        vibration_layout = QVBoxLayout()
        self.serial_port_label = QLabel('시리얼 포트')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM7')
        self.samplerate_label = QLabel('보드레이트 (bps)')
        self.samplerate_input = QLineEdit(self)
        self.samplerate_input.setText('19200')
        vibration_layout.addWidget(self.serial_port_label)
        vibration_layout.addWidget(self.serial_port_input)
        vibration_layout.addWidget(self.samplerate_label)
        vibration_layout.addWidget(self.samplerate_input)
        vibration_frame.setLayout(vibration_layout)
        
        # 음향데이터 프레임
        audio_frame = QGroupBox('음향데이터')
        audio_layout = QVBoxLayout()
        self.audio_samplerate_label = QLabel('오디오 샘플링레이트 (Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')
        audio_layout.addWidget(self.audio_samplerate_label)
        audio_layout.addWidget(self.audio_samplerate_input)
        audio_frame.setLayout(audio_layout)
        
        # 실험번호 및 실험날짜 프레임
        exp_frame = QGroupBox('실험정보')
        exp_layout = QVBoxLayout()
        self.exp_date_label = QLabel('실험날짜 (YYMMDD)')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        self.exp_num_label = QLabel('실험번호')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')
        exp_layout.addWidget(self.exp_date_label)
        exp_layout.addWidget(self.exp_date_input)
        exp_layout.addWidget(self.exp_num_label)
        exp_layout.addWidget(self.exp_num_input)
        exp_frame.setLayout(exp_layout)
        
        # 진행률 프레임
        progress_frame = QGroupBox('진행률')
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel('개별 진행률')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.total_progress_label = QLabel('전체 진행률')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.total_progress_label)
        progress_layout.addWidget(self.total_progress_bar)
        progress_frame.setLayout(progress_layout)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('데이터 수집 시작', self)
        self.start_button.clicked.connect(self.start_collection)
        self.stop_button = QPushButton('데이터 수집 중단', self)
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        # 로그 출력창
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(status_frame)
        layout.addWidget(self.savedir_label)
        layout.addWidget(self.savedir_input)
        layout.addWidget(self.savedir_button)
        layout.addWidget(duration_frame)
        layout.addWidget(vibration_frame)
        layout.addWidget(audio_frame)
        layout.addWidget(exp_frame)
        layout.addLayout(button_layout)
        layout.addWidget(progress_frame)
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)

    def toggle_repeat_input(self):
        if self.unlimited_repeat_checkbox.isChecked():
            self.repeat_num_input.setEnabled(False)
            self.total_progress_bar.setMaximum(0)
            self.total_progress_bar.setStyleSheet("QProgressBar { text-align: center; } QProgressBar::chunk { background-color: lightblue; }")
        else:
            self.repeat_num_input.setEnabled(True)
            self.total_progress_bar.setStyleSheet("")
            self.total_progress_bar.setMaximum(int(self.repeat_num_input.text()))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)
    
    def start_collection(self):
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.samplerate = int(self.samplerate_input.text())
        self.serial_port = self.serial_port_input.text()
        self.unlimited = self.unlimited_repeat_checkbox.isChecked()
        if not self.unlimited:
            try:
                self.repeat_num = int(self.repeat_num_input.text())
            except ValueError:
                self.repeat_num = 1
        else:
            self.repeat_num = 1  # 임의의 값, 실제로 사용되지 않음
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        self.audio_samplerate = int(self.audio_samplerate_input.text())
        self.audio_duration = int(self.duration_input.text())
        
        self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')
        
        self.stop_event = threading.Event()
        
        self.logger.log(f"=======데이터 수집 시작=======\n파일 교체 주기: {self.duration}초\n반복 횟수: {'무제한' if self.unlimited else self.repeat_num}\n저장 위치 (센서): {self.sensor_recordings_folder_path}\n저장 위치 (오디오): {self.audio_recordings_folder_path}\n")

        # 음향 녹음 작업자 설정
        self.recorder_worker = RecorderWorker(self.audio_duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event, self.unlimited)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)
        
        self.recorder_thread.started.connect(self.recorder_worker.run)
        
        # 시리얼 포트 설정 및 데이터 수집 작업자 설정
        try:
            self.data_collector_worker = DataCollectorWorker(self.serial_port, self.samplerate, self.sensor_recordings_folder_path, self.duration, self.repeat_num, self.exp_date, self.exp_num, self.stop_event, self.unlimited)
            self.data_collector_thread = QThread()
            self.data_collector_worker.moveToThread(self.data_collector_thread)
            
            self.data_collector_worker.progress_signal.connect(self.update_progress)
            self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
            self.data_collector_worker.log_signal.connect(self.logger.log)
            self.data_collector_worker.finished_signal.connect(self.collection_finished)
            
            self.data_collector_thread.started.connect(self.data_collector_worker.run)
            
            self.data_collector_thread.start()
        except serial.SerialException as e:
            self.logger.log(f"시리얼 포트를 열 수 없습니다: {e}. 음향 녹음만 진행합니다.")
        
        self.recorder_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.progress_bar.setMaximum(max(self.duration, self.audio_duration))
        self.progress_bar.setValue(0)
        if not self.unlimited:
            self.total_progress_bar.setMaximum(self.repeat_num)
            self.total_progress_bar.setValue(0)
        
    def stop_collection(self):
        self.stop_event.set()
        if hasattr(self, 'data_collector_thread') and self.data_collector_thread.isRunning():
            self.data_collector_thread.quit()
            self.data_collector_thread.wait()
        if hasattr(self, 'recorder_thread') and self.recorder_thread.isRunning():
            self.recorder_thread.quit()
            self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.logger.log("데이터 수집이 중지되었습니다.")
    
    def collection_finished(self):
        if hasattr(self, 'data_collector_thread') and self.data_collector_thread.isRunning():
            self.data_collector_thread.quit()
            self.data_collector_thread.wait()
        if hasattr(self, 'recorder_thread') and self.recorder_thread.isRunning():
            self.recorder_thread.quit()
            self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value):
        if not self.unlimited_repeat_checkbox.isChecked():
            self.total_progress_bar.setMaximum(int(self.repeat_num_input.text()))
        self.total_progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorApp()
    ex.show()
    sys.exit(app.exec_())