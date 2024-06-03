import sys
import os
import time
import threading
from datetime import datetime
import serial
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QProgressBar, QTextEdit)
from PyQt5.QtCore import  pyqtSignal, QObject, QThread

class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, serial_port, samplerate, folder_path, duration, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.serial_port = serial_port
        self.samplerate = samplerate
        self.folder_path = folder_path
        self.duration = duration
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.samplerate)
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
            self.log_signal.emit("데이터 수집 중지 요청됨.")
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
                self.progress_signal.emit(_ + 1, self.duration)
            new_file, filename = self.create_new_file()
            with lock:
                txt_file_ref[0].close()
                txt_file_ref[0] = new_file
            self.log_signal.emit(f"{filename} 저장완료.")
            self.total_progress_signal.emit(i + 1, self.repeat_num)
        self.log_signal.emit("모든 데이터 획득이 완료하였습니다.")
        self.finished_signal.emit()

def create_folder(savedir, exp_date, exp_num):
    folder_name = f"{exp_date}_{exp_num}_sensors"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

class DataCollectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.logger = Logger()
        self.logger.log_signal.connect(self.update_log)

    def initUI(self):
        self.setWindowTitle('OLTC Sensor Recorder Tlab')
        self.resize(500, 500)

        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
        
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        
        self.duration_label = QLabel('파일 교체 주기(초)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        
        self.samplerate_label = QLabel('보드레이트 (bps)')
        self.samplerate_input = QLineEdit(self)
        self.samplerate_input.setText('19200')
        
        self.serial_port_label = QLabel('시리얼 포트')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM3')
        
        self.repeat_num_label = QLabel('반복횟수')
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        
        self.exp_num_label = QLabel('실험번호')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')
        
        self.exp_date_label = QLabel('실험날짜 (YYMMDD)')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        
        self.start_button = QPushButton('데이터 수집 시작', self)
        self.start_button.clicked.connect(self.start_collection)

        self.stop_button = QPushButton('데이터 수집 중단', self)
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setEnabled(False)

        self.progress_label = QLabel('개별 진행률')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.total_progress_label = QLabel('전체 진행률')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.savedir_label)
        layout.addWidget(self.savedir_input)
        layout.addWidget(self.savedir_button)
        layout.addWidget(self.duration_label)
        layout.addWidget(self.duration_input)
        layout.addWidget(self.samplerate_label)
        layout.addWidget(self.samplerate_input)
        layout.addWidget(self.serial_port_label)
        layout.addWidget(self.serial_port_input)
        layout.addWidget(self.repeat_num_label)
        layout.addWidget(self.repeat_num_input)
        layout.addWidget(self.exp_num_label)
        layout.addWidget(self.exp_num_input)
        layout.addWidget(self.exp_date_label)
        layout.addWidget(self.exp_date_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.total_progress_label)
        layout.addWidget(self.total_progress_bar)
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)
    
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)
    
    def start_collection(self):
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.samplerate = int(self.samplerate_input.text())
        self.serial_port = self.serial_port_input.text()
        self.repeat_num = int(self.repeat_num_input.text())
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        
        self.recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num)
        
        self.stop_event = threading.Event()
        
        self.logger.log(f"=======데이터 수집 시작=======\n파일 교체 주기: {self.duration}초\n반복 횟수: {self.repeat_num}\n저장 위치: {self.recordings_folder_path}\n")
        
        self.data_collector_worker = DataCollectorWorker(self.serial_port, self.samplerate, self.recordings_folder_path, self.duration, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.data_collector_thread = QThread()
        self.data_collector_worker.moveToThread(self.data_collector_thread)

        self.data_collector_worker.progress_signal.connect(self.update_progress)
        self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
        self.data_collector_worker.log_signal.connect(self.logger.log)
        self.data_collector_worker.finished_signal.connect(self.collection_finished)
        
        self.data_collector_thread.started.connect(self.data_collector_worker.run)
        self.data_collector_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.progress_bar.setMaximum(self.duration)
        self.progress_bar.setValue(0)
        self.total_progress_bar.setMaximum(self.repeat_num)
        self.total_progress_bar.setValue(0)
        
    def stop_collection(self):
        self.stop_event.set()
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.logger.log("데이터 수집이 중지되었습니다.")
    
    def collection_finished(self):
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorApp()
    ex.show()
    sys.exit(app.exec_())