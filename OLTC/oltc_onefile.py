#pip install pyqt5 sounddevice soundfile pyserial pyinstaller
# pyinstaller --onedir --windowed your_script.py

import sys
import os
import time
import threading
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import serial
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer

machine_error = 1 # 0 = 정상 1 = 고장예측 2 = 고장

# 로그 메시지를 UI로 전달
class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)

#음향 데이터
class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration # 녹음 시간
        self.samplerate = samplerate # 샘플레이트
        self.channels = channels # 채널 (기본 스테레오)
        self.folder_path = folder_path # 저장 위치
        self.repeat_num = repeat_num # 반복 횟수
        self.exp_date = exp_date # 실험일
        self.exp_num = exp_num # 실험 번호
        self.stop_event = stop_event # 정지

    def run(self):
        for i in range(self.repeat_num):
            if self.stop_event.is_set():
                self.log_signal.emit("음향 진단 중지")
                break
            #컴퓨터의 오늘 날짜로 불러오기
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #파일 이름 설정
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')#확장자 wav설정

            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
            
            #녹음주기 반복
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')#무손실 음원 형식
                    self.log_signal.emit(f"{filename} 저장.")
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100) #진행상황
                time.sleep(1)
            if self.stop_event.is_set():
                break
            sd.wait()
            sf.write(filename, recording, self.samplerate, format='FLAC')
            self.log_signal.emit(f"{filename} 저장.")
            self.progress_signal.emit(100, 100)
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)
        
        self.log_signal.emit("설정한 기간의 음향 진단 완료")
        self.finished_signal.emit()
        self.stop_event.set()

#진동데이터
class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration # 데이터 획득 주기
        self.baud_rate = baud_rate # 보드레이트
        self.serial_port = serial_port # 시리얼 포트번호
        self.folder_path = folder_path # 저장위치
        self.repeat_num = repeat_num # 반복 횟수
        self.exp_date = exp_date # 실험일
        self.exp_num = exp_num # 실험 번호
        self.stop_event = stop_event # 정리

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)
        except serial.SerialException as e:
            self.log_signal.emit(f"직렬 포트를 열 수 없습니다: {e}")#포트에러시(미완성)
            self.finished_signal.emit()
            return

        txt_file_ref, initial_filename = self.create_new_file()#더미 객체 initial_filename
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
            
    #파일저장
    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')
        #txt확장자로 저장
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
        #프로그램 타이틀
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')
        
        #GUI 사이즈
        self.resize(500, 700)
        
        # 고장진단상황 프레임
        status_frame = QGroupBox('고장진단상황')
        status_frame.setStyleSheet('background-color: white')
        status_layout = QVBoxLayout()
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 48px')
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        
        #저장위치
        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        
        # 고장진단 주기 및 기간 설정
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
        
        # 진동진단
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
        
        # 음향진단
        audio_frame = QGroupBox('음향 진단 설정')
        audio_layout = QVBoxLayout()
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')
        audio_layout.addWidget(self.audio_samplerate_label)
        audio_layout.addWidget(self.audio_samplerate_input)
        audio_frame.setLayout(audio_layout)
        
        # 데이터 획득 날짜 및 번호
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
        
        # 진행률
        progress_frame = QGroupBox('진행률')
        progress_layout = QVBoxLayout()
        # 주기
        self.progress_label = QLabel('고장 진단 주기')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        # 기간
        self.total_progress_label = QLabel('고장 진단 기간')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.total_progress_label)
        progress_layout.addWidget(self.total_progress_bar)
        progress_frame.setLayout(progress_layout)
        
        # 고장진단 버튼
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('고장 진단 시작', self)
        self.start_button.clicked.connect(self.start_collection)
        self.stop_button = QPushButton('고장 진단 중단', self)
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

    def toggle_minute_checkbox(self):#1분 체크박스
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')
            self.duration_input.setEnabled(False)
        else:
            self.duration_input.setEnabled(True)

    def toggle_hour_checkbox(self):#1시간 체크박스
        if self.hour_checkbox.isChecked():
            self.repeat_num_input.setText('60')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def toggle_month_checkbox(self):#1달 체크박스
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def browse_folder(self):#폴더선택 버튼
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)
    
    def start_collection(self): # 시작 맟 종료 버튼
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.baud_rate = int(self.baud_rate_input.text())
        self.serial_port = self.serial_port_input.text()
        self.repeat_num = int(self.repeat_num_input.text())
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        self.audio_samplerate = int(self.audio_samplerate_input.text())
        
        self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')
        
        self.stop_event = threading.Event()
        
        self.logger.log(f"=======ECOTAP 고장 진단 시작=======\n설정 주기: {self.duration}초\n설정 기간: {self.repeat_num}\n진동데이터 저장위치: {self.sensor_recordings_folder_path}\n음향데이터 저장위치: {self.audio_recordings_folder_path}\n")
        
        #고장진단 코드
        if self.machine_error == 0:
            self.timer.start(500)  # 0.5초 간격으로 상태 업데이트
        elif self.machine_error == 1:
            self.timer.start(300)  # 0.3초 간격으로 상태 업데이트
        elif self.machine_error == 2:
            self.timer.start(100)  # 0.1초 간격으로 상태 업데이트
        
        #음향
        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event) #2 = 스테레오
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)
        
        #진동
        self.data_collector_worker = DataCollectorWorker(self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.data_collector_thread = QThread()
        self.data_collector_worker.moveToThread(self.data_collector_thread)
        self.data_collector_worker.progress_signal.connect(self.update_progress)
        self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
        self.data_collector_worker.log_signal.connect(self.logger.log)
        self.data_collector_worker.finished_signal.connect(self.collection_finished)
         
        #각 스래드로 시작
        self.data_collector_thread.started.connect(self.data_collector_worker.run)
        self.recorder_thread.started.connect(self.recorder_worker.run)
        self.data_collector_thread.start()
        self.recorder_thread.start()

        #진행버튼
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        #진행률바
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
    
    #고장진단 상황
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