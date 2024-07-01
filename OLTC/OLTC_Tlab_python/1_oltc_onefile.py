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

# 로그 메시지를 UI로 전달하는 클래스
class Logger(QObject):
    log_signal = pyqtSignal(str) # 문자열 로그 신호를 정의합니다.

    def __init__(self):
        super().__init__()

    # 로그 메시지를 방출하는 메서드
    def log(self, message):
        self.log_signal.emit(message)

# 음향 데이터를 녹음하는 클래스
class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int) # 진행 상황을 전달하는 신호
    total_progress_signal = pyqtSignal(int, int) # 전체 진행 상황을 전달하는 신호
    log_signal = pyqtSignal(str) # 로그 메시지를 전달하는 신호
    finished_signal = pyqtSignal() # 작업 완료를 알리는 신호

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration # 녹음 시간
        self.samplerate = samplerate # 샘플레이트
        self.channels = channels # 채널 수 (기본 스테레오)
        self.folder_path = folder_path # 저장 경로
        self.repeat_num = repeat_num # 반복 횟수
        self.exp_date = exp_date # 실험 날짜
        self.exp_num = exp_num # 실험 번호
        self.stop_event = stop_event # 정지 이벤트

    def run(self):
        for i in range(self.repeat_num): # 반복 횟수만큼 루프 실행
            if self.stop_event.is_set(): # 정지 이벤트가 설정되었는지 확인
                self.log_signal.emit("음향 진단 중지")
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 시간을 포맷에 맞게 변환
            folder_name = f"{self.exp_date}_{self.exp_num}_sound" # 폴더 이름 설정
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav') # 파일 경로 설정

            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels) # 녹음 시작
            for second in range(self.duration): # 녹음 시간 동안 반복
                if self.stop_event.is_set(): # 정지 이벤트 확인
                    sd.stop() # 녹음 중지
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC') # 녹음된 부분까지 파일 저장
                    self.log_signal.emit(f"{filename} 저장.") # 로그 메시지 출력
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100) # 진행 상황 신호 방출
                time.sleep(1) # 1초 대기
            if self.stop_event.is_set():
                break
            sd.wait() # 녹음 완료 대기
            sf.write(filename, recording, self.samplerate, format='FLAC') # 전체 녹음 파일 저장
            self.log_signal.emit(f"{filename} 저장.") # 로그 메시지 출력
            self.progress_signal.emit(100, 100) # 진행 상황 완료 신호 방출
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100) # 전체 진행 상황 신호 방출
        
        self.log_signal.emit("설정한 기간의 음향 진단 완료") # 로그 메시지 출력
        self.finished_signal.emit() # 작업 완료 신호 방출
        self.stop_event.set() # 정지 이벤트 설정

# 진동 데이터를 수집하는 클래스
class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int) # 진행 상황을 전달하는 신호
    total_progress_signal = pyqtSignal(int, int) # 전체 진행 상황을 전달하는 신호
    log_signal = pyqtSignal(str) # 로그 메시지를 전달하는 신호
    finished_signal = pyqtSignal() # 작업 완료를 알리는 신호

    def __init__(self, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration # 데이터 획득 주기
        self.baud_rate = baud_rate # 보드레이트
        self.serial_port = serial_port # 시리얼 포트번호
        self.folder_path = folder_path # 저장 경로
        self.repeat_num = repeat_num # 반복 횟수
        self.exp_date = exp_date # 실험 날짜
        self.exp_num = exp_num # 실험 번호
        self.stop_event = stop_event # 정지 이벤트

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate) # 시리얼 포트 열기
        except serial.SerialException as e:
            self.log_signal.emit(f"직렬 포트를 열 수 없습니다: {e}") # 포트 열기 실패 시 로그 메시지 출력
            self.finished_signal.emit() # 작업 완료 신호 방출
            return

        txt_file_ref, initial_filename = self.create_new_file() # 새로운 파일 생성
        txt_file_ref = [txt_file_ref] # 파일 참조 리스트
        lock = threading.Lock() # 스레드 동기화를 위한 락 객체

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True) # 파일 갱신 스레드 생성
        file_refresh_thread_obj.start() # 파일 갱신 스레드 시작

        try:
            while file_refresh_thread_obj.is_alive(): # 파일 갱신 스레드가 살아있는 동안
                if ser.in_waiting > 0: # 시리얼 포트에 읽을 데이터가 있는지 확인
                    data = ser.readline().decode('utf-8').strip() # 데이터 읽기 및 디코딩
                    if data: # 데이터가 있으면
                        with lock: # 락을 걸고
                            txt_file_ref[0].write(f'{data}\n') # 파일에 데이터 쓰기
                            txt_file_ref[0].flush() # 파일 버퍼 플러시
        except KeyboardInterrupt:
            self.log_signal.emit("진동 진단 중지") # 로그 메시지 출력
        finally:
            with lock:
                txt_file_ref[0].close() # 파일 닫기
            ser.close() # 시리얼 포트 닫기
            self.finished_signal.emit() # 작업 완료 신호 방출

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 시간을 포맷에 맞게 변환
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors" # 폴더 이름 설정
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt') # 파일 경로 설정
        return open(filename, mode='w'), filename # 파일 열기 및 경로 반환

    def file_refresh_thread(self, txt_file_ref, lock):
        for i in range(self.repeat_num): # 반복 횟수만큼 루프 실행
            for _ in range(self.duration): # 주기 동안 루프 실행
                if self.stop_event.is_set(): # 정지 이벤트 확인
                    return
                time.sleep(1) # 1초 대기
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100) # 진행 상황 신호 방출
            new_file, filename = self.create_new_file() # 새로운 파일 생성
            with lock: # 락을 걸고
                txt_file_ref[0].close() # 기존 파일 닫기
                txt_file_ref[0] = new_file # 새로운 파일로 교체
            self.log_signal.emit(f"{filename} 저장.") # 로그 메시지 출력
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100) # 전체 진행 상황 신호 방출
        self.log_signal.emit("설정한 기간의 진동 진단 완료") # 로그 메시지 출력
        self.finished_signal.emit() # 작업 완료 신호 방출

# 폴더를 생성하는 함수
def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}" # 폴더 이름 설정
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name) # 폴더 경로 설정
    os.makedirs(recordings_folder_path, exist_ok=True) # 폴더 생성
    return recordings_folder_path # 폴더 경로 반환

# 메인 애플리케이션 클래스
class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.initUI() # UI 초기화
        self.logger = Logger() # Logger 인스턴스 생성
        self.machine_error = machine_error # 고장 상태
        self.logger.log_signal.connect(self.update_log) # 로그 신호 연결
        self.timer = QTimer(self) # 타이머 생성
        self.timer.timeout.connect(self.update_status) # 타이머 타임아웃 시 업데이트 연결
        self.status_visible = True # 상태 표시 여부

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab') # 프로그램 타이틀 설정
        self.resize(500, 700) # 윈도우 크기 설정

        # 고장진단상황 프레임
        status_frame = QGroupBox('고장진단상황')
        status_frame.setStyleSheet('background-color: white') # 배경색 설정
        status_layout = QVBoxLayout()
        self.status_label = QLabel('') # 상태 레이블
        self.status_label.setAlignment(Qt.AlignCenter) # 중앙 정렬
        self.status_label.setStyleSheet('font-size: 48px') # 글꼴 크기 설정
        status_layout.addWidget(self.status_label) # 레이아웃에 추가
        status_frame.setLayout(status_layout)

        # 저장위치 설정
        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads') # 기본 저장 위치 설정
        self.savedir_label = QLabel('저장위치') # 저장위치 레이블
        self.savedir_input = QLineEdit(self) # 저장위치 입력창
        self.savedir_input.setText(desktop_path) # 기본 경로 설정
        self.savedir_button = QPushButton('폴더선택', self) # 폴더 선택 버튼
        self.savedir_button.clicked.connect(self.browse_folder) # 버튼 클릭 시 폴더 선택 연결

        # 고장진단 주기 및 기간 설정
        duration_frame = QGroupBox('고장진단 주기 및 기간 설정')
        duration_layout = QVBoxLayout()
        self.duration_label = QLabel('고장진단 주기(초)') # 주기 레이블
        self.duration_input = QLineEdit(self) # 주기 입력창
        self.duration_input.setText('60') # 기본값 설정
        duration_hbox = QHBoxLayout()
        self.minute_checkbox = QCheckBox("1 min") # 1분 체크박스
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox) # 체크박스 상태 변경 시 연결
        duration_hbox.addWidget(self.duration_input) # 입력창 추가
        duration_hbox.addWidget(self.minute_checkbox) # 체크박스 추가
        self.repeat_num_label = QLabel('고장진단 기간(주기반복횟수)') # 기간 레이블
        repeat_num_layout = QHBoxLayout()
        self.repeat_num_input = QLineEdit(self) # 기간 입력창
        self.repeat_num_input.setText('60') # 기본값 설정
        self.hour_checkbox = QCheckBox("1 hour") # 1시간 체크박스
        self.hour_checkbox.stateChanged.connect(self.toggle_hour_checkbox) # 체크박스 상태 변경 시 연결
        self.month_checkbox = QCheckBox("1 month") # 1달 체크박스
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox) # 체크박스 상태 변경 시 연결
        repeat_num_layout.addWidget(self.repeat_num_input) # 입력창 추가
        repeat_num_layout.addWidget(self.hour_checkbox) # 체크박스 추가
        repeat_num_layout.addWidget(self.month_checkbox) # 체크박스 추가
        duration_layout.addWidget(self.duration_label) # 레이블 추가
        duration_layout.addLayout(duration_hbox) # 레이아웃 추가
        duration_layout.addWidget(self.repeat_num_label) # 레이블 추가
        duration_layout.addLayout(repeat_num_layout) # 레이아웃 추가
        duration_frame.setLayout(duration_layout)

        # 진동 진단 설정
        vibration_frame = QGroupBox('진동 진단 설정')
        vibration_layout = QVBoxLayout()
        self.serial_port_label = QLabel('COM Port') # 포트 레이블
        self.serial_port_input = QLineEdit(self) # 포트 입력창
        self.serial_port_input.setText('COM3') # 기본값 설정
        self.baud_rate_label = QLabel('Arduino Baud Rate(bps)') # 보드레이트 레이블
        self.baud_rate_input = QLineEdit(self) # 보드레이트 입력창
        self.baud_rate_input.setText('19200') # 기본값 설정
        vibration_layout.addWidget(self.serial_port_label) # 레이블 추가
        vibration_layout.addWidget(self.serial_port_input) # 입력창 추가
        vibration_layout.addWidget(self.baud_rate_label) # 레이블 추가
        vibration_layout.addWidget(self.baud_rate_input) # 입력창 추가
        vibration_frame.setLayout(vibration_layout)

        # 음향 진단 설정
        audio_frame = QGroupBox('음향 진단 설정')
        audio_layout = QVBoxLayout()
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)') # 샘플레이트 레이블
        self.audio_samplerate_input = QLineEdit(self) # 샘플레이트 입력창
        self.audio_samplerate_input.setText('44100') # 기본값 설정
        audio_layout.addWidget(self.audio_samplerate_label) # 레이블 추가
        audio_layout.addWidget(self.audio_samplerate_input) # 입력창 추가
        audio_frame.setLayout(audio_layout)

        # 데이터 획득 설정
        exp_frame = QGroupBox('데이터 획득 설정')
        exp_layout = QVBoxLayout()
        self.exp_date_label = QLabel('날짜(YYMMDD)') # 날짜 레이블
        self.exp_date_input = QLineEdit(self) # 날짜 입력창
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d')) # 기본값 설정
        self.exp_num_label = QLabel('번호') # 번호 레이블
        self.exp_num_input = QLineEdit(self) # 번호 입력창
        self.exp_num_input.setText('1') # 기본값 설정
        exp_layout.addWidget(self.exp_date_label) # 레이블 추가
        exp_layout.addWidget(self.exp_date_input) # 입력창 추가
        exp_layout.addWidget(self.exp_num_label) # 레이블 추가
        exp_layout.addWidget(self.exp_num_input) # 입력창 추가
        exp_frame.setLayout(exp_layout)

        # 진행률 설정
        progress_frame = QGroupBox('진행률')
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel('고장 진단 주기') # 주기 레이블
        self.progress_bar = QProgressBar(self) # 주기 진행 바
        self.progress_bar.setMinimum(0) # 최소값 설정
        self.progress_bar.setValue(0) # 초기값 설정
        self.total_progress_label = QLabel('고장 진단 기간') # 기간 레이블
        self.total_progress_bar = QProgressBar(self) # 기간 진행 바
        self.total_progress_bar.setMinimum(0) # 최소값 설정
        self.total_progress_bar.setValue(0) # 초기값 설정
        progress_layout.addWidget(self.progress_label) # 레이블 추가
        progress_layout.addWidget(self.progress_bar) # 진행 바 추가
        progress_layout.addWidget(self.total_progress_label) # 레이블 추가
        progress_layout.addWidget(self.total_progress_bar) # 진행 바 추가
        progress_frame.setLayout(progress_layout)

        # 고장진단 버튼 설정
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('고장 진단 시작', self) # 시작 버튼
        self.start_button.clicked.connect(self.start_collection) # 버튼 클릭 시 연결
        self.stop_button = QPushButton('고장 진단 중단', self) # 중단 버튼
        self.stop_button.clicked.connect(self.stop_collection) # 버튼 클릭 시 연결
        self.stop_button.setEnabled(False) # 초기 상태 비활성화
        button_layout.addWidget(self.start_button) # 버튼 추가
        button_layout.addWidget(self.stop_button) # 버튼 추가

        # 로그 출력창 설정
        self.log_output = QTextEdit(self) # 로그 출력창
        self.log_output.setReadOnly(True) # 읽기 전용 설정
        layout = QVBoxLayout()
        layout.addWidget(status_frame) # 상태 프레임 추가
        layout.addWidget(self.savedir_label) # 저장위치 레이블 추가
        layout.addWidget(self.savedir_input) # 저장위치 입력창 추가
        layout.addWidget(self.savedir_button) # 저장위치 버튼 추가
        layout.addWidget(duration_frame) # 주기 및 기간 프레임 추가
        layout.addWidget(vibration_frame) # 진동 프레임 추가
        layout.addWidget(audio_frame) # 음향 프레임 추가
        layout.addWidget(exp_frame) # 데이터 획득 프레임 추가
        layout.addLayout(button_layout) # 버튼 레이아웃 추가
        layout.addWidget(progress_frame) # 진행률 프레임 추가
        layout.addWidget(self.log_output) # 로그 출력창 추가
        self.setLayout(layout) # 레이아웃 설정

    # 체크박스 이벤트 핸들러
    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked(): # 체크박스 선택 시
            self.duration_input.setText('60') # 입력창 값 설정
            self.duration_input.setEnabled(False) # 입력창 비활성화
        else:
            self.duration_input.setEnabled(True) # 입력창 활성화

    def toggle_hour_checkbox(self):
        if self.hour_checkbox.isChecked(): # 체크박스 선택 시
            self.repeat_num_input.setText('60') # 입력창 값 설정
            self.repeat_num_input.setEnabled(False) # 입력창 비활성화
        else:
            self.repeat_num_input.setEnabled(True) # 입력창 활성화

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked(): # 체크박스 선택 시
            self.repeat_num_input.setText('43200') # 입력창 값 설정
            self.repeat_num_input.setEnabled(False) # 입력창 비활성화
        else:
            self.repeat_num_input.setEnabled(True) # 입력창 활성화

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory') # 폴더 선택 대화창 열기
        if folder:
            self.savedir_input.setText(folder) # 선택한 폴더 경로 설정

    def start_collection(self): # 고장 진단 시작
        self.savedir = self.savedir_input.text() # 저장 경로
        self.duration = int(self.duration_input.text()) # 주기
        self.baud_rate = int(self.baud_rate_input.text()) # 보드레이트
        self.serial_port = self.serial_port_input.text() # 시리얼 포트
        self.repeat_num = int(self.repeat_num_input.text()) # 반복 횟수
        self.exp_num = int(self.exp_num_input.text()) # 실험 번호
        self.exp_date = self.exp_date_input.text() # 실험 날짜
        self.audio_samplerate = int(self.audio_samplerate_input.text()) # 샘플레이트

        self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors') # 진동 데이터 저장 경로 생성
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound') # 음향 데이터 저장 경로 생성

        self.stop_event = threading.Event() # 정지 이벤트 생성

        self.logger.log(f"=======ECOTAP 고장 진단 시작=======\n설정 주기: {self.duration}초\n설정 기간: {self.repeat_num}\n진동데이터 저장위치: {self.sensor_recordings_folder_path}\n음향데이터 저장위치: {self.audio_recordings_folder_path}\n") # 로그 메시지 출력

        # 고장 상태에 따른 타이머 설정
        if self.machine_error == 0:
            self.timer.start(500)  # 0.5초 간격으로 상태 업데이트
        elif self.machine_error == 1:
            self.timer.start(300)  # 0.3초 간격으로 상태 업데이트
        elif self.machine_error == 2:
            self.timer.start(100)  # 0.1초 간격으로 상태 업데이트

        # 음향 진단 워커와 스레드 생성 및 연결
        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)

        # 진동 진단 워커와 스레드 생성 및 연결
        self.data_collector_worker = DataCollectorWorker(self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.data_collector_thread = QThread()
        self.data_collector_worker.moveToThread(self.data_collector_thread)
        self.data_collector_worker.progress_signal.connect(self.update_progress)
        self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
        self.data_collector_worker.log_signal.connect(self.logger.log)
        self.data_collector_worker.finished_signal.connect(self.collection_finished)

        self.data_collector_thread.started.connect(self.data_collector_worker.run) # 진동 진단 스레드 시작
        self.recorder_thread.started.connect(self.recorder_worker.run) # 음향 진단 스레드 시작
        self.data_collector_thread.start() # 진동 진단 스레드 시작
        self.recorder_thread.start() # 음향 진단 스레드 시작

        self.start_button.setEnabled(False) # 시작 버튼 비활성화
        self.stop_button.setEnabled(True) # 중단 버튼 활성화

        self.progress_bar.setMaximum(100) # 진행 바 최대값 설정
        self.progress_bar.setValue(0) # 진행 바 초기값 설정
        self.total_progress_bar.setMaximum(100) # 전체 진행 바 최대값 설정
        self.total_progress_bar.setValue(0) # 전체 진행 바 초기값 설정

    def stop_collection(self):
        self.stop_event.set() # 정지 이벤트 설정
        self.data_collector_thread.quit() # 진동 진단 스레드 종료
        self.data_collector_thread.wait() # 스레드 종료 대기
        self.recorder_thread.quit() # 음향 진단 스레드 종료
        self.recorder_thread.wait() # 스레드 종료 대기
        self.start_button.setEnabled(True) # 시작 버튼 활성화
        self.stop_button.setEnabled(False) # 중단 버튼 비활성화
        self.logger.log("고장진단이 중지되었습니다.") # 로그 메시지 출력
        self.timer.stop() # 타이머 중지
        self.status_label.setText('') # 상태 레이블 초기화

    def collection_finished(self):
        self.data_collector_thread.quit() # 진동 진단 스레드 종료
        self.data_collector_thread.wait() # 스레드 종료 대기
        self.recorder_thread.quit() # 음향 진단 스레드 종료
        self.recorder_thread.wait() # 스레드 종료 대기
        self.start_button.setEnabled(True) # 시작 버튼 활성화
        self.stop_button.setEnabled(False) # 중단 버튼 비활성화
        self.timer.stop() # 타이머 중지
        self.status_label.setText('') # 상태 레이블 초기화

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum) # 진행 바 최대값 설정
        self.progress_bar.setValue(value) # 진행 바 값 설정

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum) # 전체 진행 바 최대값 설정
        self.total_progress_bar.setValue(value) # 전체 진행 바 값 설정

    def update_log(self, message):
        self.log_output.append(message) # 로그 출력창에 메시지 추가

    def update_status(self):
        if self.machine_error == 0:
            if self.status_visible:
                self.status_label.setText('정상') # 상태 레이블 설정
            else:
                self.status_label.setText('') # 상태 레이블 초기화
            self.status_label.setStyleSheet('color: green; font-size: 48px') # 글꼴 및 색상 설정
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_label.setText('고장 예측') # 상태 레이블 설정
            else:
                self.status_label.setText('') # 상태 레이블 초기화
            self.status_label.setStyleSheet('color: orange; font-size: 48px') # 글꼴 및 색상 설정
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_label.setText('고장') # 상태 레이블 설정
            else:
                self.status_label.setText('') # 상태 레이블 초기화
            self.status_label.setStyleSheet('color: red; font-size: 48px') # 글꼴 및 색상 설정

        self.status_visible = not self.status_visible # 상태 표시 여부 토글

if __name__ == '__main__':
    app = QApplication(sys.argv) # QApplication 생성
    ex = DataCollectorApp(machine_error) # DataCollectorApp 인스턴스 생성
    ex.show() # 애플리케이션 실행
    sys.exit(app.exec_()) # 애플리케이션 종료 시 시스템 종료