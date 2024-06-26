import os  # 운영 체제 모듈 가져오기
from datetime import datetime  # datetime 모듈에서 datetime 가져오기
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)  # PyQt5 위젯 및 레이아웃 클래스 가져오기
from PyQt5.QtCore import pyqtSignal  # PyQt5 코어 클래스 및 시그널 가져오기

class SettingsWidget(QWidget):
    start_signal = pyqtSignal()  # 시작 신호 정의
    stop_signal = pyqtSignal()  # 중지 신호 정의

    def __init__(self):
        super().__init__()  # 부모 클래스 초기화
        self.initUI()  # UI 초기화

    def initUI(self):
        self.layout = QVBoxLayout()  # 레이아웃 설정

        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')  # 기본 폴더 경로 설정
        savedir_layout = QHBoxLayout()  # 폴더 경로 레이아웃 설정
        self.savedir_label = QLabel('Folder path')  # 폴더 경로 라벨 설정
        self.savedir_input = QLineEdit(self)  # 폴더 경로 입력창 설정
        self.savedir_input.setText(desktop_path)  # 입력창에 기본 경로 설정
        self.savedir_button = QPushButton('Browse')  # 폴더 탐색 버튼 설정
        self.savedir_button.clicked.connect(self.browse_folder)  # 버튼 클릭 시 폴더 탐색 액션 연결
        savedir_layout.addWidget(self.savedir_label)  # 폴더 경로 레이아웃에 라벨 추가
        savedir_layout.addWidget(self.savedir_input)  # 폴더 경로 레이아웃에 입력창 추가
        savedir_layout.addWidget(self.savedir_button)  # 폴더 경로 레이아웃에 버튼 추가
        self.layout.addLayout(savedir_layout)  # 메인 레이아웃에 폴더 경로 레이아웃 추가

        duration_frame = QGroupBox('Diagnosis setup')  # 진단 설정 프레임 설정
        duration_layout = QHBoxLayout()  # 진단 설정 레이아웃 설정
        self.duration_label = QLabel('Cycle(sec)')  # 진단 주기 라벨 설정
        self.duration_input = QLineEdit(self)  # 진단 주기 입력창 설정
        self.duration_input.setText('60')  # 입력창에 기본 값 설정
        self.minute_checkbox = QCheckBox("1 min")  # 1분 체크박스 설정
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)  # 체크박스 상태 변경 시 액션 연결
        self.repeat_num_label = QLabel('Repeat')  # 반복 횟수 라벨 설정
        self.repeat_num_input = QLineEdit(self)  # 반복 횟수 입력창 설정
        self.repeat_num_input.setText('60')  # 입력창에 기본 값 설정
        self.month_checkbox = QCheckBox("1 month")  # 1개월 체크박스 설정
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)  # 체크박스 상태 변경 시 액션 연결
        duration_layout.addWidget(self.duration_label)  # 진단 설정 레이아웃에 라벨 추가
        duration_layout.addWidget(self.duration_input)  # 진단 설정 레이아웃에 입력창 추가
        duration_layout.addWidget(self.minute_checkbox)  # 진단 설정 레이아웃에 체크박스 추가
        duration_layout.addWidget(self.repeat_num_label)  # 진단 설정 레이아웃에 라벨 추가
        duration_layout.addWidget(self.repeat_num_input)  # 진단 설정 레이아웃에 입력창 추가
        duration_layout.addWidget(self.month_checkbox)  # 진단 설정 레이아웃에 체크박스 추가
        duration_frame.setLayout(duration_layout)  # 프레임에 진단 설정 레이아웃 설정
        self.layout.addWidget(duration_frame)  # 메인 레이아웃에 프레임 추가

        vibration_frame = QGroupBox('Vibration')  # 진동 설정 프레임 설정
        vibration_layout = QHBoxLayout()  # 진동 설정 레이아웃 설정
        self.serial_port_label = QLabel('COM Port')  # 시리얼 포트 라벨 설정
        self.serial_port_input = QLineEdit(self)  # 시리얼 포트 입력창 설정
        self.serial_port_input.setText('COM7')  # 입력창에 기본 값 설정
        self.baud_rate_label = QLabel('Baud Rate')  # 보드 레이트 라벨 설정
        self.baud_rate_input = QLineEdit(self)  # 보드 레이트 입력창 설정
        self.baud_rate_input.setText('19200')  # 입력창에 기본 값 설정
        self.no_vibration_sensor_checkbox = QCheckBox("Not connected")  # 진동 센서 미연결 체크박스 설정
        self.no_vibration_sensor_checkbox.stateChanged.connect(self.toggle_vibration_sensor)  # 체크박스 상태 변경 시 액션 연결
        vibration_layout.addWidget(self.serial_port_label)  # 진동 설정 레이아웃에 라벨 추가
        vibration_layout.addWidget(self.serial_port_input)  # 진동 설정 레이아웃에 입력창 추가
        vibration_layout.addWidget(self.baud_rate_label)  # 진동 설정 레이아웃에 라벨 추가
        vibration_layout.addWidget(self.baud_rate_input)  # 진동 설정 레이아웃에 입력창 추가
        vibration_layout.addWidget(self.no_vibration_sensor_checkbox)  # 진동 설정 레이아웃에 체크박스 추가
        vibration_frame.setLayout(vibration_layout)  # 프레임에 진동 설정 레이아웃 설정
        self.layout.addWidget(vibration_frame)  # 메인 레이아웃에 프레임 추가

        audio_frame = QGroupBox('Sound')  # 오디오 설정 프레임 설정
        audio_layout = QHBoxLayout()  # 오디오 설정 레이아웃 설정
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')  # 오디오 샘플링 레이트 라벨 설정
        self.audio_samplerate_input = QLineEdit(self)  # 오디오 샘플링 레이트 입력창 설정
        self.audio_samplerate_input.setText('44100')  # 입력창에 기본 값 설정
        audio_layout.addWidget(self.audio_samplerate_label)  # 오디오 설정 레이아웃에 라벨 추가
        audio_layout.addWidget(self.audio_samplerate_input)  # 오디오 설정 레이아웃에 입력창 추가
        audio_frame.setLayout(audio_layout)  # 프레임에 오디오 설정 레이아웃 설정
        self.layout.addWidget(audio_frame)  # 메인 레이아웃에 프레임 추가

        exp_frame = QGroupBox('Data name')  # 데이터 이름 설정 프레임 설정
        exp_layout = QHBoxLayout()  # 데이터 이름 설정 레이아웃 설정
        self.exp_date_label = QLabel('YYMMDD')  # 실험 날짜 라벨 설정
        self.exp_date_input = QLineEdit(self)  # 실험 날짜 입력창 설정
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))  # 입력창에 현재 날짜 설정
        self.exp_num_label = QLabel('Number')  # 실험 번호 라벨 설정
        self.exp_num_input = QLineEdit(self)  # 실험 번호 입력창 설정
        self.exp_num_input.setText('1')  # 입력창에 기본 값 설정
        exp_layout.addWidget(self.exp_date_label)  # 데이터 이름 설정 레이아웃에 라벨 추가
        exp_layout.addWidget(self.exp_date_input)  # 데이터 이름 설정 레이아웃에 입력창 추가
        exp_layout.addWidget(self.exp_num_label)  # 데이터 이름 설정 레이아웃에 라벨 추가
        exp_layout.addWidget(self.exp_num_input)  # 데이터 이름 설정 레이아웃에 입력창 추가
        exp_frame.setLayout(exp_layout)  # 프레임에 데이터 이름 설정 레이아웃 설정
        self.layout.addWidget(exp_frame)  # 메인 레이아웃에 프레임 추가

        progress_frame = QGroupBox('Progress')  # 진행 상태 프레임 설정
        progress_layout = QVBoxLayout()  # 진행 상태 레이아웃 설정
        self.progress_label = QLabel('Cycle')  # 진행 상태 라벨 설정
        self.progress_bar = QProgressBar(self)  # 진행 상태 진행 바 설정
        self.progress_bar.setMinimum(0)  # 진행 바 최소값 설정
        self.progress_bar.setValue(0)  # 진행 바 초기값 설정
        self.total_progress_label = QLabel('Repeat')  # 총 진행 상태 라벨 설정
        self.total_progress_bar = QProgressBar(self)  # 총 진행 상태 진행 바 설정
        self.total_progress_bar.setMinimum(0)  # 총 진행 바 최소값 설정
        self.total_progress_bar.setValue(0)  # 총 진행 바 초기값 설정
        progress_layout.addWidget(self.progress_label)  # 진행 상태 레이아웃에 라벨 추가
        progress_layout.addWidget(self.progress_bar)  # 진행 상태 레이아웃에 진행 바 추가
        progress_layout.addWidget(self.total_progress_label)  # 총 진행 상태 레이아웃에 라벨 추가
        progress_layout.addWidget(self.total_progress_bar)  # 총 진행 상태 레이아웃에 진행 바 추가
        progress_frame.setLayout(progress_layout)  # 프레임에 진행 상태 레이아웃 설정
        self.layout.addWidget(progress_frame)  # 메인 레이아웃에 프레임 추가

        button_layout = QHBoxLayout()  # 버튼 레이아웃 설정
        self.start_button = QPushButton('Start')  # 시작 버튼 설정
        self.start_button.clicked.connect(self.start_collection)  # 시작 버튼 클릭 시 액션 연결
        self.start_button.setFixedSize(60, 30)  # 시작 버튼 크기 설정
        self.stop_button = QPushButton('Stop')  # 중지 버튼 설정
        self.stop_button.clicked.connect(self.stop_collection)  # 중지 버튼 클릭 시 액션 연결
        self.stop_button.setFixedSize(60, 30)  # 중지 버튼 크기 설정
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        button_layout.addWidget(self.start_button)  # 버튼 레이아웃에 시작 버튼 추가
        button_layout.addWidget(self.stop_button)  # 버튼 레이아웃에 중지 버튼 추가
        self.layout.addLayout(button_layout)  # 메인 레이아웃에 버튼 레이아웃 추가

        self.log_output = QTextEdit(self)  # 로그 출력창 설정
        self.log_output.setReadOnly(True)  # 로그 출력창 읽기 전용 설정
        self.layout.addWidget(self.log_output)  # 메인 레이아웃에 로그 출력창 추가

    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')  # 1분 체크박스가 체크된 경우 주기를 60초로 설정
            self.duration_input.setEnabled(False)  # 주기 입력창 비활성화
        else:
            self.duration_input.setEnabled(True)  # 주기 입력창 활성화

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')  # 1개월 체크박스가 체크된 경우 반복 횟수를 43200으로 설정
            self.repeat_num_input.setEnabled(False)  # 반복 횟수 입력창 비활성화
        else:
            self.repeat_num_input.setEnabled(True)  # 반복 횟수 입력창 활성화

    def toggle_vibration_sensor(self):
        if self.no_vibration_sensor_checkbox.isChecked():
            self.serial_port_input.setEnabled(False)  # 진동 센서 미연결 체크박스가 체크된 경우 시리얼 포트 입력창 비활성화
            self.baud_rate_input.setEnabled(False)  # 보드 레이트 입력창 비활성화
        else:
            self.serial_port_input.setEnabled(True)  # 시리얼 포트 입력창 활성화
            self.baud_rate_input.setEnabled(True)  # 보드 레이트 입력창 활성화

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')  # 폴더 탐색기 열기
        if folder:
            self.savedir_input.setText(folder)  # 선택된 폴더 경로 설정

    def start_collection(self):
        self.start_signal.emit()  # 시작 신호 발생

    def stop_collection(self):
        self.stop_signal.emit()  # 중지 신호 발생

    def get_settings(self):
        return {
            'savedir': self.savedir_input.text(),  # 저장 경로
            'duration': int(self.duration_input.text()),  # 주기
            'baud_rate': int(self.baud_rate_input.text()),  # 보드 레이트
            'serial_port': self.serial_port_input.text(),  # 시리얼 포트
            'repeat_num': int(self.repeat_num_input.text()),  # 반복 횟수
            'exp_num': int(self.exp_num_input.text()),  # 실험 번호
            'exp_date': self.exp_date_input.text(),  # 실험 날짜
            'audio_samplerate': int(self.audio_samplerate_input.text()),  # 오디오 샘플링 레이트
            'no_vibration_sensor': self.no_vibration_sensor_checkbox.isChecked()  # 진동 센서 연결 여부
        }

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)  # 진행 바 최대값 설정
        self.progress_bar.setValue(value)  # 진행 바 현재값 설정

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)  # 총 진행 바 최대값 설정
        self.total_progress_bar.setValue(value)  # 총 진행 바 현재값 설정
