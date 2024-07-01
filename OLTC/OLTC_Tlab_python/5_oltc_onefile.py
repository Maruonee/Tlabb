import sys  # 시스템 관련 모듈
import os  # 운영체제 관련 모듈
import time  # 시간 관련 모듈
import threading  # 멀티스레딩 관련 모듈
import re  # 정규 표현식 모듈
from datetime import datetime  # 날짜와 시간 관련 모듈
import sounddevice as sd  # 사운드 녹음 모듈
import soundfile as sf  # 사운드 파일 저장 모듈
import serial  # 시리얼 포트 통신 모듈
import numpy as np  # 수치 계산 모듈
import matplotlib.pyplot as plt  # 그래프 그리기 모듈
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib을 PyQt5에 통합
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)  # PyQt5 위젯들
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer  # PyQt5 핵심 모듈

machine_error = 0  # 기계 오류 상태
tap_op = 20000 # 탭 동작횟수
tap_position = 5  # 탭 위치
tap_voltage = 0  # 탭 전압
tap_up = 0  # 탭 업 상태
tap_down = 0  # 탭 다운 상태

class Logger(QObject):
    log_signal = pyqtSignal(str)  # 로그 신호 정의

    def __init__(self):
        super().__init__()  # 부모 클래스 초기화

    def log(self, message):
        self.log_signal.emit(message)  # 로그 신호 방출

class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 진행률 신호 정의
    total_progress_signal = pyqtSignal(int, int)  # 총 진행률 신호 정의
    log_signal = pyqtSignal(str)  # 로그 신호 정의
    finished_signal = pyqtSignal()  # 완료 신호 정의

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()  # 부모 클래스 초기화
        self.duration = duration  # 녹음 시간
        self.samplerate = samplerate  # 샘플링 레이트
        self.channels = channels  # 채널 수
        self.folder_path = folder_path  # 저장 폴더 경로
        self.repeat_num = repeat_num  # 반복 횟수
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호
        self.stop_event = stop_event  # 중지 이벤트

    def run(self):
        for i in range(self.repeat_num):  # 반복 횟수 만큼 녹음 실행
            if self.stop_event.is_set():  # 중지 이벤트가 설정되면
                self.log_signal.emit("Diagnosis(sound) stop.")  # 중지 로그 출력
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 타임스탬프로 저장
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"  # 폴더 이름 생성
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')  # 파일 이름 생성
            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)  # 녹음 시작
            for second in range(self.duration):  # 녹음 시간 동안 반복
                if self.stop_event.is_set():  # 중지 이벤트가 설정되면
                    sd.stop()  # 녹음 중지
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')  # 파일 저장
                    self.log_signal.emit(f"{filename} saved.")  # 로그 출력
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)  # 진행률 업데이트
                time.sleep(1)  # 1초 대기
            if self.stop_event.is_set():  # 중지 이벤트가 설정되면
                break
            sd.wait()  # 녹음 완료 대기
            sf.write(filename, recording, self.samplerate, format='FLAC')  # 파일 저장
            self.log_signal.emit(f"{filename} saved.")  # 로그 출력
            self.progress_signal.emit(100, 100)  # 진행률 100% 설정
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행률 업데이트

            threading.Thread(target=self.plot_sound, args=(filename,)).start()  # 별도 스레드에서 그래프 그리기
        
        self.log_signal.emit("Diagnosis(sound) done.")  # 녹음 완료 로그 출력
        self.finished_signal.emit()  # 완료 신호 방출
        self.stop_event.set()  # 중지 이벤트 설정

    def plot_sound(self, filename):
        data, samplerate = sf.read(filename)  # 파일 읽기
        duration = len(data) / samplerate  # 녹음 길이 계산
        time = np.linspace(0., duration, len(data))  # 시간 배열 생성
        ex.sound_plot.ax.clear()  # 기존 그래프 지우기
        ex.sound_plot.ax.plot(time, data)  # 새 그래프 그리기
        ex.sound_plot.ax.set_title("Sound")  # 그래프 제목 설정
        ex.sound_plot.draw()  # 그래프 업데이트

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 진행률 신호 정의
    total_progress_signal = pyqtSignal(int, int)  # 총 진행률 신호 정의
    log_signal = pyqtSignal(str)  # 로그 신호 정의
    finished_signal = pyqtSignal()  # 완료 신호 정의

    def __init__(self, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()  # 부모 클래스 초기화
        self.duration = duration  # 데이터 수집 시간
        self.baud_rate = baud_rate  # 보드 레이트
        self.serial_port = serial_port  # 시리얼 포트
        self.folder_path = folder_path  # 저장 폴더 경로
        self.repeat_num = repeat_num  # 반복 횟수
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호
        self.stop_event = stop_event  # 중지 이벤트

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)  # 시리얼 포트 열기
        except serial.SerialException as e:
            self.log_signal.emit(f"Unable to open serial port: {e}")  # 시리얼 포트 오류 로그 출력
            self.finished_signal.emit()  # 완료 신호 방출
            return

        txt_file_ref, initial_filename = self.create_new_file()  # 새 파일 생성
        txt_file_ref = [txt_file_ref]  # 파일 참조 저장
        lock = threading.Lock()  # 스레드 락 생성

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)  # 파일 갱신 스레드 생성
        file_refresh_thread_obj.start()  # 파일 갱신 스레드 시작

        try:
            while file_refresh_thread_obj.is_alive():  # 파일 갱신 스레드가 살아있는 동안
                if ser.in_waiting > 0:  # 시리얼 포트에 데이터가 있을 경우
                    data = ser.readline().decode('utf-8').strip()  # 데이터 읽기
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')  # 데이터 파일에 쓰기
                            txt_file_ref[0].flush()  # 버퍼 비우기
        except KeyboardInterrupt:
            self.log_signal.emit("Diagnosis(Vibration) stop.")  # 중지 로그 출력
        finally:
            with lock:
                txt_file_ref[0].close()  # 파일 닫기
            ser.close()  # 시리얼 포트 닫기
            self.finished_signal.emit()  # 완료 신호 방출

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 타임스탬프로 저장
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"  # 폴더 이름 생성
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')  # 파일 이름 생성
        return open(filename, mode='w'), filename  # 파일 열기

    def file_refresh_thread(self, txt_file_ref, lock):
        for i in range(self.repeat_num):  # 반복 횟수 만큼 파일 갱신 실행
            for _ in range(self.duration):  # 데이터 수집 시간 동안 반복
                if self.stop_event.is_set():  # 중지 이벤트가 설정되면
                    return
                time.sleep(1)  # 1초 대기
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100)  # 진행률 업데이트
            new_file, filename = self.create_new_file()  # 새 파일 생성
            with lock:
                txt_file_ref[0].close()  # 이전 파일 닫기
                txt_file_ref[0] = new_file  # 새 파일 참조 저장
            self.log_signal.emit(f"{filename} saved.")  # 파일 저장 로그 출력
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행률 업데이트

            threading.Thread(target=self.plot_vibration, args=(filename,)).start()  # 별도 스레드에서 그래프 그리기
        
        self.log_signal.emit("Diagnosis(Vibration) Done.")  # 데이터 수집 완료 로그 출력
        self.finished_signal.emit()  # 완료 신호 방출

    def plot_vibration(self, filename):
        vr1_values = []
        vr2_values = []
        with open(filename, 'r') as file:
            pattern = re.compile(r"VR1\s*:\s*(\d+)\s*VR2\s*:\s*(\d+)")
            extracted_data = [pattern.search(line).groups() for line in file if pattern.search(line)]  # 정규 표현식을 사용해 데이터 추출
            vr1_values = [int(vr1) for vr1, vr2 in extracted_data]  # VR1 값 추출
            vr2_values = [int(vr2) for vr1, vr2 in extracted_data]  # VR2 값 추출
            x_values = list(range(1, len(vr1_values) + 1))  # x축 값 생성
        ex.vibration_plot.ax.clear()  # 기존 그래프 지우기
        ex.vibration_plot.ax.plot(x_values, vr1_values, label="VR1")  # VR1 그래프 그리기
        ex.vibration_plot.ax.plot(x_values, vr2_values, label="VR2")  # VR2 그래프 그리기
        ex.vibration_plot.ax.set_title("Vibration")  # 그래프 제목 설정
        ex.vibration_plot.ax.legend()  # 범례 추가
        ex.vibration_plot.draw()  # 그래프 업데이트

def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"  # 폴더 이름 생성
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)  # 폴더 경로 생성
    os.makedirs(recordings_folder_path, exist_ok=True)  # 폴더 생성
    return recordings_folder_path  # 폴더 경로 반환

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, title=""):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)  # 그래프 설정
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)  # 부모 위젯 설정
        self.ax.set_title(title)  # 그래프 제목 설정
        self.ax.plot([])  # 빈 그래프 초기화

class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.initUI()  # UI 초기화
        self.logger = Logger()  # 로그 객체 생성
        self.machine_error = machine_error  # 기계 오류 상태 설정
        self.logger.log_signal.connect(self.update_log)  # 로그 업데이트 연결
        self.timer = QTimer(self)  # 타이머 객체 생성
        self.timer.timeout.connect(self.update_status)  # 타이머 연결
        self.status_visible = True  # 상태 가시성 설정

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')  # 윈도우 제목 설정
        self.resize(1800, 1000)  # 윈도우 크기 설정

        main_layout = QHBoxLayout(self)  # 메인 레이아웃 설정

        left_panel_layout = QVBoxLayout()  # 좌측 패널 레이아웃 설정
        left_panel_layout.setSpacing(10)  # 패널 간격 설정

        # Diagnosis Results 프레임을 좌측 패널로 이동
        status_frame = QGroupBox('Diagnosis Results')  # 상태 프레임 설정
        status_frame.setStyleSheet('background-color: white')  # 배경 색상 설정
        self.status_label = QLabel('ECOTAP\nDiagnosis\nStatus')  # 상태 라벨 설정
        self.status_label.setAlignment(Qt.AlignCenter)  # 상태 라벨 정렬 설정
        self.status_label.setStyleSheet('font-size: 100px')  # 상태 라벨 폰트 크기 설정
        status_frame_layout = QVBoxLayout()  # 상태 프레임 레이아웃 설정
        status_frame_layout.addWidget(self.status_label)  # 상태 프레임 레이아웃에 상태 라벨 추가
        status_frame.setLayout(status_frame_layout)  # 상태 프레임에 레이아웃 설정
        left_panel_layout.addWidget(status_frame)  # 좌측 패널 레이아웃에 상태 프레임 추가

        # ECOTAP Status 프레임을 좌측 패널로 이동
        ecotap_status_frame = QGroupBox('ECOTAP Status')  # ECOTAP 상태 프레임 설정
        ecotap_status_layout = QVBoxLayout()  
        self.tap_op_label = QLabel(f'Tap Operations: {tap_op}')  # 탭 위치 라벨 설정
        ecotap_status_layout.addWidget(self.tap_op_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_position_label = QLabel(f'Tap position: {tap_position}')  # 탭 위치 라벨 설정
        ecotap_status_layout.addWidget(self.tap_position_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_voltage_label = QLabel(f'Tap voltage: {tap_voltage}')  # 탭 전압 라벨 설정
        ecotap_status_layout.addWidget(self.tap_voltage_label)  # 탭 전압 라벨 추가
        self.tap_up_button = QPushButton('Tap Up', self)  # 탭 업 버튼 설정
        self.tap_up_button.clicked.connect(self.tap_up_action)  # 버튼 클릭 연결
        ecotap_status_layout.addWidget(self.tap_up_button)  # 탭 업 버튼 추가
        self.tap_down_button = QPushButton('Tap Down', self)  # 탭 다운 버튼 설정
        self.tap_down_button.clicked.connect(self.tap_down_action)  # 버튼 클릭 연결
        ecotap_status_layout.addWidget(self.tap_down_button)  # 탭 다운 버튼 추가
        ecotap_status_frame.setLayout(ecotap_status_layout)  # ECOTAP 상태 프레임에 레이아웃 설정
        left_panel_layout.addWidget(ecotap_status_frame)  # 좌측 패널 레이아웃에 ECOTAP 상태 프레임 추가

        mid_panel_layout = QVBoxLayout()
        mid_panel_layout.setSpacing(10)  # 패널 간격 설정

        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')  # 기본 저장 경로 설정
        self.savedir_label = QLabel('Folder path')  # 저장 경로 라벨 설정
        self.savedir_input = QLineEdit(self)  # 저장 경로 입력란 설정
        self.savedir_input.setText(desktop_path)  # 기본 저장 경로 입력란에 설정
        self.savedir_button = QPushButton('Browse', self)  # 저장 경로 버튼 설정
        self.savedir_button.setFixedWidth(self.savedir_button.fontMetrics().width('Browse') + 20)  # 버튼 길이 설정
        self.savedir_button.clicked.connect(self.browse_folder)  # 버튼 클릭 연결
        savedir_layout = QHBoxLayout()  # 저장 경로 레이아웃 설정
        savedir_layout.addWidget(self.savedir_input)  # 저장 경로 레이아웃에 입력란 추가
        savedir_layout.addWidget(self.savedir_button)  # 저장 경로 레이아웃에 버튼 추가
        mid_panel_layout.addWidget(self.savedir_label)  # 저장 경로 레이아웃에 라벨 추가
        mid_panel_layout.addLayout(savedir_layout)  # 저장 경로 레이아웃 추가

        vibration_frame = QGroupBox('Vibration')  # 진동 설정 프레임 설정
        vibration_layout = QHBoxLayout()  # 진동 설정 레이아웃 설정
        self.serial_port_label = QLabel('COM Port')  # 시리얼 포트 라벨 설정
        self.serial_port_input = QLineEdit(self)  # 시리얼 포트 입력란 설정
        self.serial_port_input.setText('COM7')  # 기본 시리얼 포트 설정
        self.baud_rate_label = QLabel('Baud Rate')  # 보드 레이트 라벨 설정
        self.baud_rate_input = QLineEdit(self)  # 보드 레이트 입력란 설정
        self.baud_rate_input.setText('19200')  # 기본 보드 레이트 설정
        self.no_vibration_sensor_checkbox = QCheckBox("Not connected")  # 진동 센서 미연결 체크박스 설정
        self.no_vibration_sensor_checkbox.stateChanged.connect(self.toggle_vibration_sensor)  # 체크박스 상태 변경 연결
        vibration_layout.addWidget(self.serial_port_label)  # 진동 설정 레이아웃에 라벨 추가
        vibration_layout.addWidget(self.serial_port_input)  # 진동 설정 레이아웃에 입력란 추가
        vibration_layout.addWidget(self.baud_rate_label)  # 진동 설정 레이아웃에 라벨 추가
        vibration_layout.addWidget(self.baud_rate_input)  # 진동 설정 레이아웃에 입력란 추가
        vibration_layout.addWidget(self.no_vibration_sensor_checkbox)  # 진동 설정 레이아웃에 체크박스 추가
        vibration_frame.setLayout(vibration_layout)  # 진동 설정 프레임에 레이아웃 설정
        mid_panel_layout.addWidget(vibration_frame)  # 좌측 패널 레이아웃에 진동 설정 프레임 추가

        audio_frame = QGroupBox('Sound')  # 사운드 설정 프레임 설정
        audio_layout = QHBoxLayout()  # 사운드 설정 레이아웃 설정
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')  # 샘플링 레이트 라벨 설정
        self.audio_samplerate_input = QLineEdit(self)  # 샘플링 레이트 입력란 설정
        self.audio_samplerate_input.setText('44100')  # 기본 샘플링 레이트 설정
        audio_layout.addWidget(self.audio_samplerate_label)  # 사운드 설정 레이아웃에 라벨 추가
        audio_layout.addWidget(self.audio_samplerate_input)  # 사운드 설정 레이아웃에 입력란 추가
        audio_frame.setLayout(audio_layout)  # 사운드 설정 프레임에 레이아웃 설정
        mid_panel_layout.addWidget(audio_frame)  # 좌측 패널 레이아웃에 사운드 설정 프레임 추가

        exp_frame = QGroupBox('Data name')  # 데이터 이름 설정 프레임 설정
        exp_layout = QHBoxLayout()  # 데이터 이름 설정 레이아웃 설정
        self.exp_date_label = QLabel('YYMMDD')  # 날짜 라벨 설정
        self.exp_date_input = QLineEdit(self)  # 날짜 입력란 설정
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))  # 기본 날짜 설정
        self.exp_num_label = QLabel('Number')  # 번호 라벨 설정
        self.exp_num_input = QLineEdit(self)  # 번호 입력란 설정
        self.exp_num_input.setText('1')  # 기본 번호 설정
        exp_layout.addWidget(self.exp_date_label)  # 데이터 이름 설정 레이아웃에 라벨 추가
        exp_layout.addWidget(self.exp_date_input)  # 데이터 이름 설정 레이아웃에 입력란 추가
        exp_layout.addWidget(self.exp_num_label)  # 데이터 이름 설정 레이아웃에 라벨 추가
        exp_layout.addWidget(self.exp_num_input)  # 데이터 이름 설정 레이아웃에 입력란 추가
        exp_frame.setLayout(exp_layout)  # 데이터 이름 설정 프레임에 레이아웃 설정
        mid_panel_layout.addWidget(exp_frame)  # 좌측 패널 레이아웃에 데이터 이름 설정 프레임 추가
        
        duration_frame = QGroupBox('Diagnosis setup')  # 진단 설정 프레임 설정
        duration_layout = QHBoxLayout()  # 진단 설정 레이아웃 설정
        self.duration_label = QLabel('Sec')  # 주기 라벨 설정
        self.duration_input = QLineEdit(self)  # 주기 입력란 설정
        self.duration_input.setText('60')  # 기본 주기 설정
        self.minute_checkbox = QCheckBox("1min /")  # 1분 체크박스 설정
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)  # 체크박스 상태 변경 연결
        self.repeat_num_label = QLabel('Rep')  # 반복 횟수 라벨 설정
        self.repeat_num_input = QLineEdit(self)  # 반복 횟수 입력란 설정
        self.repeat_num_input.setText('60')  # 기본 반복 횟수 설정
        self.month_checkbox = QCheckBox("month")  # 1개월 체크박스 설정
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)  # 체크박스 상태 변경 연결
        duration_layout.addWidget(self.duration_label)  # 진단 설정 레이아웃에 라벨 추가
        duration_layout.addWidget(self.duration_input)  # 진단 설정 레이아웃에 입력란 추가
        duration_layout.addWidget(self.minute_checkbox)  # 진단 설정 레이아웃에 체크박스 추가
        duration_layout.addWidget(self.repeat_num_label)  # 진단 설정 레이아웃에 라벨 추가
        duration_layout.addWidget(self.repeat_num_input)  # 진단 설정 레이아웃에 입력란 추가
        duration_layout.addWidget(self.month_checkbox)  # 진단 설정 레이아웃에 체크박스 추가
        duration_frame.setLayout(duration_layout)  # 진단 설정 프레임에 레이아웃 설정
        mid_panel_layout.addWidget(duration_frame)  # 좌측 패널 레이아웃에 진단 설정 프레임 추가
        
        progress_frame = QGroupBox('Progress')  # 진행률 프레임 설정
        progress_layout = QVBoxLayout()  # 진행률 레이아웃 설정
        self.progress_label = QLabel('Cycle')  # 주기 라벨 설정
        self.progress_bar = QProgressBar(self)  # 주기 진행률 바 설정
        self.progress_bar.setMinimum(0)  # 진행률 바 최소값 설정
        self.progress_bar.setValue(0)  # 진행률 바 초기값 설정
        self.total_progress_label = QLabel('Repeat')  # 반복 라벨 설정
        self.total_progress_bar = QProgressBar(self)  # 총 진행률 바 설정
        self.total_progress_bar.setMinimum(0)  # 총 진행률 바 최소값 설정
        self.total_progress_bar.setValue(0)  # 총 진행률 바 초기값 설정
        progress_layout.addWidget(self.progress_label)  # 진행률 레이아웃에 라벨 추가
        progress_layout.addWidget(self.progress_bar)  # 진행률 레이아웃에 진행률 바 추가
        progress_layout.addWidget(self.total_progress_label)  # 진행률 레이아웃에 라벨 추가
        progress_layout.addWidget(self.total_progress_bar)  # 진행률 레이아웃에 총 진행률 바 추가
        progress_frame.setLayout(progress_layout)  # 진행률 프레임에 레이아웃 설정
        mid_panel_layout.addWidget(progress_frame)  # 좌측 패널 레이아웃에 진행률 프레임 추가

        button_layout = QHBoxLayout()  # 버튼 레이아웃 설정
        self.start_button = QPushButton('Start', self)  # 시작 버튼 설정
        self.start_button.clicked.connect(self.start_collection)  # 버튼 클릭 연결
        self.stop_button = QPushButton('Stop', self)  # 중지 버튼 설정
        self.stop_button.clicked.connect(self.stop_collection)  # 버튼 클릭 연결
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        button_layout.addWidget(self.start_button)  # 버튼 레이아웃에 시작 버튼 추가
        button_layout.addWidget(self.stop_button)  # 버튼 레이아웃에 중지 버튼 추가
        mid_panel_layout.addLayout(button_layout)  # 중간 패널 레이아웃에 버튼 레이아웃 추가

        self.log_output = QTextEdit(self)  # 로그 출력란 설정
        self.log_output.setReadOnly(True)  # 로그 출력란 읽기 전용 설정
        mid_panel_layout.addWidget(self.log_output)  # 좌측 패널 레이아웃에 로그 출력란 추가

        plot_layout = QVBoxLayout()  # 그래프 레이아웃 설정
        plot_layout.setSpacing(10)  # 패널 간격 설정

        self.sound_plot = PlotCanvas(self, title="Sound")  # 사운드 그래프 설정
        self.vibration_plot = PlotCanvas(self, title="Vibration")  # 진동 그래프 설정
        self.voltage_plot = PlotCanvas(self, title="Voltage")  # 전압 그래프 설정
        self.current_plot = PlotCanvas(self, title="Current")  # 전류 그래프 설정

        plot_layout.addWidget(self.sound_plot)  # 그래프 레이아웃에 사운드 그래프 추가
        plot_layout.addWidget(self.vibration_plot)  # 그래프 레이아웃에 진동 그래프 추가
        plot_layout.addWidget(self.voltage_plot)  # 그래프 레이아웃에 전압 그래프 추가
        plot_layout.addWidget(self.current_plot)  # 그래프 레이아웃에 전류 그래프 추가

        # 동일한 너비로 설정
        main_layout.addLayout(left_panel_layout, stretch=1)
        main_layout.addLayout(mid_panel_layout, stretch=1)
        main_layout.addLayout(plot_layout, stretch=1)

        self.setLayout(main_layout)  # 메인 레이아웃 설정

    def tap_up_action(self):
        global tap_up
        tap_up = 999  # 탭 업 상태 설정
        self.logger.log(f"Tap up pressed.")  # 로그 출력

    def tap_down_action(self):
        global tap_down
        tap_down = 999  # 탭 다운 상태 설정
        self.logger.log(f"Tap down pressed.")  # 로그 출력

    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():  # 1분 체크박스가 체크되면
            self.duration_input.setText('60')  # 주기를 60초로 설정
            self.duration_input.setEnabled(False)  # 입력란 비활성화
        else:
            self.duration_input.setEnabled(True)  # 입력란 활성화

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():  # 1개월 체크박스가 체크되면
            self.repeat_num_input.setText('43200')  # 반복 횟수를 43200으로 설정
            self.repeat_num_input.setEnabled(False)  # 입력란 비활성화
        else:
            self.repeat_num_input.setEnabled(True)  # 입력란 활성화

    def toggle_vibration_sensor(self):
        if self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서 미연결 체크박스가 체크되면
            self.serial_port_input.setEnabled(False)  # 시리얼 포트 입력란 비활성화
            self.baud_rate_input.setEnabled(False)  # 보드 레이트 입력란 비활성화
        else:
            self.serial_port_input.setEnabled(True)  # 시리얼 포트 입력란 활성화
            self.baud_rate_input.setEnabled(True)  # 보드 레이트 입력란 활성화

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')  # 디렉토리 선택 대화상자 열기
        if folder:
            self.savedir_input.setText(folder)  # 선택한 디렉토리 경로 설정

    def start_collection(self):
        self.savedir = self.savedir_input.text()  # 저장 경로 설정
        self.duration = int(self.duration_input.text())  # 주기 설정
        self.baud_rate = int(self.baud_rate_input.text())  # 보드 레이트 설정
        self.serial_port = self.serial_port_input.text()  # 시리얼 포트 설정
        self.repeat_num = int(self.repeat_num_input.text())  # 반복 횟수 설정
        self.exp_num = int(self.exp_num_input.text())  # 실험 번호 설정
        self.exp_date = self.exp_date_input.text()  # 실험 날짜 설정
        self.audio_samplerate = int(self.audio_samplerate_input.text())  # 샘플링 레이트 설정

        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')  # 센서 데이터 저장 폴더 생성
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')  # 사운드 데이터 저장 폴더 생성

        self.stop_event = threading.Event()  # 중지 이벤트 설정

        self.logger.log(f"ECOTAP Diagnosis Start\nCycle: {self.duration} sec\nRepeat: {self.repeat_num}\nData save path: {getattr(self, 'sensor_recordings_folder_path', 'Vibration sensor not connected')}\nData save path: {self.audio_recordings_folder_path}\n")  # 시작 로그 출력

        if self.machine_error == 0:
            self.timer.start(500)  # 타이머 시작 (0.5초 간격)
        elif self.machine_error == 1:
            self.timer.start(300)  # 타이머 시작 (0.3초 간격)
        elif self.machine_error == 2:
            self.timer.start(100)  # 타이머 시작 (0.1초 간격)

        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)  # 녹음 작업자 설정
        self.recorder_thread = QThread()  # 녹음 스레드 설정
        self.recorder_worker.moveToThread(self.recorder_thread)  # 녹음 작업자를 스레드로 이동
        self.recorder_worker.progress_signal.connect(self.update_progress)  # 진행률 업데이트 연결
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)  # 총 진행률 업데이트 연결
        self.recorder_worker.log_signal.connect(self.logger.log)  # 로그 업데이트 연결
        self.recorder_worker.finished_signal.connect(self.collection_finished)  # 완료 신호 연결

        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.data_collector_worker = DataCollectorWorker(self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)  # 데이터 수집 작업자 설정
            self.data_collector_thread = QThread()  # 데이터 수집 스레드 설정
            self.data_collector_worker.moveToThread(self.data_collector_thread)  # 데이터 수집 작업자를 스레드로 이동
            self.data_collector_worker.progress_signal.connect(self.update_progress)  # 진행률 업데이트 연결
            self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)  # 총 진행률 업데이트 연결
            self.data_collector_worker.log_signal.connect(self.logger.log)  # 로그 업데이트 연결
            self.data_collector_worker.finished_signal.connect(self.collection_finished)  # 완료 신호 연결
            self.data_collector_thread.started.connect(self.data_collector_worker.run)  # 스레드 시작 시 실행할 메서드 연결
            self.data_collector_thread.start()  # 데이터 수집 스레드 시작

        self.recorder_thread.started.connect(self.recorder_worker.run)  # 스레드 시작 시 실행할 메서드 연결
        self.recorder_thread.start()  # 녹음 스레드 시작

        self.start_button.setEnabled(False)  # 시작 버튼 비활성화
        self.stop_button.setEnabled(True)  # 중지 버튼 활성화

        self.progress_bar.setMaximum(100)  # 진행률 바 최대값 설정
        self.progress_bar.setValue(0)  # 진행률 바 초기값 설정
        self.total_progress_bar.setMaximum(100)  # 총 진행률 바 최대값 설정
        self.total_progress_bar.setValue(0)  # 총 진행률 바 초기값 설정

    def stop_collection(self):
        self.stop_event.set()  # 중지 이벤트 설정
        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.data_collector_thread.quit()  # 데이터 수집 스레드 중지
            self.data_collector_thread.wait()  # 데이터 수집 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 중지
        self.recorder_thread.wait()  # 녹음 스레드 종료 대기
        self.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.logger.log("Diagnosis stop.")  # 중지 로그 출력
        self.timer.stop()  # 타이머 중지
        self.status_label.setText('')  # 상태 라벨 초기화

    def collection_finished(self):
        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.data_collector_thread.quit()  # 데이터 수집 스레드 중지
            self.data_collector_thread.wait()  # 데이터 수집 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 중지
        self.recorder_thread.wait()  # 녹음 스레드 종료 대기
        self.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.timer.stop()  # 타이머 중지
        self.status_label.setText('')  # 상태 라벨 초기화

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)  # 진행률 바 최대값 설정
        self.progress_bar.setValue(value)  # 진행률 바 값 설정

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)  # 총 진행률 바 최대값 설정
        self.total_progress_bar.setValue(value)  # 총 진행률 바 값 설정

    def update_log(self, message):
        self.log_output.append(message)  # 로그 출력란에 로그 추가

    def update_status(self):
        if self.machine_error == 0:
            if self.status_visible:
                self.status_label.setText('Normal')  # 상태 라벨에 "Normal" 설정
            else:
                self.status_label.setText('')  # 상태 라벨 초기화
            self.status_label.setStyleSheet('color: green; font-size: 120px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_label.setText('Maintenance\nRequired')  # 상태 라벨에 "Predictive Maintenance Required" 설정
            else:
                self.status_label.setText('')  # 상태 라벨 초기화
            self.status_label.setStyleSheet('color: orange; font-size: 80px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_label.setText('Error')  # 상태 라벨에 "Error" 설정
            else:
                self.status_label.setText('')  # 상태 라벨 초기화
            self.status_label.setStyleSheet('color: red; font-size: 120px')  # 상태 라벨 스타일 설정

        self.status_visible = not self.status_visible  # 상태 가시성 토글

if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 객체 생성
    ex = DataCollectorApp(machine_error)  # DataCollectorApp 객체 생성
    ex.show()  # 앱 창 표시
    sys.exit(app.exec_())  # 앱 실행 및 종료
