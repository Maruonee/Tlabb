import os  # 운영 체제 모듈 가져오기
import threading  # 스레딩 모듈 가져오기
from datetime import datetime  # datetime 모듈에서 datetime 가져오기
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout  # PyQt5 위젯 및 레이아웃 클래스 가져오기
from PyQt5.QtCore import QThread, QTimer, pyqtSignal  # PyQt5 코어 클래스 및 시그널 가져오기
from .status_widget import StatusWidget  # 사용자 정의 StatusWidget 클래스 가져오기
from .settings_widget import SettingsWidget  # 사용자 정의 SettingsWidget 클래스 가져오기
from .plot_canvas import PlotCanvas  # 사용자 정의 PlotCanvas 클래스 가져오기
from utils.logger import Logger  # 사용자 정의 Logger 클래스 가져오기
from workers.recorder_worker import RecorderWorker  # 사용자 정의 RecorderWorker 클래스 가져오기
from workers.data_collector_worker import DataCollectorWorker  # 사용자 정의 DataCollectorWorker 클래스 가져오기
from utils.folder_helper import create_folder  # 사용자 정의 create_folder 함수 가져오기

class MainWindow(QWidget):
    start_signal = pyqtSignal()  # 시작 신호 정의
    stop_signal = pyqtSignal()  # 중지 신호 정의

    def __init__(self, machine_error, tap_position, tap_voltage, tap_up, tap_down):
        super().__init__()  # 부모 클래스 초기화
        self.machine_error = machine_error  # 기계 오류 상태 설정
        self.tap_position = tap_position  # 탭 위치 설정
        self.tap_voltage = tap_voltage  # 탭 전압 설정
        self.tap_up = tap_up  # 탭 업 상태 설정
        self.tap_down = tap_down  # 탭 다운 상태 설정
        self.logger = Logger()  # Logger 인스턴스 생성
        self.timer = QTimer(self)  # QTimer 인스턴스 생성
        self.timer.timeout.connect(self.update_status)  # 타이머 타임아웃 시 update_status 메서드 호출
        self.status_visible = True  # 상태 표시 여부 설정

        self.initUI()  # UI 초기화

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')  # 윈도우 제목 설정
        self.resize(1200, 600)  # 윈도우 크기 설정

        main_layout = QHBoxLayout(self)  # 메인 레이아웃 설정

        self.status_widget = StatusWidget(self.tap_position, self.tap_voltage, self.tap_up, self.tap_down)  # StatusWidget 인스턴스 생성
        self.settings_widget = SettingsWidget()  # SettingsWidget 인스턴스 생성
        self.settings_widget.start_signal.connect(self.start_collection)  # 시작 신호 연결
        self.settings_widget.stop_signal.connect(self.stop_collection)  # 중지 신호 연결
        
        plot_layout = QVBoxLayout()  # 플롯 레이아웃 설정
        self.sound_plot = PlotCanvas(self, title="Sound")  # Sound 플롯 생성
        self.vibration_plot = PlotCanvas(self, title="Vibration")  # Vibration 플롯 생성
        self.voltage_plot = PlotCanvas(self, title="Voltage")  # Voltage 플롯 생성
        self.current_plot = PlotCanvas(self, title="Current")  # Current 플롯 생성
        plot_layout.addWidget(self.sound_plot)  # 플롯 레이아웃에 Sound 플롯 추가
        plot_layout.addWidget(self.vibration_plot)  # 플롯 레이아웃에 Vibration 플롯 추가
        plot_layout.addWidget(self.voltage_plot)  # 플롯 레이아웃에 Voltage 플롯 추가
        plot_layout.addWidget(self.current_plot)  # 플롯 레이아웃에 Current 플롯 추가

        main_layout.addLayout(self.status_widget.layout)  # 메인 레이아웃에 StatusWidget 레이아웃 추가
        main_layout.addLayout(self.settings_widget.layout)  # 메인 레이아웃에 SettingsWidget 레이아웃 추가
        main_layout.addLayout(plot_layout)  # 메인 레이아웃에 플롯 레이아웃 추가

        self.setLayout(main_layout)  # 메인 레이아웃 설정

    def start_collection(self):
        settings = self.settings_widget.get_settings()  # 설정 가져오기
        self.savedir = settings['savedir']  # 저장 디렉토리 설정
        self.duration = settings['duration']  # 지속 시간 설정
        self.baud_rate = settings['baud_rate']  # 보드 레이트 설정
        self.serial_port = settings['serial_port']  # 시리얼 포트 설정
        self.repeat_num = settings['repeat_num']  # 반복 횟수 설정
        self.exp_num = settings['exp_num']  # 실험 번호 설정
        self.exp_date = settings['exp_date']  # 실험 날짜 설정
        self.audio_samplerate = settings['audio_samplerate']  # 오디오 샘플링 레이트 설정

        if not settings['no_vibration_sensor']:  # 진동 센서가 연결되지 않은 경우
            self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')  # 진동 데이터 저장 폴더 생성
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')  # 오디오 데이터 저장 폴더 생성

        self.stop_event = threading.Event()  # 중지 이벤트 설정

        self.logger.log(f"ECOTAP Diagnosis Start\nCycle: {self.duration} sec\nRepeat: {self.repeat_num}\nData save path: {getattr(self, 'sensor_recordings_folder_path', 'Vibration sensor not connected')}\nData save path: {self.audio_recordings_folder_path}\n")  # 진단 시작 로그 기록

        if self.machine_error == 0:
            self.timer.start(500)  # 타이머 시작 (500ms 간격)
        elif self.machine_error == 1:
            self.timer.start(300)  # 타이머 시작 (300ms 간격)
        elif self.machine_error == 2:
            self.timer.start(100)  # 타이머 시작 (100ms 간격)

        self.recorder_worker = RecorderWorker(self, self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)  # RecorderWorker 인스턴스 생성
        self.recorder_thread = QThread()  # QThread 인스턴스 생성
        self.recorder_worker.moveToThread(self.recorder_thread)  # RecorderWorker를 별도의 스레드로 이동
        self.recorder_worker.progress_signal.connect(self.settings_widget.update_progress)  # 진행 상태 업데이트 신호 연결
        self.recorder_worker.total_progress_signal.connect(self.settings_widget.update_total_progress)  # 총 진행 상태 업데이트 신호 연결
        self.recorder_worker.log_signal.connect(self.logger.log)  # 로그 신호 연결
        self.recorder_worker.finished_signal.connect(self.collection_finished)  # 작업 완료 신호 연결

        if not settings['no_vibration_sensor']:
            self.data_collector_worker = DataCollectorWorker(self, self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)  # DataCollectorWorker 인스턴스 생성
            self.data_collector_thread = QThread()  # QThread 인스턴스 생성
            self.data_collector_worker.moveToThread(self.data_collector_thread)  # DataCollectorWorker를 별도의 스레드로 이동
            self.data_collector_worker.progress_signal.connect(self.settings_widget.update_progress)  # 진행 상태 업데이트 신호 연결
            self.data_collector_worker.total_progress_signal.connect(self.settings_widget.update_total_progress)  # 총 진행 상태 업데이트 신호 연결
            self.data_collector_worker.log_signal.connect(self.logger.log)  # 로그 신호 연결
            self.data_collector_worker.finished_signal.connect(self.collection_finished)  # 작업 완료 신호 연결
            self.data_collector_thread.started.connect(self.data_collector_worker.run)  # 스레드 시작 시 run 메서드 호출
            self.data_collector_thread.start()  # 스레드 시작

        self.recorder_thread.started.connect(self.recorder_worker.run)  # 스레드 시작 시 run 메서드 호출
        self.recorder_thread.start()  # 스레드 시작

        self.settings_widget.start_button.setEnabled(False)  # 시작 버튼 비활성화
        self.settings_widget.stop_button.setEnabled(True)  # 중지 버튼 활성화

        self.settings_widget.progress_bar.setMaximum(100)  # 진행 바 최대값 설정
        self.settings_widget.progress_bar.setValue(0)  # 진행 바 초기값 설정
        self.settings_widget.total_progress_bar.setMaximum(100)  # 총 진행 바 최대값 설정
        self.settings_widget.total_progress_bar.setValue(0)  # 총 진행 바 초기값 설정

    def stop_collection(self):
        self.stop_event.set()  # 중지 이벤트 설정
        if not self.settings_widget.no_vibration_sensor_checkbox.isChecked():
            self.data_collector_thread.quit()  # 데이터 수집 스레드 종료
            self.data_collector_thread.wait()  # 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 종료
        self.recorder_thread.wait()  # 스레드 종료 대기
        self.settings_widget.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.settings_widget.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.logger.log("Diagnosis stop.")  # 중지 로그 기록
        self.timer.stop()  # 타이머 중지
        self.status_widget.status_label.setText('')  # 상태 라벨 초기화

    def collection_finished(self):
        if not self.settings_widget.no_vibration_sensor_checkbox.isChecked():
            self.data_collector_thread.quit()  # 데이터 수집 스레드 종료
            self.data_collector_thread.wait()  # 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 종료
        self.recorder_thread.wait()  # 스레드 종료 대기
        self.settings_widget.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.settings_widget.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.timer.stop()  # 타이머 중지
        self.status_widget.status_label.setText('')  # 상태 라벨 초기화

    def update_status(self):
        if self.machine_error == 0:
            if self.status_visible:
                self.status_widget.status_label.setText('Normal')  # 정상 상태
            else:
                self.status_widget.status_label.setText('')
            self.status_widget.status_label.setStyleSheet('color: green; border-style: solid; font-size: 80px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_widget.status_label.setText('Predictive Maintenance Required')  # 예방 유지보수 필요 상태
            else:
                self.status_widget.status_label.setText('')
            self.status_widget.status_label.setStyleSheet('color: orange; border-style: solid; font-size: 80px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_widget.status_label.setText('Error')  # 오류 상태
            else:
                self.status_widget.status_label.setText('')
            self.status_widget.status_label.setStyleSheet('color: red; border-style: solid; font-size: 80px')  # 상태 라벨 스타일 설정

        self.status_visible = not self.status_visible  # 상태 표시 토글
