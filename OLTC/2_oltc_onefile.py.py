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

# 전역 변수 설정
machine_error = 0  # 기계 오류 상태
tap_position = 5  # 초기 탭 위치
tap_voltage = 0  # 초기 탭 전압
tap_up = 0  # 초기 탭 업 상태
tap_down = 0  # 초기 탭 다운 상태

# Logger 클래스 정의
class Logger(QObject):
    log_signal = pyqtSignal(str)  # 로그 메시지 전송을 위한 시그널 정의

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)  # 로그 메시지를 시그널로 방출

# RecorderWorker 클래스 정의
class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 녹음 진행 상태 시그널
    total_progress_signal = pyqtSignal(int, int)  # 총 녹음 진행 상태 시그널
    log_signal = pyqtSignal(str)  # 로그 시그널
    finished_signal = pyqtSignal()  # 작업 완료 시그널

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration  # 녹음 지속 시간
        self.samplerate = samplerate  # 샘플링 레이트
        self.channels = channels  # 채널 수
        self.folder_path = folder_path  # 파일 저장 경로
        self.repeat_num = repeat_num  # 반복 횟수
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호
        self.stop_event = stop_event  # 작업 중지 이벤트

    def run(self):
        for i in range(self.repeat_num):
            if self.stop_event.is_set():
                self.log_signal.emit("Diagnosis(sound) stop.")
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간 타임스탬프 생성
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"  # 폴더 이름 생성
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')  # 파일 이름 생성
            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)  # 녹음 시작
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()  # 녹음 중지
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')  # 중간 파일 저장
                    self.log_signal.emit(f"{filename} saved.")
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)  # 진행 상태 업데이트
                time.sleep(1)  # 1초 대기
            if self.stop_event.is_set():
                break
            sd.wait()  # 녹음 완료 대기
            sf.write(filename, recording, self.samplerate, format='FLAC')  # 녹음 파일 저장
            self.log_signal.emit(f"{filename} saved.")
            self.progress_signal.emit(100, 100)  # 진행 상태 완료로 업데이트
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행 상태 업데이트
        
        self.log_signal.emit("Diagnosis(sound) done.")
        self.finished_signal.emit()  # 작업 완료 시그널 전송
        self.stop_event.set()  # 중지 이벤트 설정

# DataCollectorWorker 클래스 정의
class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 데이터 수집 진행 상태 시그널
    total_progress_signal = pyqtSignal(int, int)  # 총 데이터 수집 진행 상태 시그널
    log_signal = pyqtSignal(str)  # 로그 시그널
    finished_signal = pyqtSignal()  # 작업 완료 시그널

    def __init__(self, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration  # 데이터 수집 지속 시간
        self.baud_rate = baud_rate  # 시리얼 통신 속도
        self.serial_port = serial_port  # 시리얼 포트
        self.folder_path = folder_path  # 파일 저장 경로
        self.repeat_num = repeat_num  # 반복 횟수
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호
        self.stop_event = stop_event  # 작업 중지 이벤트

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)  # 시리얼 포트 열기
        except serial.SerialException as e:
            self.log_signal.emit(f"Unable to open serial port: {e}")  # 오류 메시지 로그
            self.finished_signal.emit()
            return

        txt_file_ref, initial_filename = self.create_new_file()  # 새로운 파일 생성
        txt_file_ref = [txt_file_ref]
        lock = threading.Lock()  # 스레드 안전을 위한 락 생성

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)  # 파일 갱신 스레드 생성
        file_refresh_thread_obj.start()

        try:
            while file_refresh_thread_obj.is_alive():
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()  # 시리얼 데이터 읽기
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')  # 파일에 데이터 쓰기
                            txt_file_ref[0].flush()
        except KeyboardInterrupt:
            self.log_signal.emit("Diagnosis(Vibration) stop.")
        finally:
            with lock:
                txt_file_ref[0].close()  # 파일 닫기
            ser.close()  # 시리얼 포트 닫기
            self.finished_signal.emit()

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간 타임스탬프 생성
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"  # 폴더 이름 생성
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')  # 파일 이름 생성
        return open(filename, mode='w'), filename  # 파일 열기 및 파일 이름 반환

    def file_refresh_thread(self, txt_file_ref, lock):
        for i in range(self.repeat_num):
            for _ in range(self.duration):
                if self.stop_event.is_set():
                    return
                time.sleep(1)  # 1초 대기
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100)  # 진행 상태 업데이트
            new_file, filename = self.create_new_file()  # 새로운 파일 생성
            with lock:
                txt_file_ref[0].close()  # 이전 파일 닫기
                txt_file_ref[0] = new_file  # 새로운 파일 참조 업데이트
            self.log_signal.emit(f"{filename} saved.")
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행 상태 업데이트
        self.log_signal.emit("Diagnosis(Vibration) Done.")
        self.finished_signal.emit()  # 작업 완료 시그널 전송

# 폴더 생성 함수 정의
def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"  # 폴더 이름 생성
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)  # 폴더 경로 생성
    os.makedirs(recordings_folder_path, exist_ok=True)  # 폴더 생성
    return recordings_folder_path

# DataCollectorApp 클래스 정의
class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.initUI()  # UI 초기화
        self.logger = Logger()  # Logger 객체 생성
        self.machine_error = machine_error  # 기계 오류 상태 설정
        self.logger.log_signal.connect(self.update_log)  # 로그 시그널 연결
        self.timer = QTimer(self)  # 타이머 생성
        self.timer.timeout.connect(self.update_status)  # 타이머 타임아웃 시 상태 업데이트
        self.status_visible = True  # 상태 표시 여부

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')  # 창 제목 설정
        self.resize(250, 350)  # 창 크기 설정

        # 진단 결과 그룹박스 설정
        status_frame = QGroupBox('Diagnosis Results')
        status_frame.setStyleSheet('background-color: white')  # 배경 색상 설정
        status_layout = QVBoxLayout()
        self.status_label = QLabel('')  # 상태 레이블
        self.status_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬
        self.status_label.setStyleSheet('font-size: 24px')  # 글꼴 크기 설정
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)

        # 폴더 경로 설정 레이아웃
        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')
        savedir_layout = QHBoxLayout()
        self.savedir_label = QLabel('Folder path')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)  # 기본 폴더 경로 설정
        self.savedir_button = QPushButton('Browse', self)
        self.savedir_button.clicked.connect(self.browse_folder)  # 폴더 브라우징 연결
        savedir_layout.addWidget(self.savedir_label)
        savedir_layout.addWidget(self.savedir_input)
        savedir_layout.addWidget(self.savedir_button)

        # 진단 설정 그룹박스
        duration_frame = QGroupBox('Diagnosis setup')
        duration_layout = QHBoxLayout()
        self.duration_label = QLabel('Cycle(sec)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')  # 기본 진단 주기 설정
        self.minute_checkbox = QCheckBox("1 min")
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)  # 체크박스 상태 변경 연결
        self.repeat_num_label = QLabel('Repeat')
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')  # 기본 반복 횟수 설정
        self.month_checkbox = QCheckBox("1 month")
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)  # 체크박스 상태 변경 연결
        duration_layout.addWidget(self.duration_label)
        duration_layout.addWidget(self.duration_input)
        duration_layout.addWidget(self.minute_checkbox)
        duration_layout.addWidget(self.repeat_num_label)
        duration_layout.addWidget(self.repeat_num_input)
        duration_layout.addWidget(self.month_checkbox)
        duration_frame.setLayout(duration_layout)

        # 진동 센서 설정 그룹박스
        vibration_frame = QGroupBox('Vibration')
        vibration_layout = QHBoxLayout()
        self.serial_port_label = QLabel('COM Port')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM7')  # 기본 시리얼 포트 설정
        self.baud_rate_label = QLabel('Baud Rate')
        self.baud_rate_input = QLineEdit(self)
        self.baud_rate_input.setText('19200')  # 기본 통신 속도 설정
        self.no_vibration_sensor_checkbox = QCheckBox("Not connected")
        self.no_vibration_sensor_checkbox.stateChanged.connect(self.toggle_vibration_sensor)  # 체크박스 상태 변경 연결
        vibration_layout.addWidget(self.serial_port_label)
        vibration_layout.addWidget(self.serial_port_input)
        vibration_layout.addWidget(self.baud_rate_label)
        vibration_layout.addWidget(self.baud_rate_input)
        vibration_layout.addWidget(self.no_vibration_sensor_checkbox)
        vibration_frame.setLayout(vibration_layout)

        # 소리 설정 그룹박스
        audio_frame = QGroupBox('Sound')
        audio_layout = QHBoxLayout()
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')  # 기본 샘플링 레이트 설정
        audio_layout.addWidget(self.audio_samplerate_label)
        audio_layout.addWidget(self.audio_samplerate_input)
        audio_frame.setLayout(audio_layout)

        # 데이터 이름 설정 그룹박스
        exp_frame = QGroupBox('Data name')
        exp_layout = QHBoxLayout()
        self.exp_date_label = QLabel('YYMMDD')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))  # 기본 날짜 설정
        self.exp_num_label = QLabel('Number')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')  # 기본 번호 설정
        exp_layout.addWidget(self.exp_date_label)
        exp_layout.addWidget(self.exp_date_input)
        exp_layout.addWidget(self.exp_num_label)
        exp_layout.addWidget(self.exp_num_input)
        exp_frame.setLayout(exp_layout)

        # ECOTAP 상태 그룹박스
        ecotap_status_frame = QGroupBox('ECOTAP Status')
        ecotap_status_layout = QVBoxLayout()
        tap_position_layout = QHBoxLayout()
        self.tap_position_label = QLabel(f'Tap position: {tap_position}')  # 탭 위치 표시
        tap_position_layout.addWidget(self.tap_position_label)
        self.tap_up_button = QPushButton('Tap Up', self)
        self.tap_up_button.clicked.connect(self.tap_up_action)  # 탭 업 버튼 클릭 연결
        self.tap_down_button = QPushButton('Tap Down', self)
        self.tap_down_button.clicked.connect(self.tap_down_action)  # 탭 다운 버튼 클릭 연결
        tap_position_layout.addWidget(self.tap_up_button)
        tap_position_layout.addWidget(self.tap_down_button)
        ecotap_status_layout.addLayout(tap_position_layout)
        self.tap_voltage_label = QLabel(f'Tap voltage: {tap_voltage}')  # 탭 전압 표시
        ecotap_status_layout.addWidget(self.tap_voltage_label)
        ecotap_status_frame.setLayout(ecotap_status_layout)

        # 진행 상태 그룹박스
        progress_frame = QGroupBox('Progress')
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel('Cycle')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.total_progress_label = QLabel('Repeat')
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
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_collection)  # 시작 버튼 클릭 연결
        self.start_button.setFixedSize(60, 30)
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_collection)  # 중지 버튼 클릭 연결
        self.stop_button.setFixedSize(60, 30)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # 로그 출력 설정
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(status_frame)
        layout.addLayout(savedir_layout)
        layout.addWidget(duration_frame)
        layout.addWidget(vibration_frame)
        layout.addWidget(audio_frame)
        layout.addWidget(exp_frame)
        layout.addWidget(ecotap_status_frame)
        layout.addLayout(button_layout)
        layout.addWidget(progress_frame)
        layout.addWidget(self.log_output)
        self.setLayout(layout)

    # 탭 업 버튼 클릭 시 호출되는 메서드
    def tap_up_action(self):
        global tap_up
        tap_up = 999  # 탭 업 값 설정
        self.logger.log(f"Tap up pressed. tap_up is now {tap_up}")  # 로그 메시지 추가

    # 탭 다운 버튼 클릭 시 호출되는 메서드
    def tap_down_action(self):
        global tap_down
        tap_down = 999  # 탭 다운 값 설정
        self.logger.log(f"Tap down pressed. tap_down is now {tap_down}")  # 로그 메시지 추가

    # 1분 체크박스 상태 변경 시 호출되는 메서드
    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')
            self.duration_input.setEnabled(False)
        else:
            self.duration_input.setEnabled(True)

    # 1달 체크박스 상태 변경 시 호출되는 메서드
    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    # 진동 센서 연결 상태 변경 시 호출되는 메서드
    def toggle_vibration_sensor(self):
        if self.no_vibration_sensor_checkbox.isChecked():
            self.serial_port_input.setEnabled(False)
            self.baud_rate_input.setEnabled(False)
        else:
            self.serial_port_input.setEnabled(True)
            self.baud_rate_input.setEnabled(True)

    # 폴더 브라우징 메서드
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)

    # 데이터 수집 시작 메서드
    def start_collection(self):
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.baud_rate = int(self.baud_rate_input.text())
        self.serial_port = self.serial_port_input.text()
        self.repeat_num = int(self.repeat_num_input.text())
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        self.audio_samplerate = int(self.audio_samplerate_input.text())

        if not self.no_vibration_sensor_checkbox.isChecked():
            self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')
        self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')

        self.stop_event = threading.Event()

        self.logger.log(f"ECOTAP Diagnosis Start\nCycle: {self.duration} sec\nRepeat: {self.repeat_num}\nData save path: {getattr(self, 'sensor_recordings_folder_path', 'Vibration sensor not connected')}\nData save path: {self.audio_recordings_folder_path}\n")

        if self.machine_error == 0:
            self.timer.start(500)
        elif self.machine_error == 1:
            self.timer.start(300)
        elif self.machine_error == 2:
            self.timer.start(100)

        # 녹음 작업을 위한 스레드 및 워커 설정
        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)

        # 데이터 수집 작업을 위한 스레드 및 워커 설정
        if not self.no_vibration_sensor_checkbox.isChecked():
            self.data_collector_worker = DataCollectorWorker(self.duration, self.baud_rate, self.serial_port, self.sensor_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
            self.data_collector_thread = QThread()
            self.data_collector_worker.moveToThread(self.data_collector_thread)
            self.data_collector_worker.progress_signal.connect(self.update_progress)
            self.data_collector_worker.total_progress_signal.connect(self.update_total_progress)
            self.data_collector_worker.log_signal.connect(self.logger.log)
            self.data_collector_worker.finished_signal.connect(self.collection_finished)
            self.data_collector_thread.started.connect(self.data_collector_worker.run)
            self.data_collector_thread.start()

        self.recorder_thread.started.connect(self.recorder_worker.run)
        self.recorder_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.total_progress_bar.setMaximum(100)
        self.total_progress_bar.setValue(0)

    # 데이터 수집 중지 메서드
    def stop_collection(self):
        self.stop_event.set()
        if not self.no_vibration_sensor_checkbox.isChecked():
            self.data_collector_thread.quit()
            self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.logger.log("Diagnosis stop.")
        self.timer.stop()
        self.status_label.setText('')

    # 데이터 수집 완료 시 호출되는 메서드
    def collection_finished(self):
        if not self.no_vibration_sensor_checkbox.isChecked():
            self.data_collector_thread.quit()
            self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.timer.stop()
        self.status_label.setText('')

    # 진행 상태 업데이트 메서드
    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    # 총 진행 상태 업데이트 메서드
    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    # 로그 업데이트 메서드
    def update_log(self, message):
        self.log_output.append(message)

    # 상태 업데이트 메서드
    def update_status(self):
        if self.machine_error == 0:
            if self.status_visible:
                self.status_label.setText('Normal')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: green; font-size: 24px')
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_label.setText('Predictive Maintenance Required')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: orange; font-size: 24px')
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_label.setText('Error')
            else:
                self.status_label.setText('')
            self.status_label.setStyleSheet('color: red; font-size: 24px')

        self.status_visible = not self.status_visible

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorApp(machine_error)
    ex.show()
    sys.exit(app.exec_())
