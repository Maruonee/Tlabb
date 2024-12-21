#exe파일화
"""
pip install pyinstaller
pyinstaller --onefile --windowed --icon=ecotap.ico 8_oltc_onefile.py

"""


import sys  # 시스템 관련 모듈
import os  # 운영체제 관련 모듈
import threading  # 멀티스레딩 관련 모듈
import re  # 정규 표현식 모듈
from datetime import datetime  # 날짜와 시간 관련 모듈
import sounddevice as sd  # 사운드 녹음 모듈
import soundfile as sf  # 사운드 파일 저장 모듈
import numpy as np  # 수치 계산 모듈
import matplotlib.pyplot as plt  # 그래프 그리기 모듈
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib을 PyQt5에 통합
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)  # PyQt5 위젯들
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer  # PyQt5 핵심 모듈
import time
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
# from modbus_tk import modbus_tcp

machine_error = 0  # 기계 오류 상태

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
        
        self.finished_signal.emit()  # 완료 신호 방출

    def plot_sound(self, filename):
        data, samplerate = sf.read(filename)  # 파일 읽기
        duration = len(data) / samplerate  # 녹음 길이 계산
        time = np.linspace(0., duration, len(data))  # 시간 배열 생성
        ex.sound_plot.ax.clear()  # 기존 그래프 지우기
        ex.sound_plot.ax.plot(time, data)  # 새 그래프 그리기
        ex.sound_plot.ax.set_title("Sound")  # 그래프 제목 설정
        ex.sound_plot.draw()  # 그래프 업데이트

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    plot_signal = pyqtSignal(str)  # 파일 경로를 전달하는 시그널 추가

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
        self.lock = threading.Lock()
        
        # 시그널을 plot_vibration과 안전하게 연결
        self.plot_signal.connect(self.plot_vibration)

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        except serial.SerialException as e:
            self.log_signal.emit(f"Serial port error: {str(e)}")
            return

        txt_file_ref, initial_filename = self.create_new_file()
        txt_file_ref = [txt_file_ref]

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref,), daemon=True)
        file_refresh_thread_obj.start()

        try:
            while not self.stop_event.is_set():
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    if data:
                        with self.lock:
                            txt_file_ref[0].write(f'{data}\n')
                            txt_file_ref[0].flush()
        except Exception as e:
            self.log_signal.emit(f"Error during data collection: {str(e)}")
        finally:
            with self.lock:
                txt_file_ref[0].close()
            ser.close()
            self.finished_signal.emit()

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')
        return open(filename, mode='w'), filename

    def file_refresh_thread(self, txt_file_ref):
        for i in range(self.repeat_num):
            # 설정된 duration 만큼 기다리며 진행률 업데이트
            for _ in range(self.duration):
                if self.stop_event.is_set():
                    return
                time.sleep(1)
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100)

            # 설정된 duration이 지난 후 1초 지연 후에 현재 파일을 닫고 그래프에 업데이트
            time.sleep(1)  # 1초 지연

            with self.lock:
                old_file_path = txt_file_ref[0].name  # 현재 파일 경로 저장
                txt_file_ref[0].close()  # 현재 파일 닫기
                # 잠금 해제 후 시그널 방출
            self.plot_signal.emit(old_file_path)  # 이전 파일 경로로 그래프 업데이트 호출

            # 새 파일 생성
            with self.lock:
                new_file, filename = self.create_new_file()
                txt_file_ref[0] = new_file
            
            self.log_signal.emit(f"{filename} saved.")
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)

    def plot_vibration(self, filename):
        vr1_values = []
        
        def extract_vr1_values(file_path):
            vr1_values = []
            with open(file_path, 'r') as file:
                for line in file:
                    match = re.search(r'VR1\s*:\s*(\d+)', line)
                    if match:
                        vr1_value = int(match.group(1))
                        vr1_values.append(vr1_value)
            return vr1_values

        vr1_values = extract_vr1_values(filename)
        
        if vr1_values:
            ex.vibration_plot.ax.clear()
            ex.vibration_plot.ax.plot(vr1_values, label="Vibration Data", marker='o', linestyle='-', markersize=3, linewidth=0.8)
            ex.vibration_plot.ax.set_title("Vibration")
            ex.vibration_plot.ax.set_xlabel("time")
            ex.vibration_plot.ax.set_ylabel("Value")
            ex.vibration_plot.ax.legend()
            ex.vibration_plot.draw()
        else:
            self.log_signal.emit("No valid VR1 data found in file for plotting.")

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

class ModbusRTUClient:
    def __init__(self, ecotap_port, folder_path, exp_date, exp_num, ip_address='192.168.0.173', interval=0.1):
        super().__init__()
        ## 시리얼 포트 설정
        self.serial_port = serial.Serial(
            port=ecotap_port,            
            baudrate=38400,       
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE, 
            bytesize=serial.EIGHTBITS,
            timeout=0.1 
        )
        self.master = modbus_rtu.RtuMaster(self.serial_port) 
        
        ## TCP/IP 설정
        # self.master = modbus_tcp.TcpMaster(host=ip_address)
        # self.interval = interval  # 읽기 간격

        self.master.set_timeout(0.1) 
        self.master.set_verbose(True) 
        self.stop_event = threading.Event() 
        
        self.folder_path = folder_path  # 데이터 저장 폴더 경로
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)  # 폴더가 없으면 생성
        # 데이터 읽기 시작
        self.start_reading()

    def read_registers(self):
        # 홀딩 레지스터 읽기 (슬레이브 ID: 1, 시작 주소: 0, 레지스터 수: 7)
        holding_registers = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0, 7)
        # 입력 레지스터 읽기 (슬레이브 ID: 1, 시작 주소: 3, 레지스터 수: 1)
        input_registers = self.master.execute(1, cst.READ_INPUT_REGISTERS, 3, 1)
        # 데이터 형식 결정 및 출력
        tap_op = holding_registers[3]  # 탭 동작횟수
        tap_de_voltage = holding_registers[6] # 탭 원하는 전압
        tap_position = holding_registers[1]  # 탭 위치
        tap_voltage = input_registers[0] / 2  # 탭 전압        
        tap_mode_raw = holding_registers[0]
        if  tap_mode_raw == 1:
            tap_mode = "AVR AUTO"
        elif tap_mode_raw == 2:
            tap_mode = "AVR MANUAL"
        elif tap_mode_raw == 3:
            tap_mode = "EXTERNAL CONTROL"
        else:
            tap_mode = "INVALID MODE"
        return tap_op, tap_de_voltage, tap_position, tap_voltage, tap_mode

    def start_reading(self):
        self.stop_event.clear()  # 스레드를 중지시키기 위한 이벤트 초기화
        ## TCP/IP 설정
        # self.thread = threading.Thread(target=self._update_registers)
        # 시리얼 포트 설정
        self.thread = threading.Thread(target=self._update_registers)
        
        self.thread.start()

    def _update_registers(self):
        while not self.stop_event.is_set():
            tap_op, tap_de_voltage, tap_position, tap_voltage, tap_mode = self.read_registers()
            self.save_to_file(tap_op, tap_de_voltage, tap_position, tap_voltage)
            time.sleep(0.1)
        self.stop_event.set()  # 작업이 완료되면 스레드를 중지시킴

    def save_to_file(self, tap_op, tap_de_voltage, tap_position, tap_voltage):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 타임스탬프로 저장
        folder_name = f"{self.exp_date}_{self.exp_num}_ecotap"  # 폴더 이름 생성
        filename = os.path.join(self.folder_path, f'{folder_name}.txt')  # 파일 경로 생성
        with open(filename, "a") as f:
            f.write(f"{timestamp}, Desire Voltage: {tap_de_voltage}V, Tap Operations Counter: {tap_op}, Current Tap Position: {tap_position}, Current Tap Voltage: {tap_voltage}V\n")
        
    def stop_reading(self):
        self.stop_event.set()
        self.thread.join()
        # 시리얼 포트 설정
        self.serial_port.close()
        ## TCP/IP 설정
        # self.master._do_close() 
    
    def get_latest_data(self):
        return self.read_registers()

    def tap_up(self):
        # Coil address 1의 status를 1로 변경
        self.master.execute(1, cst.WRITE_SINGLE_COIL, 0, output_value=1)
        time.sleep(1)
        # 1초 후에 Coil address 1의 status를 0으로 변경
        self.master.execute(1, cst.WRITE_SINGLE_COIL, 0, output_value=0)

    def tap_down(self):
        # Coil address 0의 status를 1로 변경
        self.master.execute(1, cst.WRITE_SINGLE_COIL, 1, output_value=1)
        time.sleep(1)
        # 1초 후에 Coil address 0의 status를 0으로 변경
        self.master.execute(1, cst.WRITE_SINGLE_COIL, 1, output_value=0)

class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.tap_op = 0  # Tap Operations 초기값 설정
        self.tap_position = 0  # Tap Position 초기값 설정
        self.tap_voltage = 0  # Tap Voltage 초기값 설정
        self.tap_de_voltage = 0
        self.tap_mode = ''
        self.ecotap_port = ''  # ecotap_port 초기값 설정
        self.initUI()  # UI 초기화
        self.logger = Logger()  # 로그 객체 생성
        self.machine_error = machine_error  # 기계 오류 상태 설정
        self.logger.log_signal.connect(self.update_log)  # 로그 업데이트 연결
        self.timer = QTimer(self)  # 타이머 객체 생성
        self.timer.timeout.connect(self.update_status)  # 타이머 연결
        self.status_visible = True  # 상태 가시성 설정

        # ECOTAP 데이터 업데이트를 위한 타이머 설정
        self.ecotap_timer = QTimer(self)
        self.ecotap_timer.timeout.connect(self.update_ecotap_status)
    
    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by MICS')  # 윈도우 제목 설정
        self.resize(1500, 800)  # 윈도우 크기 설정
        
        main_layout = QHBoxLayout(self)  # 메인 레이아웃 설정
        
        left_panel_layout = QVBoxLayout()  # 좌측 패널 레이아웃 설정
        left_panel_layout.setSpacing(0)  # 패널 간격 설정

        # Diagnosis Results 프레임을 좌측 패널로 이동
        status_frame = QGroupBox('Diagnosis Results')  # 상태 프레임 설정
        status_frame.setStyleSheet('background-color: white')  # 배경 색상 설정
        self.status_label = QLabel('ECOTAP\nDiagnosis\nStatus')  # 상태 라벨 설정
        self.status_label.setAlignment(Qt.AlignCenter)  # 상태 라벨 정렬 설정
        self.status_label.setStyleSheet('font-size: 50px')  # 상태 라벨 폰트 크기 설정
        status_frame_layout = QVBoxLayout()  # 상태 프레임 레이아웃 설정
        status_frame_layout.addWidget(self.status_label)  # 상태 프레임 레이아웃에 상태 라벨 추가
        status_frame.setLayout(status_frame_layout)  # 상태 프레임에 레이아웃 설정
        left_panel_layout.addWidget(status_frame)  # 좌측 패널 레이아웃에 상태 프레임 추가

        # ECOTAP Status 프레임을 좌측 패널로 이동
        ecotap_status_frame = QGroupBox('ECOTAP Status')  # ECOTAP 상태 프레임 설정
        ecotap_status_layout = QVBoxLayout()
        self.ecotap_port_label = QLabel('ECOTAP Port')  # ecotap_port 라벨 설정
        self.ecotap_port_input = QLineEdit(self)  # ecotap_port 입력란 설정
        self.ecotap_port_input.setText('COM20')  # 기본 시리얼 포트 설정
        ecotap_status_layout.addWidget(self.ecotap_port_label)  # ecotap_port 라벨 추가
        ecotap_status_layout.addWidget(self.ecotap_port_input)  # ecotap_port 입력란 추가   
        self.tap_mode_label = QLabel(f'Operating mode : {self.tap_mode}')  # 탭 위치 라벨 설정
        ecotap_status_layout.addWidget(self.tap_mode_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_op_label = QLabel(f'Tap Operations: {self.tap_op}')  # 탭 위치 라벨 설정
        ecotap_status_layout.addWidget(self.tap_op_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_position_label = QLabel(f'Tap Position: {self.tap_position}')  # 탭 위치 라벨 설정
        ecotap_status_layout.addWidget(self.tap_position_label)  # 탭 위치 레이아웃에 라벨 추가
        self.tap_de_voltage_label = QLabel(f'Desire Voltage: {self.tap_de_voltage}')  # 탭 전압 라벨 설정
        ecotap_status_layout.addWidget(self.tap_de_voltage_label)  # 탭 전압 라벨 추가
        self.tap_voltage_label = QLabel(f'Tap Voltage: {self.tap_voltage}')  # 탭 전압 라벨 설정
        ecotap_status_layout.addWidget(self.tap_voltage_label)  # 탭 전압 라벨 추가
            
        tap_button_layout = QHBoxLayout()
        self.tap_up_button = QPushButton('Tap Up', self)  # 탭 업 버튼 설정
        self.tap_up_button.clicked.connect(self.tap_up_action)  # 버튼 클릭 연결
        tap_button_layout.addWidget(self.tap_up_button)  # 탭 업 버튼 추가
        self.tap_down_button = QPushButton('Tap Down', self)  # 탭 다운 버튼 설정
        self.tap_down_button.clicked.connect(self.tap_down_action)  # 버튼 클릭 연결
        tap_button_layout.addWidget(self.tap_down_button)  # 탭 다운 버튼 추가
        # Add Test 1 and Test 2 buttons next to Tap Up and Tap Down
        self.test_1_button = QPushButton('Test 1', self)  # Test 1 버튼 설정
        self.test_1_button.clicked.connect(self.tap_test_control_1)  # 버튼 클릭 연결
        tap_button_layout.addWidget(self.test_1_button)  # Test 1 버튼 추가
        self.test_2_button = QPushButton('Test 2', self)  # Test 2 버튼 설정
        self.test_2_button.clicked.connect(self.tap_test_control_2)  # 버튼 클릭 연결
        tap_button_layout.addWidget(self.test_2_button)  # Test 2 버튼 추가
        ecotap_status_layout.addLayout(tap_button_layout)  # 버튼 레이아웃 추가
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)
        self.test_1_button.setEnabled(False)
        self.test_2_button.setEnabled(False)
        
        self.no_modbus_checkbox = QCheckBox("Not connected")  # Modbus 미연결 체크박스 설정
        self.no_modbus_checkbox.stateChanged.connect(self.toggle_modbus_sensor)  # 체크박스 상태 변경 연결
        ecotap_status_layout.addWidget(self.no_modbus_checkbox)  # Modbus 미연결 체크박스 추가  
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

        # Plot 영역 추가
        plot_layout = QVBoxLayout()  # 그래프 레이아웃 설정
        plot_layout.setSpacing(8)  # 패널 간격 설정

        self.sound_plot = PlotCanvas(self, title="Sound")  # 사운드 그래프 설정
        self.vibration_plot = PlotCanvas(self, title="Vibration")  # 진동 그래프 설정
        self.voltage_plot = PlotCanvas(self, title="Voltage")  # 전압 그래프 설정
        self.current_plot = PlotCanvas(self, title="Current")  # 전류 그래프 설정

        plot_layout.addWidget(self.sound_plot)  # 그래프 레이아웃에 사운드 그래프 추가
        plot_layout.addWidget(self.vibration_plot)  # 그래프 레이아웃에 진동 그래프 추가
        plot_layout.addWidget(self.voltage_plot)  # 그래프 레이아웃에 전압 그래프 추가
        plot_layout.addWidget(self.current_plot)  # 그래프 레이아웃에 전류 그래프 추가

        vibration_frame = QGroupBox('Vibration')  # 진동 설정 프레임 설정
        vibration_layout = QHBoxLayout()  # 진동 설정 레이아웃 설정
        self.serial_port_label = QLabel('COM Port')  # 시리얼 포트 라벨 설정
        self.serial_port_input = QLineEdit(self)  # 시리얼 포트 입력란 설정
        self.serial_port_input.setText('COM19')  # 기본 시리얼 포트 설정
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
        def run_tap_up():
            self.logger.log("Tap Up")  # address 0 의 값을 1로 변경
            self.modbus_client.tap_up()

        # Tap Up 버튼 비활성화
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)

        # Tap Up 작업 실행
        threading.Thread(target=run_tap_up).start()

        # 3초 후에 버튼을 다시 활성화하는 타이머 설정
        QTimer.singleShot(5000, lambda: self.tap_up_button.setEnabled(True))
        QTimer.singleShot(5000, lambda: self.tap_down_button.setEnabled(True))


    def tap_down_action(self):
        def run_tap_down():
            self.logger.log("Tap Down")  # address 1 의 값을 1로 변경
            self.modbus_client.tap_down()

        # Tap Down 버튼 비활성화
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)

        # Tap Down 작업 실행
        threading.Thread(target=run_tap_down).start()

        # 3초 후에 버튼을 다시 활성화하는 타이머 설정
        QTimer.singleShot(5000, lambda: self.tap_up_button.setEnabled(True))
        QTimer.singleShot(5000, lambda: self.tap_down_button.setEnabled(True))
        
    def tap_test_control_1(self):
        self.logger.log("Starting Test 1")

        # 비활성화된 동안 버튼이 중복으로 눌리지 않도록 설정
        self.test_1_button.setEnabled(False)
        self.test_2_button.setEnabled(False)
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)

        def perform_tap_up_down(up_count, down_count, repeat_count):
            if self.stop_event.is_set():# 중단 이벤트가 설정되면 종료
                return
            if repeat_count > 0:
                if up_count > 0:
                    self.tap_up_action()  # Tap Up 실행
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(up_count - 1, down_count, repeat_count))
                elif down_count > 0:
                    self.tap_down_action()  # Tap Down 실행
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(up_count, down_count - 1, repeat_count))
                else:
                    # 한 번의 Up-Down 루틴이 끝나면, 다음 루틴으로 진행
                    self.logger.log(f"Completed cycle {60 - repeat_count + 1}")
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(8, 8, repeat_count - 1))  # up_count와 down_count를 8로 재설정
            else:
                # 모든 작업이 끝나면 버튼을 다시 활성화
                self.logger.log("Test 1 completed.")
                self.test_1_button.setEnabled(True)
                self.test_2_button.setEnabled(True)
                self.tap_up_button.setEnabled(True)
                self.tap_down_button.setEnabled(True)

        # Tap Up 8번, Tap Down 8번 실행하는 루틴을 60번 반복
        perform_tap_up_down(8, 8, 60)


    def tap_test_control_2(self):
        self.logger.log("Starting Test 2")

        # 비활성화된 동안 버튼이 중복으로 눌리지 않도록 설정
        self.test_1_button.setEnabled(False)
        self.test_2_button.setEnabled(False)
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)

        def perform_tap_up_down(up_count, down_count, repeat_count):
            if self.stop_event.is_set():# 중단 이벤트가 설정되면 종료
                return
            if repeat_count > 0:
                if up_count > 0:
                    self.tap_up_action()  # Tap Up 실행
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(up_count - 1, down_count, repeat_count))
                elif down_count > 0:
                    self.tap_down_action()  # Tap Down 실행
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(up_count, down_count - 1, repeat_count))
                else:
                    # 한 번의 Up-Down 루틴이 끝나면, 다음 루틴으로 진행
                    self.logger.log(f"Completed cycle {60 - repeat_count + 1}")
                    QTimer.singleShot(5000, lambda: perform_tap_up_down(1, 1, repeat_count - 1))  # up_count와 down_count를 1로 재설정
            else:
                # 모든 작업이 끝나면 버튼을 다시 활성화
                self.logger.log("Test 2 completed.")
                self.test_1_button.setEnabled(True)
                self.test_2_button.setEnabled(True)
                self.tap_up_button.setEnabled(True)
                self.tap_down_button.setEnabled(True)

        # Tap Up 1번, Tap Down 1번 실행하는 루틴을 60번 반복
        perform_tap_up_down(1, 1, 60)

    def toggle_minute_checkbox(self):
        """1분 설정 체크박스 동작"""
        if self.minute_checkbox.isChecked():  # 1분 체크박스가 체크되면
            self.duration_input.setText('60')  # 주기를 60초로 설정
            self.duration_input.setEnabled(False)  # 입력란 비활성화
            self.month_checkbox.setChecked(False)  # 1개월 설정 해제
        else:
            self.duration_input.setEnabled(True)  # 입력란 활성화

    def toggle_month_checkbox(self):
        """1개월 설정 체크박스 동작"""
        if self.month_checkbox.isChecked():  # 1개월 체크박스가 체크되면
            self.repeat_num_input.setText('43200')  # 반복 횟수를 43200으로 설정
            self.repeat_num_input.setEnabled(False)  # 입력란 비활성화
            self.minute_checkbox.setChecked(False)  # 1분 설정 해제
        else:
            self.repeat_num_input.setEnabled(True)  # 입력란 활성화

    def toggle_vibration_sensor(self):
        if self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서 미연결 체크박스가 체크되면
            self.serial_port_input.setEnabled(False)  # 시리얼 포트 입력란 비활성화
            self.baud_rate_input.setEnabled(False)  # 보드 레이트 입력란 비활성화
        else:
            self.serial_port_input.setEnabled(True)  # 시리얼 포트 입력란 활성화
            self.baud_rate_input.setEnabled(True)  # 보드 레이트 입력란 활성화

    def toggle_modbus_sensor(self):
        if self.no_modbus_checkbox.isChecked():  # Modbus 미연결 체크박스가 체크되면
            self.ecotap_port_input.setEnabled(False)  # Modbus 포트 입력란 비활성화
            self.tap_up_button.setEnabled(False)  # Tap Up 버튼 비활성화
            self.tap_down_button.setEnabled(False)  # Tap Down 버튼 비활성화
            self.test_1_button.setEnabled(False) 
            self.test_2_button.setEnabled(False) 
        else:
            self.ecotap_port_input.setEnabled(True)  # Modbus 포트 입력란 활성화
            self.tap_up_button.setEnabled(True)  # Tap Up 버튼 활성화
            self.tap_down_button.setEnabled(True)  # Tap Down 버튼 활성화
            self.test_1_button.setEnabled(True)
            self.test_2_button.setEnabled(True)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')  # 디렉토리 선택 대화상자 열기
        if folder:
            self.savedir_input.setText(folder)  # 선택한 디렉토리 경로 설정

    def start_collection(self):
       self.savedir = self.savedir_input.text()  # 저장 경로 설정
       self.duration = int(self.duration_input.text())  # 주기 설정
       self.repeat_num = int(self.repeat_num_input.text())  # 반복 횟수 설정
       self.stop_event = threading.Event()  # 중지 이벤트 설정
           # **디버깅용 로그 출력**
       print(f"[DEBUG] Duration: {self.duration}, Repeat Num: {self.repeat_num}")
       self.logger.log(f"Starting collection with Duration: {self.duration}, Repeat: {self.repeat_num}")

       self.progress_bar.setMaximum(self.duration)  # 진행률 바 최대값 설정
       self.progress_bar.setValue(0)  # 진행률 바 초기값 설정
       self.total_progress_bar.setMaximum(self.repeat_num)  # 총 진행률 바 최대값 설정
       self.total_progress_bar.setValue(0)  # 총 진행률 바 초기값 설정
              
       self.baud_rate = int(self.baud_rate_input.text())  # 보드 레이트 설정
       self.serial_port = self.serial_port_input.text()  # 시리얼 포트 설정
       self.ecotap_port = self.ecotap_port_input.text()  # ecotap 포트 설정
       self.exp_num = int(self.exp_num_input.text())  # 실험 번호 설정
       self.exp_date = self.exp_date_input.text()  # 실험 날짜 설정
       self.audio_samplerate = int(self.audio_samplerate_input.text())  # 샘플링 레이트 설정
       self.ecotap_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'ecotap')  # 센서 데이터 저장 폴더 생성
       if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
           self.sensor_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sensors')  # 센서 데이터 저장 폴더 생성
       self.audio_recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num, 'sound')  # 사운드 데이터 저장 폴더 생성
       
       self.tap_up_button.setEnabled(True)
       self.tap_down_button.setEnabled(True)
       self.test_1_button.setEnabled(True)
       self.test_2_button.setEnabled(True)      
              
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

       if not self.no_modbus_checkbox.isChecked():  # Modbus가 연결된 경우
           self.modbus_client = ModbusRTUClient(self.ecotap_port, self.ecotap_recordings_folder_path, self.exp_date, self.exp_num)
           self.ecotap_timer.start(1000)  # ECOTAP 데이터 업데이트 타이머 시작 (1초 간격)

       self.start_button.setEnabled(False)  # 시작 버튼 비활성화
       self.stop_button.setEnabled(True)  # 중지 버튼 활성화

    def stop_collection(self):
        self.stop_event.set()  # 중지 이벤트 설정
        
        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.data_collector_thread.quit()  # 데이터 수집 스레드 중지
            self.data_collector_thread.wait()  # 데이터 수집 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 중지
        self.recorder_thread.wait()  # 녹음 스레드 종료 대기

        # Modbus RTU 데이터 읽기 중지
        if hasattr(self, 'modbus_client'):
            self.modbus_client.stop_reading()
            self.ecotap_timer.stop()  # ECOTAP 데이터 업데이트 타이머 중지

        self.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.test_1_button.setEnabled(False)
        self.test_2_button.setEnabled(False)
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)
        self.logger.log("ECOTAP Diagnosis Stop.")  # 중지 로그 출력
        self.timer.stop()  # 타이머 중지
        self.status_label.setText('')  # 상태 라벨 초기화
        
    def collection_finished(self):
        self.stop_event.set()  # 모든 작업 중지
        if not self.no_vibration_sensor_checkbox.isChecked():  # 진동 센서가 연결된 경우
            self.data_collector_thread.quit()  # 데이터 수집 스레드 중지
            self.data_collector_thread.wait()  # 데이터 수집 스레드 종료 대기
        self.recorder_thread.quit()  # 녹음 스레드 중지
        self.recorder_thread.wait()  # 녹음 스레드 종료 대기
                
        self.start_button.setEnabled(True)  # 시작 버튼 활성화
        self.stop_button.setEnabled(False)  # 중지 버튼 비활성화
        self.test_1_button.setEnabled(False)
        self.test_2_button.setEnabled(False)
        self.tap_up_button.setEnabled(False)
        self.tap_down_button.setEnabled(False)
        
        # Modbus RTU 종료
        if hasattr(self, 'modbus_client'):
            self.modbus_client.stop_reading()
            self.ecotap_timer.stop()
            
        self.timer.stop()  # 타이머 중지
        self.logger.log("ECOTAP Diagnosis Done.")
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
            self.status_label.setStyleSheet('color: green; font-size: 60px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 1:
            if self.status_visible:
                self.status_label.setText('Maintenance\nRequired')  # 상태 라벨에 "Predictive Maintenance Required" 설정
            else:
                self.status_label.setText('')  # 상태 라벨 초기화
            self.status_label.setStyleSheet('color: orange; font-size: 40px')  # 상태 라벨 스타일 설정
        elif self.machine_error == 2:
            if self.status_visible:
                self.status_label.setText('Error')  # 상태 라벨에 "Error" 설정
            else:
                self.status_label.setText('')  # 상태 라벨 초기화
            self.status_label.setStyleSheet('color: red; font-size: 60px')  # 상태 라벨 스타일 설정

        self.status_visible = not self.status_visible  # 상태 가시성 토글

    def update_ecotap_status(self):
        if hasattr(self, 'modbus_client'):
            tap_op, tap_de_voltage, tap_position, tap_voltage, tap_mode= self.modbus_client.get_latest_data()
            self.tap_mode_label.setText(f'Operating mode: {tap_mode}')
            self.tap_op_label.setText(f'Tap Operations: {tap_op}')
            self.tap_position_label.setText(f'Tap Position: {tap_position}')
            self.tap_voltage_label.setText(f'Tap Voltage: {tap_voltage}')
            self.tap_de_voltage_label.setText(f"Desire Voltage: {tap_de_voltage}")

if __name__ == '__main__':
    global ex
    app = QApplication(sys.argv)  # QApplication 객체 생성
    ex = DataCollectorApp(machine_error)  # DataCollectorApp 객체 생성
    ex.show()  # 앱 창 표시
    sys.exit(app.exec_())  # 앱 실행 및 종료
