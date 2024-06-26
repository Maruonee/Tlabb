import os  # 운영 체제 모듈 가져오기
import time  # 시간 모듈 가져오기
import threading  # 스레딩 모듈 가져오기
import re  # 정규 표현식 모듈 가져오기
from datetime import datetime  # datetime 모듈에서 datetime 가져오기
import serial  # 시리얼 통신 모듈 가져오기
from PyQt5.QtCore import pyqtSignal, QObject  # PyQt5 코어 클래스 및 시그널 가져오기

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 진행 상태 시그널 정의
    total_progress_signal = pyqtSignal(int, int)  # 총 진행 상태 시그널 정의
    log_signal = pyqtSignal(str)  # 로그 시그널 정의
    finished_signal = pyqtSignal()  # 작업 완료 시그널 정의

    def __init__(self, main_window, duration, baud_rate, serial_port, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()  # 부모 클래스 초기화
        self.main_window = main_window  # 메인 윈도우 인스턴스 설정
        self.duration = duration  # 데이터 수집 지속 시간 설정
        self.baud_rate = baud_rate  # 보드 레이트 설정
        self.serial_port = serial_port  # 시리얼 포트 설정
        self.folder_path = folder_path  # 저장 폴더 경로 설정
        self.repeat_num = repeat_num  # 반복 횟수 설정
        self.exp_date = exp_date  # 실험 날짜 설정
        self.exp_num = exp_num  # 실험 번호 설정
        self.stop_event = stop_event  # 중지 이벤트 설정

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)  # 시리얼 포트 열기
        except serial.SerialException as e:
            self.log_signal.emit(f"Unable to open serial port: {e}")  # 시리얼 포트 열기 실패 로그 기록
            self.finished_signal.emit()  # 작업 완료 시그널 발생
            return

        txt_file_ref, initial_filename = self.create_new_file()  # 새로운 파일 생성
        txt_file_ref = [txt_file_ref]  # 파일 참조 리스트 설정
        lock = threading.Lock()  # 스레드 락 설정

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)  # 파일 갱신 스레드 생성
        file_refresh_thread_obj.start()  # 파일 갱신 스레드 시작

        try:
            while file_refresh_thread_obj.is_alive():
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()  # 시리얼 포트에서 데이터 읽기
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')  # 데이터 파일에 쓰기
                            txt_file_ref[0].flush()  # 파일 버퍼 플러시
        except KeyboardInterrupt:
            self.log_signal.emit("Diagnosis(Vibration) stop.")  # 진단 중지 로그 기록
        finally:
            with lock:
                txt_file_ref[0].close()  # 파일 닫기
            ser.close()  # 시리얼 포트 닫기
            self.finished_signal.emit()  # 작업 완료 시그널 발생

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 타임스탬프 생성
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
                txt_file_ref[0].close()  # 현재 파일 닫기
                txt_file_ref[0] = new_file  # 새로운 파일로 교체
            self.log_signal.emit(f"{filename} saved.")  # 파일 저장 로그 기록
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행 상태 업데이트

            threading.Thread(target=self.plot_vibration, args=(filename,)).start()  # 별도의 스레드로 플롯 그리기 실행
        
        self.log_signal.emit("Diagnosis(Vibration) Done.")  # 진단 완료 로그 기록
        self.finished_signal.emit()  # 작업 완료 시그널 발생

    def plot_vibration(self, filename):
        vr1_values = []  # VR1 값 리스트
        vr2_values = []  # VR2 값 리스트
        with open(filename, 'r') as file:
            pattern = re.compile(r"VR1\s*:\s*(\d+)\s*VR2\s*:\s*(\d+)")  # 정규 표현식 패턴 설정
            extracted_data = [pattern.search(line).groups() for line in file if pattern.search(line)]  # 데이터 추출
            vr1_values = [int(vr1) for vr1, vr2 in extracted_data]  # VR1 값 리스트 생성
            vr2_values = [int(vr2) for vr1, vr2 in extracted_data]  # VR2 값 리스트 생성
            x_values = list(range(1, len(vr1_values) + 1))  # x 값 리스트 생성
        self.main_window.vibration_plot.ax.clear()  # 플롯 초기화
        self.main_window.vibration_plot.ax.plot(x_values, vr1_values, label="VR1")  # VR1 값 플롯 그리기
        self.main_window.vibration_plot.ax.plot(x_values, vr2_values, label="VR2")  # VR2 값 플롯 그리기
        self.main_window.vibration_plot.ax.set_title("Vibration")  # 플롯 제목 설정
        self.main_window.vibration_plot.ax.legend()  # 범례 설정
        self.main_window.vibration_plot.draw
