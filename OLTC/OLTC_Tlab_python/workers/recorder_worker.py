import os  # 운영 체제 모듈 가져오기
import time  # 시간 모듈 가져오기
import threading  # 스레딩 모듈 가져오기
from datetime import datetime  # datetime 모듈에서 datetime 가져오기
import sounddevice as sd  # sounddevice 모듈 가져오기
import soundfile as sf  # soundfile 모듈 가져오기
import numpy as np  # numpy 모듈 가져오기
from PyQt5.QtCore import pyqtSignal, QObject  # PyQt5 코어 클래스 및 시그널 가져오기

class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)  # 진행 상태 시그널 정의
    total_progress_signal = pyqtSignal(int, int)  # 총 진행 상태 시그널 정의
    log_signal = pyqtSignal(str)  # 로그 시그널 정의
    finished_signal = pyqtSignal()  # 작업 완료 시그널 정의

    def __init__(self, main_window, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()  # 부모 클래스 초기화
        self.main_window = main_window  # 메인 윈도우 인스턴스 설정
        self.duration = duration  # 녹음 지속 시간 설정
        self.samplerate = samplerate  # 샘플링 레이트 설정
        self.channels = channels  # 채널 수 설정
        self.folder_path = folder_path  # 저장 폴더 경로 설정
        self.repeat_num = repeat_num  # 반복 횟수 설정
        self.exp_date = exp_date  # 실험 날짜 설정
        self.exp_num = exp_num  # 실험 번호 설정
        self.stop_event = stop_event  # 중지 이벤트 설정

    def run(self):
        for i in range(self.repeat_num):
            if self.stop_event.is_set():
                self.log_signal.emit("Diagnosis(sound) stop.")  # 중지 로그 기록
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 타임스탬프 생성
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"  # 폴더 이름 생성
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')  # 파일 이름 생성
            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)  # 녹음 시작
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()  # 녹음 중지
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')  # 녹음 데이터 저장
                    self.log_signal.emit(f"{filename} saved.")  # 파일 저장 로그 기록
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)  # 진행 상태 업데이트
                time.sleep(1)  # 1초 대기
            if self.stop_event.is_set():
                break
            sd.wait()  # 녹음 완료 대기
            sf.write(filename, recording, self.samplerate, format='FLAC')  # 녹음 데이터 저장
            self.log_signal.emit(f"{filename} saved.")  # 파일 저장 로그 기록
            self.progress_signal.emit(100, 100)  # 진행 상태 업데이트
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)  # 총 진행 상태 업데이트

            threading.Thread(target=self.plot_sound, args=(filename,)).start()  # 별도의 스레드로 플롯 그리기 실행
        
        self.log_signal.emit("Diagnosis(sound) done.")  # 진단 완료 로그 기록
        self.finished_signal.emit()  # 작업 완료 시그널 발생
        self.stop_event.set()  # 중지 이벤트 설정

    def plot_sound(self, filename):
        data, samplerate = sf.read(filename)  # 녹음 데이터 읽기
        duration = len(data) / samplerate  # 녹음 지속 시간 계산
        time = np.linspace(0., duration, len(data))  # 시간 배열 생성
        self.main_window.sound_plot.ax.clear()  # 플롯 초기화
        self.main_window.sound_plot.ax.plot(time, data)  # 플롯에 데이터 그리기
        self.main_window.sound_plot.ax.set_title("Sound")  # 플롯 제목 설정
        self.main_window.sound_plot.draw()  # 플롯 업데이트
