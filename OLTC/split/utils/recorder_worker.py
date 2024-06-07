import os
import time
from datetime import datetime
import sounddevice as sd
import soundfile as sf
from PyQt5.QtCore import QObject, pyqtSignal

class RecorderWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, duration, samplerate, channels, folder_path, repeat_num, exp_date, exp_num, stop_event):
        super().__init__()
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels
        self.folder_path = folder_path
        self.repeat_num = repeat_num
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.stop_event = stop_event

    def run(self):
        for i in range(self.repeat_num):
            if self.stop_event.is_set():
                self.log_signal.emit("음향 진단 중지")
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')

            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')
                    self.log_signal.emit(f"{filename} 저장.")
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)
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