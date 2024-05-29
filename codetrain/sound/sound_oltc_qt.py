import sys
import os
import time
import threading
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QProgressBar, QTextEdit, QComboBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread

class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def log(self, message):
        self.log_signal.emit(message)

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
                self.log_signal.emit("녹음 중지 요청됨.")
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')

            with open(filename, 'wb') as f:
                recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
                for second in range(self.duration):
                    if self.stop_event.is_set():
                        sd.stop()
                        write(filename, self.samplerate, recording[:int(second * self.samplerate)])
                        self.log_signal.emit(f"{filename} 저장완료.")
                        return
                    self.progress_signal.emit(second + 1, self.duration)
                    time.sleep(1)
                if self.stop_event.is_set():
                    break
                sd.wait()
                write(filename, self.samplerate, recording)
            self.log_signal.emit(f"{filename} 저장완료.")
            self.progress_signal.emit(self.duration, self.duration)
            self.total_progress_signal.emit(i + 1, self.repeat_num)
        
        self.log_signal.emit("모든 녹음이 완료되었습니다.")
        self.finished_signal.emit()
        self.stop_event.set()

def create_folder(savedir, exp_date, exp_num):
    folder_name = f"{exp_date}_{exp_num}_sound"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

class RecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.logger = Logger()
        self.logger.log_signal.connect(self.update_log)

    def initUI(self):
        self.setWindowTitle('OLTC Audio Recorder Tlab')
        self.resize(500, 500)

        desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
        
        self.savedir_label = QLabel('저장위치')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('폴더선택', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        
        self.duration_label = QLabel('녹음시간(초)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        
        self.samplerate_label = QLabel('Sample Rate (Hz)')
        self.samplerate_input = QLineEdit(self)
        self.samplerate_input.setText('44100')
        
        self.channels_label = QLabel('채널 (스테레오/모노)')
        self.channels_input = QComboBox(self)
        self.channels_input.addItem("스테레오")
        self.channels_input.addItem("모노")
        
        self.repeat_num_label = QLabel('반복횟수')
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        
        self.exp_num_label = QLabel('실험번호')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')
        
        self.exp_date_label = QLabel('실험날짜 (YYMMDD)')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        
        self.start_button = QPushButton('녹음 시작', self)
        self.start_button.clicked.connect(self.start_recording)

        self.stop_button = QPushButton('녹음 중단', self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)

        self.progress_label = QLabel('개별 진행률')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)

        self.total_progress_label = QLabel('전체 진행률')
        self.total_progress_bar = QProgressBar(self)
        self.total_progress_bar.setMinimum(0)
        self.total_progress_bar.setValue(0)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.savedir_label)
        layout.addWidget(self.savedir_input)
        layout.addWidget(self.savedir_button)
        layout.addWidget(self.duration_label)
        layout.addWidget(self.duration_input)
        layout.addWidget(self.samplerate_label)
        layout.addWidget(self.samplerate_input)
        layout.addWidget(self.channels_label)
        layout.addWidget(self.channels_input)
        layout.addWidget(self.repeat_num_label)
        layout.addWidget(self.repeat_num_input)
        layout.addWidget(self.exp_num_label)
        layout.addWidget(self.exp_num_input)
        layout.addWidget(self.exp_date_label)
        layout.addWidget(self.exp_date_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.total_progress_label)
        layout.addWidget(self.total_progress_bar)
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)
    
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)
    
    def start_recording(self):
        self.savedir = self.savedir_input.text()
        self.duration = int(self.duration_input.text())
        self.samplerate = int(self.samplerate_input.text())
        self.channels = 2 if self.channels_input.currentText() == "모노" else 1
        self.repeat_num = int(self.repeat_num_input.text())
        self.exp_num = int(self.exp_num_input.text())
        self.exp_date = self.exp_date_input.text()
        
        self.recordings_folder_path = create_folder(self.savedir, self.exp_date, self.exp_num)
        
        self.stop_event = threading.Event()
        
        self.logger.log(f"=======녹음 시작=======\n시간 : {self.duration}초\n반복횟수 : {self.repeat_num}\n저장위치 : {self.recordings_folder_path}\n")
        
        self.recorder_worker = RecorderWorker(self.duration, self.samplerate, self.channels, self.recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)

        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.recording_finished)
        
        self.recorder_thread.started.connect(self.recorder_worker.run)
        self.recorder_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.progress_bar.setMaximum(self.duration)
        self.progress_bar.setValue(0)
        self.total_progress_bar.setMaximum(self.repeat_num)
        self.total_progress_bar.setValue(0)
        
    def stop_recording(self):
        self.stop_event.set()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.logger.log("녹음이 중지되었습니다.")
    
    def recording_finished(self):
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RecorderApp()
    ex.show()
    sys.exit(app.exec_())