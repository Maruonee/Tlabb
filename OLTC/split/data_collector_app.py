import threading
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QThread

from utils.logger import Logger
from utils.recorder_worker import RecorderWorker
from utils.data_collector_worker import DataCollectorWorker
from gui.status import StatusFrame
from gui.settings import SettingsFrame
from gui.progress import ProgressFrame
from gui.controls import ControlsFrame
from gui.log_output import LogOutputFrame
from utils.utils import create_folder


class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.machine_error = machine_error
        self.initUI()
        self.logger = Logger()
        self.logger.log_signal.connect(self.log_output_frame.update_log)
        self.timer = self.status_frame.timer
        self.timer.timeout.connect(self.status_frame.update_status)
        self.status_frame.status_visible = True

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')
        self.resize(500, 700)

        self.status_frame = StatusFrame(self)
        self.settings_frame = SettingsFrame(self)
        self.progress_frame = ProgressFrame(self)
        self.controls_frame = ControlsFrame(self)
        self.log_output_frame = LogOutputFrame(self)

        self.controls_frame.start_button.clicked.connect(self.start_collection)
        self.controls_frame.stop_button.clicked.connect(self.stop_collection)

        layout = QVBoxLayout()
        layout.addWidget(self.status_frame)
        layout.addWidget(self.settings_frame)
        layout.addWidget(self.progress_frame)
        layout.addWidget(self.controls_frame)
        layout.addWidget(self.log_output_frame)
        self.setLayout(layout)

    def start_collection(self):
        settings = self.settings_frame.get_settings()
        self.stop_event = threading.Event()

        self.logger.log(f"=======ECOTAP 고장 진단 시작=======\n설정 주기: {settings['duration']}초\n설정 기간: {settings['repeat_num']}\n진동데이터 저장위치: {settings['sensor_recordings_folder_path']}\n음향데이터 저장위치: {settings['audio_recordings_folder_path']}\n")

        if self.machine_error == 0:
            self.timer.start(500)  # 0.5초 간격으로 상태 업데이트
        elif self.machine_error == 1:
            self.timer.start(300)  # 0.3초 간격으로 상태 업데이트
        elif self.machine_error == 2:
            self.timer.start(100)  # 0.1초 간격으로 상태 업데이트

        self.recorder_worker = RecorderWorker(settings['duration'], settings['audio_samplerate'], 2, settings['audio_recordings_folder_path'], settings['repeat_num'], settings['exp_date'], settings['exp_num'], self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)

        self.data_collector_worker = DataCollectorWorker(settings['duration'], settings['baud_rate'], settings['serial_port'], settings['sensor_recordings_folder_path'], settings['repeat_num'], settings['exp_date'], settings['exp_num'], self.stop_event)
        self.data_collector_thread = QThread()
        self.data_collector_worker.moveToThread(self.data_collector_thread)

        self.recorder_worker.progress_signal.connect(self.progress_frame.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.progress_frame.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)

        self.data_collector_worker.progress_signal.connect(self.progress_frame.update_progress)
        self.data_collector_worker.total_progress_signal.connect(self.progress_frame.update_total_progress)
        self.data_collector_worker.log_signal.connect(self.logger.log)
        self.data_collector_worker.finished_signal.connect(self.collection_finished)

        self.data_collector_thread.started.connect(self.data_collector_worker.run)
        self.recorder_thread.started.connect(self.recorder_worker.run)

        self.data_collector_thread.start()
        self.recorder_thread.start()

        self.controls_frame.start_button.setEnabled(False)
        self.controls_frame.stop_button.setEnabled(True)
        self.progress_frame.reset_progress()

    def stop_collection(self):
        self.stop_event.set()
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.controls_frame.start_button.setEnabled(True)
        self.controls_frame.stop_button.setEnabled(False)
        self.logger.log("고장진단이 중지되었습니다.")
        self.timer.stop()
        self.status_frame.status_label.setText('')

    def collection_finished(self):
        self.data_collector_thread.quit()
        self.data_collector_thread.wait()
        self.recorder_thread.quit()
        self.recorder_thread.wait()
        self.controls_frame.start_button.setEnabled(True)
        self.controls_frame.stop_button.setEnabled(False)
        self.timer.stop()
        self.status_frame.status_label.setText('')