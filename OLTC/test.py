import sys
import os
import time
import threading
import re
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer

machine_error = 0
tap_position = 5
tap_voltage = 0
tap_up = 0
tap_down = 0

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
                self.log_signal.emit("Diagnosis(sound) stop.")
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.exp_date}_{self.exp_num}_sound"
            filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.wav')
            recording = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=self.channels)
            for second in range(self.duration):
                if self.stop_event.is_set():
                    sd.stop()
                    sf.write(filename, recording[:int(second * self.samplerate)], self.samplerate, format='FLAC')
                    self.log_signal.emit(f"{filename} saved.")
                    return
                self.progress_signal.emit(int(((second + 1) / self.duration) * 100), 100)
                time.sleep(1)
            if self.stop_event.is_set():
                break
            sd.wait()
            sf.write(filename, recording, self.samplerate, format='FLAC')
            self.log_signal.emit(f"{filename} saved.")
            self.progress_signal.emit(100, 100)
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)

            threading.Thread(target=self.plot_sound, args=(filename,)).start()
        
        self.log_signal.emit("Diagnosis(sound) done.")
        self.finished_signal.emit()
        self.stop_event.set()

    def plot_sound(self, filename):
        data, samplerate = sf.read(filename)
        duration = len(data) / samplerate
        time = np.linspace(0., duration, len(data))
        ex.sound_plot.ax.clear()
        ex.sound_plot.ax.plot(time, data)
        ex.sound_plot.ax.set_title("Sound")
        ex.sound_plot.draw()

class DataCollectorWorker(QObject):
    progress_signal = pyqtSignal(int, int)
    total_progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

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

    def run(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate)
        except serial.SerialException as e:
            self.log_signal.emit(f"Unable to open serial port: {e}")
            self.finished_signal.emit()
            return

        txt_file_ref, initial_filename = self.create_new_file()
        txt_file_ref = [txt_file_ref]
        lock = threading.Lock()

        file_refresh_thread_obj = threading.Thread(target=self.file_refresh_thread, args=(txt_file_ref, lock), daemon=True)
        file_refresh_thread_obj.start()

        try:
            while file_refresh_thread_obj.is_alive():
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    if data:
                        with lock:
                            txt_file_ref[0].write(f'{data}\n')
                            txt_file_ref[0].flush()
        except KeyboardInterrupt:
            self.log_signal.emit("Diagnosis(Vibration) stop.")
        finally:
            with lock:
                txt_file_ref[0].close()
            ser.close()
            self.finished_signal.emit()

    def create_new_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_sensors"
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')
        return open(filename, mode='w'), filename

    def file_refresh_thread(self, txt_file_ref, lock):
        for i in range(self.repeat_num):
            for _ in range(self.duration):
                if self.stop_event.is_set():
                    return
                time.sleep(1)
                self.progress_signal.emit(int(((_ + 1) / self.duration) * 100), 100)
            new_file, filename = self.create_new_file()
            with lock:
                txt_file_ref[0].close()
                txt_file_ref[0] = new_file
            self.log_signal.emit(f"{filename} saved.")
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)

            threading.Thread(target=self.plot_vibration, args=(filename)).start()
        
        self.log_signal.emit("Diagnosis(Vibration) Done.")
        self.finished_signal.emit()

    def plot_vibration(self, filename):
        vr1_values = []
        vr2_values = []
        with open(filename, 'r') as file:
            pattern = re.compile(r"VR1\s*:\s*(\d+)\s*VR2\s*:\s*(\d+)")
            extracted_data = [pattern.search(line).groups() for line in file.read()]
            vr1_values = [int(vr1) for vr1, vr2 in extracted_data]
            vr2_values = [int(vr2) for vr1, vr2 in extracted_data]
            x_values = list(range(1, len(file.read()) + 1))
        ex.vibration_plot.ax.clear()
        ex.vibration_plot.ax.plot(x_values, vr1_values)
        ex.vibration_plot.ax.plot(x_values, vr2_values)
        ex.vibration_plot.ax.set_title("Vibration")
        ex.vibration_plot.ax.legend()
        ex.vibration_plot.draw()

def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, title=""):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)
        self.ax.set_title(title)
        self.ax.plot([])

class DataCollectorApp(QWidget):
    def __init__(self, machine_error):
        super().__init__()
        self.initUI()
        self.logger = Logger()
        self.machine_error = machine_error
        self.logger.log_signal.connect(self.update_log)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.status_visible = True

    def initUI(self):
        self.setWindowTitle('ECOTAP Diagnosis System by Tlab')
        self.resize(800, 600)

        main_layout = QHBoxLayout(self)

        plot_layout = QVBoxLayout()
        self.sound_plot = PlotCanvas(self, title="Sound")
        self.vibration_plot = PlotCanvas(self, title="Vibration")
        self.voltage_plot = PlotCanvas(self, title="Voltage")
        self.current_plot = PlotCanvas(self, title="Current")
        plot_layout.addWidget(self.sound_plot)
        plot_layout.addWidget(self.vibration_plot)
        plot_layout.addWidget(self.voltage_plot)
        plot_layout.addWidget(self.current_plot)

        main_layout.addLayout(plot_layout)

        right_panel_layout = QVBoxLayout()

        status_frame = QGroupBox('Diagnosis Results')
        status_frame.setStyleSheet('background-color: white')
        status_layout = QVBoxLayout()
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 24px')
        status_layout.addWidget(self.status_label)
        status_frame.setLayout(status_layout)
        right_panel_layout.addWidget(status_frame)

        desktop_path = os.path.join(os.path.expanduser("~"), 'downloads')
        savedir_layout = QHBoxLayout()
        self.savedir_label = QLabel('Folder path')
        self.savedir_input = QLineEdit(self)
        self.savedir_input.setText(desktop_path)
        self.savedir_button = QPushButton('Browse', self)
        self.savedir_button.clicked.connect(self.browse_folder)
        savedir_layout.addWidget(self.savedir_label)
        savedir_layout.addWidget(self.savedir_input)
        savedir_layout.addWidget(self.savedir_button)
        right_panel_layout.addLayout(savedir_layout)

        duration_frame = QGroupBox('Diagnosis setup')
        duration_layout = QHBoxLayout()
        self.duration_label = QLabel('Cycle(sec)')
        self.duration_input = QLineEdit(self)
        self.duration_input.setText('60')
        self.minute_checkbox = QCheckBox("1 min")
        self.minute_checkbox.stateChanged.connect(self.toggle_minute_checkbox)
        self.repeat_num_label = QLabel('Repeat')
        self.repeat_num_input = QLineEdit(self)
        self.repeat_num_input.setText('60')
        self.month_checkbox = QCheckBox("1 month")
        self.month_checkbox.stateChanged.connect(self.toggle_month_checkbox)
        duration_layout.addWidget(self.duration_label)
        duration_layout.addWidget(self.duration_input)
        duration_layout.addWidget(self.minute_checkbox)
        duration_layout.addWidget(self.repeat_num_label)
        duration_layout.addWidget(self.repeat_num_input)
        duration_layout.addWidget(self.month_checkbox)
        duration_frame.setLayout(duration_layout)
        right_panel_layout.addWidget(duration_frame)

        vibration_frame = QGroupBox('Vibration')
        vibration_layout = QHBoxLayout()
        self.serial_port_label = QLabel('COM Port')
        self.serial_port_input = QLineEdit(self)
        self.serial_port_input.setText('COM7')
        self.baud_rate_label = QLabel('Baud Rate')
        self.baud_rate_input = QLineEdit(self)
        self.baud_rate_input.setText('19200')
        self.no_vibration_sensor_checkbox = QCheckBox("Not connected")
        self.no_vibration_sensor_checkbox.stateChanged.connect(self.toggle_vibration_sensor)
        vibration_layout.addWidget(self.serial_port_label)
        vibration_layout.addWidget(self.serial_port_input)
        vibration_layout.addWidget(self.baud_rate_label)
        vibration_layout.addWidget(self.baud_rate_input)
        vibration_layout.addWidget(self.no_vibration_sensor_checkbox)
        vibration_frame.setLayout(vibration_layout)
        right_panel_layout.addWidget(vibration_frame)

        audio_frame = QGroupBox('Sound')
        audio_layout = QHBoxLayout()
        self.audio_samplerate_label = QLabel('Sampling Rate(Hz)')
        self.audio_samplerate_input = QLineEdit(self)
        self.audio_samplerate_input.setText('44100')
        audio_layout.addWidget(self.audio_samplerate_label)
        audio_layout.addWidget(self.audio_samplerate_input)
        audio_frame.setLayout(audio_layout)
        right_panel_layout.addWidget(audio_frame)

        exp_frame = QGroupBox('Data name')
        exp_layout = QHBoxLayout()
        self.exp_date_label = QLabel('YYMMDD')
        self.exp_date_input = QLineEdit(self)
        self.exp_date_input.setText(datetime.now().strftime('%y%m%d'))
        self.exp_num_label = QLabel('Number')
        self.exp_num_input = QLineEdit(self)
        self.exp_num_input.setText('1')
        exp_layout.addWidget(self.exp_date_label)
        exp_layout.addWidget(self.exp_date_input)
        exp_layout.addWidget(self.exp_num_label)
        exp_layout.addWidget(self.exp_num_input)
        exp_frame.setLayout(exp_layout)
        right_panel_layout.addWidget(exp_frame)

        ecotap_status_frame = QGroupBox('ECOTAP Status')
        ecotap_status_layout = QVBoxLayout()
        tap_position_layout = QHBoxLayout()
        self.tap_position_label = QLabel(f'Tap position: {tap_position}')
        tap_position_layout.addWidget(self.tap_position_label)
        self.tap_up_button = QPushButton('Tap Up', self)
        self.tap_up_button.clicked.connect(self.tap_up_action)
        self.tap_down_button = QPushButton('Tap Down', self)
        self.tap_down_button.clicked.connect(self.tap_down_action)
        tap_position_layout.addWidget(self.tap_up_button)
        tap_position_layout.addWidget(self.tap_down_button)
        ecotap_status_layout.addLayout(tap_position_layout)
        self.tap_voltage_label = QLabel(f'Tap voltage: {tap_voltage}')
        ecotap_status_layout.addWidget(self.tap_voltage_label)
        ecotap_status_frame.setLayout(ecotap_status_layout)
        right_panel_layout.addWidget(ecotap_status_frame)

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
        right_panel_layout.addWidget(progress_frame)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_collection)
        self.start_button.setFixedSize(60, 30)
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setFixedSize(60, 30)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        right_panel_layout.addLayout(button_layout)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        right_panel_layout.addWidget(self.log_output)

        main_layout.addLayout(right_panel_layout)
        self.setLayout(main_layout)

    def tap_up_action(self):
        global tap_up
        tap_up = 999
        self.logger.log(f"Tap up pressed. tap_up is now {tap_up}")

    def tap_down_action(self):
        global tap_down
        tap_down = 999
        self.logger.log(f"Tap down pressed. tap_down is now {tap_down}")

    def toggle_minute_checkbox(self):
        if self.minute_checkbox.isChecked():
            self.duration_input.setText('60')
            self.duration_input.setEnabled(False)
        else:
            self.duration_input.setEnabled(True)

    def toggle_month_checkbox(self):
        if self.month_checkbox.isChecked():
            self.repeat_num_input.setText('43200')
            self.repeat_num_input.setEnabled(False)
        else:
            self.repeat_num_input.setEnabled(True)

    def toggle_vibration_sensor(self):
        if self.no_vibration_sensor_checkbox.isChecked():
            self.serial_port_input.setEnabled(False)
            self.baud_rate_input.setEnabled(False)
        else:
            self.serial_port_input.setEnabled(True)
            self.baud_rate_input.setEnabled(True)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder:
            self.savedir_input.setText(folder)

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

        self.recorder_worker = RecorderWorker(self.duration, self.audio_samplerate, 2, self.audio_recordings_folder_path, self.repeat_num, self.exp_date, self.exp_num, self.stop_event)
        self.recorder_thread = QThread()
        self.recorder_worker.moveToThread(self.recorder_thread)
        self.recorder_worker.progress_signal.connect(self.update_progress)
        self.recorder_worker.total_progress_signal.connect(self.update_total_progress)
        self.recorder_worker.log_signal.connect(self.logger.log)
        self.recorder_worker.finished_signal.connect(self.collection_finished)

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

    def update_progress(self, value, maximum):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def update_total_progress(self, value, maximum):
        self.total_progress_bar.setMaximum(maximum)
        self.total_progress_bar.setValue(value)

    def update_log(self, message):
        self.log_output.append(message)

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