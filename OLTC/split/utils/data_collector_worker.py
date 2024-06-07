import os
import time
import threading
from datetime import datetime
import serial
from PyQt5.QtCore import QObject, pyqtSignal

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
            self.log_signal.emit(f"직렬 포트를 열 수 없습니다: {e}")
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
            self.log_signal.emit("진동 진단 중지")
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
            self.log_signal.emit(f"{filename} 저장.")
            self.total_progress_signal.emit(int(((i + 1) / self.repeat_num) * 100), 100)
        self.log_signal.emit("설정한 기간의 진동 진단 완료")
        self.finished_signal.emit()