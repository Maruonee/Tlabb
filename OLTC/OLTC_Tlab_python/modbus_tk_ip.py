import os
import time
import struct
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import threading
from datetime import datetime
import csv


class RootechAccura:
    def __init__(self, accura_ip, folder_path, exp_date, exp_num, port=502, interval=0.1):
        
        self.master = modbus_tcp.TcpMaster(host=accura_ip, port=port)
        self.master.set_timeout(0.1)
        
        self.folder_path = folder_path
        self.exp_date = exp_date
        self.exp_num = exp_num
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock() 

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path) 
            
        # 데이터 읽기 시작
        self.start_reading()
        
    #전압 전류 데이터 float32임
    def to_float32(self, high, low):
        raw = (high << 16) | low
        return struct.unpack('>f', raw.to_bytes(4, byteorder='big'))[0]

    def read_registers(self):
        with self.lock:
            voltage_registers = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 11100, 16)
            current_registers = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 11200, 8)

        # 전압 데이터 
        voltage_data = {
            "VLN_A": self.to_float32(voltage_registers[0], voltage_registers[1]),
            "VLN_B": self.to_float32(voltage_registers[2], voltage_registers[3]),
            "VLN_C": self.to_float32(voltage_registers[4], voltage_registers[5]),
            "VLN_AVG": self.to_float32(voltage_registers[6], voltage_registers[7]),
            "VLL_AB": self.to_float32(voltage_registers[8], voltage_registers[9]),
            "VLL_BC": self.to_float32(voltage_registers[10], voltage_registers[11]),
            "VLL_CA": self.to_float32(voltage_registers[12], voltage_registers[13]),
            "VLL_AVG": self.to_float32(voltage_registers[14], voltage_registers[15]),
        }
        # 전류 데이터 
        current_data = {
            "I_A": self.to_float32(current_registers[0], current_registers[1]),
            "I_B": self.to_float32(current_registers[2], current_registers[3]),
            "I_C": self.to_float32(current_registers[4], current_registers[5]),
            "I_AVG": self.to_float32(current_registers[6], current_registers[7]),
        }
        return voltage_data, current_data

    def start_reading(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._update_registers)
        self.thread.start()

    def _update_registers(self):
        while not self.stop_event.is_set():
            voltage_data, current_data = self.read_registers()
            self.save_to_csv("volt", voltage_data)
            self.save_to_csv("current", current_data)
            time.sleep(0.1)

    def save_to_csv(self, data_type, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_accura"
        filename = os.path.join(self.folder_path, f"{folder_name}_{data_type}.csv")
        file_exists = os.path.isfile(filename)
        
        with self.lock, open(filename, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ["Timestamp"] + list(data.keys())
                writer.writerow(headers)
            row = [timestamp] + list(data.values())
            writer.writerow(row)

    def stop_reading(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        self.master._do_close()

    def get_latest_data(self):
        return self.read_registers()


# 실행 코드
if __name__ == "__main__":
    client = RootechAccura(
        accura_ip='169.254.6.188',
        folder_path='C:\\Users\\tlab\\Downloads',
        exp_date=datetime.now().strftime("%y%m%d"),  # 파일 이름에 YYMMDD 형식 적용
        exp_num='01',
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        client.stop_reading()
    except Exception as e:
        print(f"Unexpected error: {e}")
        client.stop_reading()
        