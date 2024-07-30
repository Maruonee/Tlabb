import os
import time
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
import threading
import datetime

class ModbusRTUClient:
    def __init__(self, port, folder_path, exp_date, exp_num):

        # 시리얼 포트 설정
        self.serial_port = serial.Serial(
            port=port,            
            baudrate=38400,       
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE, 
            bytesize=serial.EIGHTBITS,
            timeout=0.1 
        )
        # Modbus RTU 마스터 설정
        self.master = modbus_rtu.RtuMaster(self.serial_port) 
        self.master.set_timeout(0.1) 
        self.master.set_verbose(True) 
        self.stop_event = threading.Event() 
        self.save_count = 0 

        self.folder_path = folder_path  # 데이터 저장 폴더 경로
        self.exp_date = exp_date  # 실험 날짜
        self.exp_num = exp_num  # 실험 번호

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)  # 폴더가 없으면 생성

    def read_registers(self):
        try:
            # 홀딩 레지스터 읽기 (슬레이브 ID: 1, 시작 주소: 0, 레지스터 수: 7)
            holding_registers = self.master.execute(1, cst.READ_HOLDING_REGISTERS, 0, 7)
            # 입력 레지스터 읽기 (슬레이브 ID: 1, 시작 주소: 3, 레지스터 수: 1)
            input_registers = self.master.execute(1, cst.READ_INPUT_REGISTERS, 3, 1)
            # 데이터 형식 결정 및 출력
            tap_op = holding_registers[3]  # 탭 동작횟수
            tap_de_voltage = holding_registers[6] # 탭 원하는 전압
            tap_position = holding_registers[1]  # 탭 위치
            tap_voltage = input_registers[0]/2  # 탭 전압

            print(f"Desire Voltage: {tap_de_voltage} V, Tap Operations counter: {tap_op}, Current Tap position: {tap_position}, Current Tap voltage: {tap_voltage} V")
        
        except modbus_tk.modbus.ModbusError as e:
            # Modbus 에러 발생 시 출력
            print(f"Modbus error: {e}")
        except Exception as e:
            # 기타 에러 발생 시 출력
            print(f"Error reading registers: {e}")
        # 읽어온 레지스터 값 반환
        return tap_op, tap_de_voltage, tap_position, tap_voltage

    def start_reading(self, interval=0.1, save_interval=60, save_count_limit=98):
        self.save_count = 0  # 저장 횟수 초기화
        self.stop_event.clear()  # 스레드를 중지시키기 위한 이벤트 초기화
        self.thread = threading.Thread(target=self._update_registers, args=(interval, save_interval, save_count_limit))
        self.thread.start()

    def _update_registers(self, interval, save_interval, save_count_limit):
        last_save_time = time.time()
        while not self.stop_event.is_set() and self.save_count < save_count_limit:
            tap_op, tap_de_voltage, tap_position, tap_voltage = self.read_registers()
            if time.time() - last_save_time >= save_interval:
                self.save_to_file(tap_op, tap_de_voltage, tap_position, tap_voltage)
                last_save_time = time.time()
                self.save_count += 1
            time.sleep(interval)
        self.stop_event.set()  # 작업이 완료되면 스레드를 중지시킴

    def save_to_file(self, tap_op, tap_de_voltage, tap_position, tap_voltage):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 타임스탬프로 저장
        folder_name = f"{self.exp_date}_{self.exp_num}_sound"  # 폴더 이름 생성
        filename = os.path.join(self.folder_path, f'{folder_name}_{timestamp}.txt')  # 파일 경로 생성
        with open(filename, "a") as f:
            f.write(f"{timestamp}, {tap_op}, {tap_de_voltage}, {tap_position}, {tap_voltage}\n")

    def stop_reading(self):
        self.stop_event.set()
        self.thread.join()

사용 예제
if __name__ == "__main__":
    # ModbusRTUClient 인스턴스 생성 및 설정
    client = ModbusRTUClient(
        port='COM5',
        folder_path='C:\\Users\\vole9\\Downloads\\',  # 데이터 저장 폴더 경로
        exp_date='20230730',  # 실험 날짜
        exp_num='01'  # 실험 번호
    )
    # 레지스터 읽기 실행 (0.1초 간격으로)
    client.start_reading(interval=0.1, save_interval=60, save_count_limit=98)

    # 작업이 완료될 때까지 대기
    client.thread.join()
    print("Reading and saving completed.")
