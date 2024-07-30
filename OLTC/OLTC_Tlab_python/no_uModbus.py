import time
import serial
from umodbus.client import serializers as s
from umodbus.client.serial import rtu

# 시리얼 포트 설정
serial_port = serial.Serial(
    port='COM5',            # 포트 이름
    baudrate=38400,         # 보드레이트
    parity=serial.PARITY_EVEN,  # 패리티
    stopbits=serial.STOPBITS_ONE,  # 정지 비트
    bytesize=serial.EIGHTBITS,  # 데이터 비트
    timeout=1               # 응답 시간 (초 단위)
)

# Modbus RTU 클라이언트 설정
client = rtu.RtuMaster(serial_port)
client.set_timeout(1.0)
client.set_verbose(True)

# 레지스터 주소 설정
start_address = 0  # 실제 주소 (논리적 주소 40001에 해당)
quantity = 20

# 데이터 표기 방식 선택 함수
def format_data(data):
    max_val = max(data)
    min_val = min(data)
    
    if max_val <= 0xFFFF and min_val >= 0:
        return "Unsigned Integer"
    elif min_val < 0:
        return "Signed Integer"
    elif any(isinstance(i, float) for i in data):
        return "Float"
    else:
        return "Unknown"

# 스캔 주기
scan_rate = 1  # 1000ms (1초)

# 데이터 읽기 및 출력
try:
    while True:
        try:
            # 홀딩 레지스터 읽기
            request = rtu.read_holding_registers(slave_id=1, starting_address=start_address, quantity=quantity)
            response = client.send_message(request)
            register_values = list(response)
            data_format = format_data(register_values)
            print(f"Register values: {register_values} (Format: {data_format})")

        except Exception as e:
            print(f"Error reading holding registers: {e}")

        # 폴링 간 지연 시간 20ms
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Terminating program...")
finally:
    serial_port.close()
