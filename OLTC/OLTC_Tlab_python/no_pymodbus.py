from pymodbus.client import ModbusSerialClient
import time

# Modbus RTU 클라이언트 설정
client = ModbusSerialClient(
    method='rtu',
    port='COM5',       # 포트 이름
    baudrate=38400,    # 보드레이트
    parity='E',        # 패리티 (Even)
    stopbits=1,        # 정지 비트
    bytesize=8,        # 데이터 비트
    timeout=1          # 응답 시간 (초 단위)
)

# 클라이언트 연결 확인
if not client.connect():
    print("Unable to connect to the Modbus server. Please check the connection and settings.")
    exit()

# 레지스터 주소 설정
start_address = 0  # 실제 주소 (논리적 주소 40001에 해당)
quantity = 20

# 데이터 표기 방식 선택 함수
def format_data(data):
    # 데이터의 최대값과 최소값을 확인하여 형식을 결정
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
            result = client.read_holding_registers(address=start_address, count=quantity, unit=1)
            if not result.isError():
                register_values = result.registers
                data_format = format_data(register_values)
                print(f"Register values: {register_values} (Format: {data_format})")
            else:
                print(f"Error reading holding registers: {result}")

        except Exception as e:
            print(f"Error reading holding registers: {e}")

        # 폴링 간 지연 시간 20ms
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Terminating program...")

# 연결 종료
client.close()
