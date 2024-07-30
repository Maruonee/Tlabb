import minimalmodbus
import time

# 시리얼 포트 설정
instrument = minimalmodbus.Instrument('COM5', 1)  # 포트 이름과 슬레이브 주소 설정
instrument.serial.baudrate = 38400
instrument.serial.bytesize = 8
instrument.serial.parity = minimalmodbus.serial.PARITY_EVEN
instrument.serial.stopbits = 1
instrument.serial.timeout = 1  # 초 단위

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
            register_values = instrument.read_registers(start_address, quantity, functioncode=3)
            data_format = format_data(register_values)
            print(f"Register values: {register_values} (Format: {data_format})")

        except Exception as e:
            print(f"Error reading holding registers: {e}")

        # 폴링 간 지연 시간 20ms
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Terminating program...")
