from pymodbus.client import ModbusSerialClient as ModbusClient
from pymodbus.exceptions import ModbusException

# Modbus RTU 설정 파라미터
serial_port = 'COM5'  # 실제 연결된 시리얼 포트로 변경 필요
baud_rate = 9600
parity = 'E'
stop_bits = 1
bytesize = 8
slave_id = 1  # Modbus 장치의 슬레이브 ID

# Modbus 클라이언트 생성
client = ModbusClient(
    method='rtu',
    port=serial_port,
    baudrate=baud_rate,
    parity=parity,
    stopbits=stop_bits,
    bytesize=bytesize,
    timeout=3  # 타임아웃을 3초로 설정
)

# 연결 시도
connection = client.connect()

if connection:
    print("Modbus RTU 장치와의 연결이 성공적으로 이루어졌습니다.")
    try:
        # Holding Register address 1 읽기 시도 (Tap Position)
        result = client.read_holding_registers(1, 1, unit=slave_id)
        if not result.isError():
            tap_position = result.registers[0]
            print(f"Holding Register address 1 Tap Position: {tap_position}")
        else:
            print("Holding Register address 1 읽기 실패:", result)
    except ModbusException as e:
        print("Modbus 예외 발생:", e)
    finally:
        client.close()
else:
    print("Modbus RTU 장치와의 연결에 실패했습니다.")