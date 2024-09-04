import serial
import time

try:
    # 시리얼 포트 설정
    ser = serial.Serial('COM19', 19200, timeout=2)  # COM 포트와 보드레이트 확인
    time.sleep(2)  # 포트가 열리고 안정화될 시간을 주기 위해 지연

    # 현재 시간 가져오기
    current_time = time.strftime("%Y%m%d%H%M%S")  # YYYYMMDDHHMMSS 형식

    # 아두이노에 시간 전송
    ser.write(current_time.encode('utf-8'))
    ser.flush()  # 버퍼 비우기

except serial.SerialException as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("done.")
