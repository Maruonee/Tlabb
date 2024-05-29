import serial
import time
import os
from datetime import datetime
from threading import Thread, Lock
from tqdm import tqdm

# 설정값 입력 함수
def get_settings():
    settings = {
        "serial_port": 'COM3',  # 아두이노가 연결된 포트
        "savedir": "C:\\Users\\,"  # 저장위치
        "duration": 3,  # 파일 교체 주기(초)
        "samplerate": 19200,  # 아두이노와 일치하는 보드레이트
        "repeat_num": 10,  # 반복 횟수
        "exp_num": 1,  # 실험횟수
        "exp_date": 240529  # 실험날짜
    }
    return settings

# 새로운 파일 생성 함수
def create_new_file(folder_path, exp_date, exp_num):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{exp_date}_{exp_num}_sensors"
    filename = os.path.join(folder_path, f'{folder_name}_{timestamp}.txt')
    return open(filename, mode='w'), filename

# 파일 갱신 스레드 함수
def file_refresh_thread(folder_path, exp_date, exp_num, txt_file_ref, duration, repeat_num, lock):
    for i in range(repeat_num):
        for _ in tqdm(range(duration), desc=f"{i+1}번째 진행률", unit="초"):
            time.sleep(1)
        new_file, filename = create_new_file(folder_path, exp_date, exp_num)
        with lock:
            txt_file_ref[0].close()
            txt_file_ref[0] = new_file
        print(f"저장된 파일: {filename}")
    print("데이터 획득을 모두 완료하였습니다.")

# 데이터 수집 및 기록 함수
def collect_and_save_data(serial_port, samplerate, folder_path, exp_date, exp_num, duration, repeat_num):
    try:
        ser = serial.Serial(serial_port, samplerate)
    except serial.SerialException as e:
        print(f"직렬 포트를 열 수 없습니다: {e}")
        return

    txt_file_ref, initial_filename = create_new_file(folder_path, exp_date, exp_num)
    print(f"저장된 파일: {initial_filename}")
    txt_file_ref = [txt_file_ref]
    lock = Lock()

    file_refresh_thread_obj = Thread(target=file_refresh_thread, args=(folder_path, exp_date, exp_num, txt_file_ref, duration, repeat_num, lock), daemon=True)
    file_refresh_thread_obj.start()

    try:
        while file_refresh_thread_obj.is_alive():
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()  # 데이터 읽기 및 디코딩
                if data:  # 데이터가 있는 경우에만 기록
                    with lock:
                        txt_file_ref[0].write(f'{data}\n')
                        txt_file_ref[0].flush()
    except KeyboardInterrupt:
        print("데이터 수집이 중지되었습니다.")
    finally:
        with lock:
            txt_file_ref[0].close()
        ser.close()

# 폴더 생성 함수
def create_folder(savedir, exp_date, exp_num):
    folder_name = f"{exp_date}_{exp_num}_sensors"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

# 메인 함수
def main():
    settings = get_settings()
    recordings_folder_path = create_folder(settings["savedir"], settings["exp_date"], settings["exp_num"])
    print(f"=======데이터 획득 시작=======\n시간 : {settings['duration']}초\n반복횟수 : {settings['repeat_num']}\n저장위치 : {recordings_folder_path}\n")
    collect_and_save_data(settings["serial_port"], settings["samplerate"], recordings_folder_path, settings["exp_date"], settings["exp_num"], settings["duration"], settings["repeat_num"])

if __name__ == "__main__":
    main()