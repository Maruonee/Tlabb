import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
from datetime import datetime
from tqdm import tqdm
import threading
import keyboard

##세팅값 입력
def get_settings():
    settings = {
        "savedir": "C:\\Users", # 저장위치
        "duration": 5, # 녹음시간 (초)
        "samplerate": 44100, # 샘플링레이트
        "channels": 2, # 1: 모노, 2: 스테레오
        "repeat_num": 60, # 반복횟수
        "exp_num": 1, # 실험횟수
        "exp_date": 240530 # 실험날짜
    }
    return settings

# 데이터 수집 및 기록 함수
def record_audio(duration, samplerate, channels, folder_path, repeat_num, stop_event, exp_date, exp_num):
    for i in range(repeat_num):
        if stop_event.is_set():
            print("녹음 중지 요청됨.")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{exp_date}_{exp_num}_sound"
        filename = os.path.join(folder_path, f'{folder_name}_{timestamp}.wav')

        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        
        for _ in tqdm(range(duration), desc=f"{i+1}번째 진행률"):
            if stop_event.is_set():
                sd.stop()
                break
            time.sleep(1)
        if stop_event.is_set():
            break
        sd.wait()

        write(filename, samplerate, recording)
        print(f"{filename} 저장완료.")

def create_folder(savedir, exp_date, exp_num):
    folder_name = f"{exp_date}_{exp_num}_sound"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path

def main():
    settings = get_settings()
    
    recordings_folder_path = create_folder(settings["savedir"], settings["exp_date"], settings["exp_num"])
    stop_event = threading.Event()

    print(f"=======녹음 시작=======\n시간 : {settings['duration']}초\n반복횟수 : {settings['repeat_num']}\n저장위치 : {recordings_folder_path}\n")
    print("중지하려면 'q' 키를 누르세요.")
    
    recording_thread = threading.Thread(target=record_audio, args=(
        settings["duration"], settings["samplerate"], settings["channels"], recordings_folder_path, settings["repeat_num"], stop_event, settings["exp_date"], settings["exp_num"]
    )) 
    recording_thread.start()

    while recording_thread.is_alive():
        if keyboard.is_pressed('q'):
            stop_event.set()
            break

    recording_thread.join()
    print("녹음이 중지되었습니다.")

if __name__ == "__main__":
    main()