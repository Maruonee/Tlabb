import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
from datetime import datetime
import dropbox
from tqdm import tqdm
import threading

def record_audio(duration, samplerate, channels, folder_path, files_to_upload):
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder_path, f'output_{timestamp}.wav')
        
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        
        for _ in tqdm(range(duration), desc=f"진행률"):
            time.sleep(1)
        sd.wait()
        write(filename, samplerate, recording)
        files_to_upload.append(filename)
        print(f"{filename} 저장완료.")

def file_dir(savedir):
    home_dir = os.path.expanduser("~")
    recordings_folder_path = os.path.join(home_dir, "Desktop", savedir)#바탕화면 폴더로 지정
    if not os.path.exists(recordings_folder_path):
        os.makedirs(recordings_folder_path)
    return recordings_folder_path

#===================================dropbox area===================================
def upload_to_dropbox(file_path, dropbox_path, dbx):
    try:
        with open(file_path, 'rb') as f:
            dbx.files_upload(f.read(), dropbox_path)
    except Exception as e:
        print(f"{file_path}에 업로드를 실패하였습니다. 에러코드 : {e}")

def upload_files_periodically(upload_interval, files_to_upload, dbx):
    while True:
        time.sleep(upload_interval)
        while files_to_upload:
            file_path = files_to_upload.pop(0)
            dropbox_path = f'/oltc/{os.path.basename(file_path)}'
            upload_to_dropbox(file_path, dropbox_path, dbx)

def load_dropbox_token(token_file_path):
    try:
        with open(token_file_path, 'r') as file:
            token = file.read().strip()
        return token
    except Exception as e:
        print(f"Dropbox token을 찾을 수 없습니다. 에러코드 : {e}")
        return None

def setup_dropbox(token_file_path):
    token = load_dropbox_token(token_file_path)
    if token:
        return dropbox.Dropbox(token)
    else:
        raise Exception("Dropbox token을 찾을 수 없습니다.")
#===================================dropbox area===================================

def token_dir(tokendir,token_file_name):
    token_file_path = os.path.join(tokendir, token_file_name)     #토큰경로
    if not os.path.exists(tokendir):
        os.makedirs(tokendir)#토큰경로
    return token_file_path

def main():
    duration = 5 # 녹음 지속 시간 (초)
    samplerate = 44100  # 샘플레이트 (Hz)
    upload_interval = 20  # 업로드 간격 (초)
    channels = 2 #서라운드
    
    # 바탕화면의 폴더 경로 지정
    recordings_folder_path = file_dir("soundfiles")
    
    #===================================dropbox area===================================
    # Dropbox 액세스 토큰 파일 경로 지정
    token_file_path = token_dir(recordings_folder_path,"dropbox.txt")
    # Dropbox 설정
    try:
        dbx = setup_dropbox(token_file_path)
    except Exception as e:
        print(e)
        return

    # 업로드 대기열
    files_to_upload = []
    
    #===================================dropbox area===================================
    try:
        print(f"=======녹음 시작======= \n시간 : {duration}초 \n저장위치 : {recordings_folder_path}")
        # 오디오 녹음 스레드 시작
        recording_thread = threading.Thread(target=record_audio, args=(duration, samplerate, channels, recordings_folder_path, files_to_upload)) 
        recording_thread.start()
    #===================================dropbox area===================================
        # 파일 업로드 스레드 시작
        upload_thread = threading.Thread(target=upload_files_periodically, args=(upload_interval, files_to_upload, dbx))
        upload_thread.start()
    #===================================dropbox area===================================
        recording_thread.join()
        upload_thread.join()
    except KeyboardInterrupt:
        print("사용자가 녹음을 중지하였습니다.")
    except Exception as e:
        print(f"녹음에러 : {e}")

if __name__ == "__main__":
    main()