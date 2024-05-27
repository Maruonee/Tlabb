import time
import pyautogui
import subprocess

def start_audacity():
    # Audacity 실행 (경로는 시스템에 따라 다를 수 있음)
    subprocess.Popen(['C:\\Program Files\\Audacity\\audacity.exe'])
r
def export_recording(file_name):
    # 파일 내보내기 (단축키 'Ctrl+Shift+E')
    pyautogui.hotkey('ctrl', 'shift', 'e')
    time.sleep(1)  # 파일 내보내기 창이 뜰 때까지 대기
    pyautogui.write(file_name)
    pyautogui.press('enter')
    time.sleep(1)  # 파일 형식 선택 창이 뜰 시간을 대기
    pyautogui.press('enter')
    
def close_audacity():
    # Audacity 종료
    pyautogui.hotkey('alt', 'f4')
    time.sleep(1) 
    pyautogui.press('n')
# 기본 FLAC 형식을 선택하고 저장

def main():
    for i in range(10):  # 10번 반복 (10분 동안 녹음)
        start_audacity()
        time.sleep(5)
        pyautogui.press('r')     # 녹음 시작 (단축키 'R')
        time.sleep(5)  # 1분 동안 녹음
        pyautogui.press('space') # 녹음 중지 (단축키 'Space')
        export_recording(f'recording_{i}.flac')
        time.sleep(1)
        close_audacity()
        
if __name__ == "__main__":
    main()