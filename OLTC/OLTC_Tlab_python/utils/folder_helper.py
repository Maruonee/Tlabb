import os  # 운영 체제 모듈 가져오기

def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"  # 폴더 이름 생성
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)  # 폴더 경로 생성
    os.makedirs(recordings_folder_path, exist_ok=True)  # 폴더 생성 (존재하지 않는 경우)
    return recordings_folder_path  # 폴더 경로 반환
