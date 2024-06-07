import os

def create_folder(savedir, exp_date, exp_num, suffix):
    folder_name = f"{exp_date}_{exp_num}_{suffix}"
    recordings_folder_path = os.path.join(savedir, str(exp_date), str(exp_num), folder_name)
    os.makedirs(recordings_folder_path, exist_ok=True)
    return recordings_folder_path