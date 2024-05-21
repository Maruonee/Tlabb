import shutil
import os

# 파일 리스트가 들어 있는 텍스트 파일 경로
file_list_path = "/home/tlab4090/datasets/opt/test.txt"

# 목적 디렉터리
destination_directory = "/home/tlab4090/datasets/test"
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# 텍스트 파일에서 파일 경로들을 읽어들입니다.
with open(file_list_path, 'r') as file:
    file_paths = file.readlines()
# 파일 경로에서 불필요한 공백 문자 제거
file_paths = [file_path.strip() for file_path in file_paths]


# 파일들을 복사합니다.
for file_path in file_paths:
    shutil.copy(file_path, destination_directory)
    print(f"파일 복사 완료: {file_path}")
print("모든 파일 복사가 완료되었습니다.")