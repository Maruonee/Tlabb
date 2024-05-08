import os
import glob

def delete_contents_in_files(directory):
    # 주어진 디렉토리에서 'not fallen'으로 시작하는 모든 .txt 파일 검색
    pattern = os.path.join(directory, 'not fallen*.txt')
    files = glob.glob(pattern)
    
    # 각 파일을 열어 내용을 삭제
    for file_path in files:
        with open(file_path, 'w') as file:
            file.truncate()

# 사용 예:
directory_path = '/home/tlab4090/datasets/labels'
delete_contents_in_files(directory_path)