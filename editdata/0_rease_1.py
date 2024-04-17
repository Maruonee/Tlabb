
#1을 지우는 코드
#==================================================================
# import os

# def clean_files(directory):
#     # 지정된 디렉토리에서 모든 파일을 검색합니다.
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):  # txt 확장자를 가진 파일만 처리
#             filepath = os.path.join(directory, filename)
#             with open(filepath, 'r') as file:
#                 lines = file.readlines()
            
#             # 1로 시작하는 줄을 제외하고 새로운 줄 목록을 생성
#             new_lines = [line for line in lines if not line.startswith('1')]
            
#             # 수정된 내용으로 파일을 다시 쓰기
#             with open(filepath, 'w') as file:
#                 file.writelines(new_lines)

# # 함수를 호출하여 지정된 폴더 내의 파일을 정리
# clean_files('/home/tlab4090/datasets/sample')
#==================================================================

# 2를 1로 바꾸는 코드
import os

def modify_files(directory):
    # 지정된 디렉토리에서 모든 파일을 검색합니다.
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # txt 확장자를 가진 파일만 처리
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            # 2로 시작하는 줄의 첫 번째 숫자를 1로 변경
            new_lines = []
            for line in lines:
                if line.startswith('2'):
                    # 첫 번째 숫자 2를 1로 변경하고 나머지는 그대로 유지
                    parts = line.split(' ', 1)
                    new_line = '1 ' + parts[1]
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            # 수정된 내용으로 파일을 다시 쓰기
            with open(filepath, 'w') as file:
                file.writelines(new_lines)

# 함수를 호출하여 지정된 폴더 내의 파일을 수정
modify_files('/home/tlab4090/datasets/labels')
