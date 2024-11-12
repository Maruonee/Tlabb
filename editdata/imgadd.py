import os
import cv2

# 폴더 경로 설정
folder_a = 'C:\\Users\\tlab\\Desktop\\img'  # a 폴더 경로
folder_b = 'C:\\Users\\tlab\\Desktop\\label'  # b 폴더 경로
output_folder = f'{folder_a}_add'  # 결과 이미지 저장 폴더

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 각 폴더의 이미지 파일 리스트 가져오기
files_a = set([f for f in os.listdir(folder_a) if f.endswith('.png')])
files_b = set([f for f in os.listdir(folder_b) if f.endswith('.png')])

# 두 폴더에 공통으로 있는 파일만 선택
common_files = files_a.intersection(files_b)


for file_name in common_files:
    # 각 폴더에서 동일한 이름의 이미지 파일 경로 설정
    path_a = os.path.join(folder_a, file_name)
    path_b = os.path.join(folder_b, file_name)
    
    # 이미지 불러오기
    image_a = cv2.imread(path_a)
    image_b = cv2.imread(path_b)
    
    # 이미지 크기와 채널 수 확인
    if image_a.shape != image_b.shape:
        print(f"{file_name}의 이미지 크기가 다름")
        continue
    
    # 이미지 더하기
    combined_image = cv2.add(image_a, image_b)
    
    # 결과 이미지 저장
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, combined_image)
    
    print(f"{file_name} done.")
print("all done.")