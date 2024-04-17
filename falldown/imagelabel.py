import json
import os
import glob

def roundtosix(number):
    # 숫자를 소수점 아래 6자리까지 포맷팅
    return "{:.6f}".format(number)

folder_path = 'C:\\Users\\tlab\\Desktop\\sample2\\'
json_files = glob.glob(os.path.join(folder_path, '*.json'))

for json_file in json_files:
    # 파일 이름에서 경로를 제외하고 확장자를 제거하여 결과 문자열 생성
    result = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
    save_folder_path = os.path.join(folder_path, result)

    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # "bboxes" 부분 추출
    bboxes = data['content']['object']['annotation'].get('bboxes', [])
    
    # bboxes가 비어있는 경우 모든 값을 0으로 설정
    if not bboxes:
        formatted_data = "0 0.000000 0.000000 0.000000 0.000000"
    else:
        x_min, y_min, x_max, y_max = bboxes[0]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        formatted_data = f"0 {roundtosix(x_center)} {roundtosix(y_center)} {roundtosix(width)} {roundtosix(height)}"
    
    with open(save_folder_path, 'w') as txt_file:
        txt_file.write(formatted_data)
    print(f"{json_file}을{save_folder_path}로 저장완료")