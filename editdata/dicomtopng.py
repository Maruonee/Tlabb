import os
import pydicom
from PIL import Image
import numpy as np

# DICOM 파일이 있는 A 폴더 경로 및 PNG 파일을 저장할 B 폴더 경로 설정
input_folder = 'C:\\Users\\tlab\\Desktop\\cobb_dicom'
output_folder = 'C:\\Users\\tlab\\Desktop\\output'

# B 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# A 폴더의 모든 DICOM 파일을 순회하며 PNG로 변환
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.dcm'):
        # DICOM 파일 경로 지정
        dicom_path = os.path.join(input_folder, filename)
        
        # DICOM 파일 읽기
        dicom_data = pydicom.dcmread(dicom_path)
        
        # 픽셀 데이터를 numpy 배열로 변환
        image_array = dicom_data.pixel_array

        # 8비트로 변환 (이미지의 최소값과 최대값을 이용해 스케일링)
        image_array = image_array.astype(np.float32)
        scaled_image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
        scaled_image_array = scaled_image_array.astype(np.uint8)
        
        # numpy 배열을 PIL 이미지로 변환
        image = Image.fromarray(scaled_image_array)

        # 파일명에서 확장자 제거 후 PNG 형식으로 저장
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)
        image.save(output_path)
        
        print(f"{filename}을(를) {output_filename}로 변환 완료")

print("모든 DICOM 파일의 변환이 완료되었습니다.")
