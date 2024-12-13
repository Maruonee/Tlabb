import h5py
from PIL import Image
import numpy as np
import os

# PNG 파일을 HDF5로 변환하는 함수
def png_to_h5(png_path, h5_path):
    # PNG 이미지 열기
    with Image.open(png_path) as img:
        img_array = np.array(img)

    # HDF5 파일 생성
    with h5py.File(h5_path, 'w') as h5_file:
        # 이미지 데이터를 HDF5 파일에 저장
        h5_file.create_dataset('image', data=img_array)

    print(f"HDF5 파일이 {h5_path}에 저장되었습니다.")

# 디렉토리 내 모든 PNG 파일 변환 함수
def convert_png_dir_to_h5(png_dir, h5_dir):
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)

    for filename in os.listdir(png_dir):
        if filename.endswith(".png"):
            png_path = os.path.join(png_dir, filename)
            h5_path = os.path.join(h5_dir, filename.replace('.png', '.h5'))
            png_to_h5(png_path, h5_path)

# 예시 사용법
png_directory = '/home/tlab4090/Downloads/origin'  # PNG 파일들이 있는 디렉토리
h5_directory = '/home/tlab4090/Downloads/h5'   # 변환된 HDF5 파일을 저장할 디렉토리

convert_png_dir_to_h5(png_directory, h5_directory)