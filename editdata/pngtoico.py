from PIL import Image

# PNG 파일 경로
png_file = 'C:\\Users\\tlab\\Desktop\\123.png'

# 변환된 ICO 파일 저장 경로
ico_file = 'C:\\Users\\tlab\\Desktop\\123.ico'

# 이미지 열기
img = Image.open(png_file)

# 이미지 변환 및 저장
img.save(ico_file, format='ICO')