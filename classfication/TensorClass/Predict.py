import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
#========================================================
#테스트 이미지
img_dir = "/XXXXX"
img_format = "tif"
custom_image_size = (512, 512)
input_name = input("파일이름 입력 : ")
#input_name = "n0333"
#모델
model_name = "VGG16"
#운영체제에 맞추어 변경해야함
model_dir = "XXXXXX.h5"
#클래스 이름
classes_name = ["Fail",
                "Pass"
                ]
#========================================================
#모델불러오기
model = load_model(model_dir)
model.summary()

#이미지 불러오기
img_path = os.path.join(img_dir, input_name , img_format)
img = image.img_to_array(
    image.load_img(img_path, target_size=custom_image_size)
    )
img_array = np.expand_dims(img, axis=0)
prediction = model.predict(img_array / 255)

#예측값 출력
print(np.array(classes_name)[np.argmax(prediction)])

#예측값 시각화
plt.figure()
plt.imshow(img_array)
plt.title(f'{model_name} predict {input_name}.{img_format}')
plt.xlabel(f'{np.array(classes_name)[np.argmax(prediction)]}')
plt.savefig(f'{img_path}_predict.png')
