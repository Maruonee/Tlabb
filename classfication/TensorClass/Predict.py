import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
#================================================================================================
#세팅
img_dir = '/home/tlab/sono/test/Fail'
model_name = "RegNetY320"
model_dir = "/home/tlab/sono/results/RegNetY320_Best.h5"
# img_file_name = '0133.tif'
#or
input_name = input("파일이름 입력 : ")
img_file_name = f"{input_name}.tif"
# train
#  -----/classname1
#  -----/classname2
#  -----/classname3
classes_name = ["Fail", "Pass"]
#================================================================================================
#모델불러오기
model = load_model(model_dir)
model.summary()

#이미지 불러오기
img_path = f'{img_dir}/{img_file_name}'
img = image.load_img(img_path, target_size=(512, 512))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255
prediction = model.predict(preprocessed_img)

#예측값 출력
print(np.array(classes_name)[np.argmax(prediction)])

#예측값 시각화
plt.figure()
plt.imshow(img_array/255)
plt.title(f'{img_path}')
plt.xlabel(f'{np.array(classes_name)[np.argmax(prediction)]}')
plt.savefig(f'{img_path}_results.png')
