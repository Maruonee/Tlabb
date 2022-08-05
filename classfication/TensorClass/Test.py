import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.image import pad_to_bounding_box
from tensorflow.image import central_crop
from tensorflow.image import resize
from keras_preprocessing.image import ImageDataGenerator
#================================================================================================
#세팅
base_dir = '/home/tlab/sono/'
model_name = "DenseNet201"
custom_batch = 16
model_dir = f"{base_dir}results/{model_name}_Best.h5"

#모델불러오기
from keras.models import load_model
model = load_model(model_dir)
model.summary()


import matplotlib.pyplot as plt
img_path = f'{base_dir}*.png'
img = image.load_img(img_path, target_size=(512, 512))
img_array = image.img_to_array(img)

plt.figure()
plt.imshow(img_array / 255)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255
prediction = model.predict(preprocessed_img)

print(np.array(prediction[0]))


# #이미지 불러오기
# bgd = image.load_img('C:/Users/Yeong/Desktop/CDSS강의_심화/심화_강의자료1/python_code/들3.jpg')
# bgd_vector = np.asarray(image.img_to_array(bgd))
# bgd_vector = bgd_vector/255
 
# #이미지 형태 확인 
# bgd_vector.shape
 
# #이미지 확인 
# plt.imshow(bgd_vector)
# plt.show()

# #이미지의 변경할 크기 설정 
# target_height = 4500
# target_width = 4500
 
# #현재 이미지의 크기 지정 
# source_height = bgd_vector.shape[0]
# source_width = bgd_vector.shape[1]
 
# #padding 실시 : pad_to_bounding_box 사용 
# bgd_vector_pad = pad_to_bounding_box(bgd_vector, 
#                                      int((target_height-source_height)/2), 
#                                      int((target_width-source_width)/2), 
#                                      target_height, 
#                                      target_width)
                                     
#  #이미지 형태 확인 
# bgd_vector_pad.shape
 
# #이미지 확인 
# plt.imshow(bgd_vector_pad)
# plt.show()
 
# #이미지 저장 
# image.save_img(r'C:\Users\Yeong\Desktop\CDSS강의_심화\심화_강의자료1\python_code\cat1_pad.png', cat_vector_pad)


# #가운데를 중심으로 50%만 crop 
# bgd_vector_crop = central_crop(bgd_vector, .5)
 
# bgd_vector_crop.shape
 
# plt.imshow(bgd_vector_crop)
# plt.show()


# w, h = bgd.size
 
# s = min(w, h) #둘 중에 작은 것 기준으로 자름 
# y = (h - s) // 2
# x = (w - s) // 2
 
# print(w, h, x, y, s)
 
# # 좌, 위, 오른쪽, 아래 픽셀 설정 
# bgd = bgd.crop((x, y, x+s, y+s))
# plt.imshow(np.asarray(bgd))
# bgd.size

# bgd_vector_resize = resize(bgd_vector, (300,300))
 
# bgd_vector_resize.shape
 
# plt.imshow(bgd_vector_resize)



# img = Image.open('들3.jpg')
# img.size
# plt.imshow(np.asarray(img))

# w, h = img.size
# s = min(w, h)
# y = (h - s) // 2
# x = (w - s) // 2
 
# print(w, h, x, y, s)
# img = img.crop((x, y, x+s, y+s))
# # 4-tuple defining the left, upper, right, and lower pixel coordinate
# plt.imshow(np.asarray(img))
# img.size


# #VGG16이 입력받는 이미지크기 확인
# model.layers[0].input_shape
 
# #이미지 리사이즈
# target_size = 224
# img = img.resize((target_size, target_size)) # resize from 280x280 to 224x224
# plt.imshow(np.asarray(img))
 
# img.size #변경된 크기 확인

# #numpy array로 변경
# np_img = image.img_to_array(img)
# np_img.shape  #(224, 224, 3) 
 
# #4차원으로 변경 
# img_batch = np.expand_dims(np_img, axis=0)
# img_batch.shape

# #feature normalization
# pre_processed = preprocess_input(img_batch)


# y_preds = model.predict(pre_processed)
 
# y_preds.shape  # 종속변수가 취할 수 있는 값의 수 = 1000
 
# np.set_printoptions(suppress=True, precision=10)
# y_preds
 
# #가장 확률이 높은 값
# np.max(y_preds)

# decode_predictions(y_preds, top=1)