import numpy as np
import tensorflow as tf
import os
from knockknock import slack_sender
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
#========================================================
#슬랙 webhook주소
webhook_slack = "@@@@"
slack_channel = "#chestpa"

#데이터 및 컴퓨터 설정
img_dir = '/home/tlab1004/dataset/raw/test'
class_num = 2
cpu_core = 16
classes_name = ["Fail", "Pass"]
custom_class_mode = 'categorical'#"categorical", "binary", "sparse", "input", "other",'None'
#모델설정
model_dir = "/home/tlab1004/dataset/raw/123123/DenseNet201_Best.h5"
model_name = "DenseNet201"
#하이퍼파라미터 설정
custom_batch = 32
custom_image_size = (512, 512)
#========================================================

#테스트 데이터 셋
predict_datagen = ImageDataGenerator(
    rescale=1./255,
    )
predict_dir = os.path.join(img_dir)
predict_dataset = predict_datagen.flow_from_directory(
    predict_dir,
    batch_size=custom_batch,
    target_size=custom_image_size,
    class_mode=custom_class_mode,
    )

#모델불러오기
model = load_model(model_dir)
model.summary()

@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Predict(your_nicest_parameters='prediction'):
    #학습
    prediction = model.predict(
    predict_dataset,
    verbose='auto',
    workers=cpu_core
    )
    # print(type(prediction))
    # print(len(prediction))
    # print(prediction)
    # print(np.argmax(prediction))
    return f'Data number : {len(prediction)}, Model : {model_name}'
Predict()