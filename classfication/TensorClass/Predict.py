import numpy as np
import tensorflow as tf
import os
from knockknock import slack_sender
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils.score import recall_m, precision_m, f1_score_m
# from google.protobuf.pyext import _message
#========================================================
#슬랙 webhook주소
webhook_slack = "https://hooks.slack.com/services/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"
slack_channel = "#chestpa"

#데이터 및 컴퓨터 설정
img_dir = '/home/tlab4090/datasets/pene'
class_num = 2
cpu_core = 16 
classes_name = ["Fail", "Pass"]
custom_class_mode = 'categorical'#"categorical", "binary", "sparse", "input", "other",'None'
#모델설정
model_dir = "/home/tlab4090/datasets/sono_pene/DenseNet201_Fine_tuning.h5"
model_name = "DenseNet201"
#하이퍼파라미터 설정
custom_batch = 16
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
model = load_model(model_dir, custom_objects={"recall_m": recall_m, "precision_m" : precision_m, "f1_score_m" : f1_score_m},compile=False)
model.summary()

@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Predict_and_return_object(your_nicest_parameters='prediction'):
    #학습
    prediction = model.predict(
    predict_dataset,
    verbose='auto',
    workers=cpu_core
    )
    # 파일 이름과 예측 결과를 딕셔너리로 저장
    predictions_dict = {}
    
    for i, (pred, filename) in enumerate(zip(prediction, predict_dataset.filenames)):
        class_0_prob = pred[0]  # 클래스 0에 대한 확률
        class_1_prob = pred[1]  # 클래스 1에 대한 확률
        
        # 딕셔너리의 키는 파일 이름, 값은 클래스 0과 클래스 1의 확률
        predictions_dict[filename] = {
            "클래스 0 확률": class_0_prob,
            "클래스 1 확률": class_1_prob
        }
    
    # 예측 결과 딕셔너리 반환
    return predictions_dict


predictions = Predict_and_return_object()

for filename, probs in predictions.items():
    print(f"파일: {filename}, 클래스 0 확률 = {probs['클래스 0 확률']:.4f}, 클래스 1 확률 = {probs['클래스 1 확률']:.4f}")