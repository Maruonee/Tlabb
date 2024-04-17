import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from knockknock import slack_sender
from keras.models import load_model
import time
#====================================================================================
#슬랙
webhook_slack = "https://hooks.slack.com/services/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"
slack_channel = "ct_train"
#데이터 및 컴퓨터 설정
base_dir = '/home/tlab1004/datasets/Class/Contrast/'
class_num = 1 #binary is 1
cpu_core = 16
#하이퍼파라미터 설정
#참고 https://keras.io/ko/preprocessing/image/
custom_class_mode = 'binary'#"categorical", "binary", "sparse", "input", "other",'None'
custom_batch = 16
custom_epoch = 300
custom_learning_rate = 0.001
custom_image_size = (512, 512)
tuning_learning_rate = 0.00001
tuning_epoch = 100
# monitor_epoch = 100 #call back에포크

#====================================================================================
#1 모델설정
#https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
model_name = "VGG19"
custom_learning_rate = 0.001
model_loss_function = 'binary_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
custom_metrics = 'binary_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
base_model = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet',# 전이학습 가중치 imagenet or None
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=class_num,
    classifier_activation='softmax'# None or "softmax"
    )
#데이터 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )
#   추가 이미지 증강 옵션
#   rotation_range=40,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=True,
#   fill_mode=`nearest`
#   featurewise_center=False,
#   samplewise_center=False,
#   featurewise_std_normalization=False,
#   samplewise_std_normalization=False,
#   zca_whitening=False,
#   zca_epsilon=1e-06,
#   rotation_range=0,
#   width_shift_range=0.0,
#   height_shift_range=0.0,
#   brightness_range=None,
#   shear_range=0.0,
#   zoom_range=0.0,
#   channel_shift_range=0.0,
#   fill_mode='nearest',
#   cval=0.0,
#   horizontal_flip=False,
#   vertical_flip=False,
#   rescale=None,
#   preprocessing_function=None,
#   data_format=None,
#   validation_split=0.0,
#   dtype=None
# 최적화 https://keras.io/api/optimizers/ 참고
model_optimizer = tf.keras.optimizers.Adam(
    learning_rate=custom_learning_rate
    )
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-07,
    # amsgraid=False,
    # weight_decay=None,
    # clipnorm=None,
    # clipvalue=None,
    # global_clipnorm=None,
    # use_ema=False,
    # ema_momentum=0.99,
    # ema_overwrte_frequency=None,
    # jit_compile=True,
    # name="Adam"
#====================================================================================
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    )
#저장위치
save_dir = os.path.join(base_dir,"results")
best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
last_model_dir = os.path.join(save_dir,f"{model_name}.h5")
fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
# # 조기종료 옵션
# callback_list=[
#     keras.callbacks.EarlyStopping(
#         monitor=f'val_{custom_metrics}',
#         patience=monitor_epoch, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
#         mode='auto',
#         restore_best_weights=True #가장 좋았던값으로 가중치를 저장
#     ),
#     #에포크마다 현재 가중치를 저장
#     keras.callbacks.ModelCheckpoint(
#         filepath=best_model_dir,
#         monitor=custom_metrics,
#         mode='auto',
#         save_best_only=True #가장 좋았던값으로 가중치를 저장
#     ),
# ]
#데이터 셋
train_dir = os.path.join(base_dir,'train')
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=custom_image_size,
    batch_size=custom_batch,
    class_mode=custom_class_mode
    )
validation_dir = os.path.join(base_dir,'val')
validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=custom_image_size,
    batch_size=custom_batch,
    class_mode=custom_class_mode
    )
#테스트 데이터 셋
test_dir = os.path.join(base_dir,'test')
test_dataset = validation_datagen.flow_from_directory(
    test_dir,
    batch_size=custom_batch,
    target_size=custom_image_size,
    class_mode=custom_class_mode,
    )

# print(len(train_dataset.keys()))

print(type(train_dataset))
print(train_dataset.samples)
