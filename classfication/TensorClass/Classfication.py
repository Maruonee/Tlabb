""" 데이터셋 구성
train
 -----/class1
 -----/class2
 -----/class3
val
 -----/class1
 -----/class2
 -----/class3
"""
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from tensorflow.compat.v1.keras.backend import set_session
from datetime import datetime
import sys
#================================================================================================
#세팅
base_dir = '/home/tlab/sono/'
save_dir = "/home/tlab/Results/Classification/"
custom_batch = 16
custom_epochs = 2000
class_num = 2
model_name = "InceptionV3"
custom_learning_rate = 0.001
#call back에서 설정값
monitor_factor = 'val_accuracy'
#call back에서 에포크설정동안 정확도가 향상되지 않으면 훈련 중지
monitor_epochs = 100
#================================================================================================
"""
알고리즘 종류
tf.keras.applications.densenet.DenseNet201
tf.keras.applications.inception_resnet_v2.InceptionResNetV2
tf.keras.applications.inception_v3.InceptionV3
tf.keras.applications.xception.Xception
tf.keras.applications.resnet50.ResNet50
tf.keras.applications.resnet_rs.ResNetRS50
tf.keras.applications.resnet_v2.ResNet50V2
tf.keras.applications.regnet.RegNetY320
tf.keras.applications.nasnet.NASNetLarge
tf.keras.applications.vgg16.VGG16
tf.keras.applications.vgg19.VGG19
"""
#================================================================================================
#callback 옵션, 성능 향상이 멈추면 훈련을 중지
callback_list=[
    keras.callbacks.EarlyStopping(
        monitor=monitor_factor,
        patience=monitor_epochs, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
        mode='auto',
        restore_best_weights=True #가장 좋았던값으로 가중치를 저장
        ),
#에포크마다 현재 가중치를 저장
    keras.callbacks.ModelCheckpoint(
        filepath=f"{save_dir}{model_name}_Best.h5",
        monitor='accuracy',
        mode='auto',
        save_best_only=True #가장 좋았던값으로 가중치를 저장
        )
    ]
"""
이미지 증강 옵션
       rotation_range=40,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       fill_mode=`nearest`
       featurewise_center=False,
       samplewise_center=False,
       featurewise_std_normalization=False,
       samplewise_std_normalization=False,
       zca_whitening=False,
       zca_epsilon=1e-06,
       rotation_range=0,
       width_shift_range=0.0,
       height_shift_range=0.0,
       brightness_range=None,
       shear_range=0.0,
       zoom_range=0.0,
       channel_shift_range=0.0,
       fill_mode='nearest',
       cval=0.0,
       horizontal_flip=False,
       vertical_flip=False,
       rescale=None,
       preprocessing_function=None,
       data_format=None,
       validation_split=0.0,
       dtype=None
"""
#트레이닝 이미지 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )
#검증 이미지 생성
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )
"""class옵션
categorical : 2D one-hot 부호화된 라벨이 반환
binary : 1D 이진 라벨이 반환
sparse : 1D 정수 라벨이 반환
None : 라벨이 반환되지 않음
"""

#학습 데이터셋
train_dir = os.path.join(base_dir,'train')
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=custom_batch,
    class_mode='categorical',
    )
#검증 데이터셋
validation_dir= os.path.join(base_dir,'val')
validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(512, 512),
    batch_size=custom_batch,
    class_mode='categorical'
    )
#================================================================================================
#모델설정
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    #전이학습 종류
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation='softmax'
    )
"""
알고리즘 종류
tf.keras.applications.densenet.DenseNet201
tf.keras.applications.inception_resnet_v2.InceptionResNetV2
tf.keras.applications.inception_v3.InceptionV3
tf.keras.applications.xception.Xception
tf.keras.applications.resnet50.ResNet50
tf.keras.applications.resnet_rs.ResNetRS50
tf.keras.applications.resnet_v2.ResNet50V2
tf.keras.applications.regnet.RegNetY320
tf.keras.applications.nasnet.NASNetLarge
tf.keras.applications.vgg16.VGG16
tf.keras.applications.vgg19.VGG19
"""
#================================================================================================
#기본모델 멈춤
base_model.trainable = False

#아웃레이어 세팅
out_layer = tf.keras.layers.Conv2D(512, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(class_num, activation='softmax')(out_layer)
model = tf.keras.models.Model(base_model.input, out_layer)

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate),
    metrics=['accuracy']
    )

#로그 저장
# sys.stdout = open(f'{save_dir}{model_name}_log.txt', 'w')

# 모델 요약 출력
model.summary()

# 학습진행
hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epochs,
    validation_data=validation_dataset,
    verbose=1,
    workers=16,
    callbacks=[callback_list]
    )
    
#모델저장
model.save(f"{save_dir}{model_name}_Last.h5")

#시각화세팅
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#손실 시각화
plt.figure()
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig(f'{save_dir}{model_name}Loss.png')

#정확도 시각화
plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.savefig(f'{save_dir}{model_name}Accuracy.png')

#로그저장종료
# sys.stdout.close()

