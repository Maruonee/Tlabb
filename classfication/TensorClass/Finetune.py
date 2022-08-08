import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from knockknock import slack_sender
from keras.models import load_model
import time
#================================================================================================
#  데이터셋 구성
# train
#  -----/class1
#  -----/class2
#  -----/class3
# val
#  -----/class1
#  -----/class2
#  -----/class3
#================================================================================================
#세팅값 입력
base_dir = '/home/tlab/sono/'
model_name = "RegNetY320" 
model_dir = "/home/tlab/sono/results/RegNetY320_Best.h5"
# DenseNet201, InceptionResNetV2, InceptionV3, Xception, ResNet50, ResNetRS50 ,ResNet50V2, RegNetY320, NASNetLarge, VGG16, VGG19
custom_batch = 16
custom_epochs = 100
class_num = 2
custom_learning_rate = 0.00001
#================================================================================================

#모델 저장위치 설정
save_dir = f"{base_dir}results/"
# 이미지 증강 옵션
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode=`nearest`
#        featurewise_center=False,
#        samplewise_center=False,
#        featurewise_std_normalization=False,
#        samplewise_std_normalization=False,
#        zca_whitening=False,
#        zca_epsilon=1e-06,
#        rotation_range=0,
#        width_shift_range=0.0,
#        height_shift_range=0.0,
#        brightness_range=None,
#        shear_range=0.0,
#        zoom_range=0.0,
#        channel_shift_range=0.0,
#        fill_mode='nearest',
#        cval=0.0,
#        horizontal_flip=False,
#        vertical_flip=False,
#        rescale=None,
#        preprocessing_function=None,
#        data_format=None,
#        validation_split=0.0,
#        dtype=None

#학습 데이터셋
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )
train_dir = os.path.join(base_dir,'train')
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=custom_batch,
    class_mode='categorical'
    )
#검증 데이터셋
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    )
validation_dir= os.path.join(base_dir,'val')
validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(512, 512),
    batch_size=custom_batch,
    class_mode='categorical'
    )
#테스트 데이터 셋
test_datagen = ImageDataGenerator(
    rescale=1./255,
    )
test_dir = os.path.join(base_dir,'test')
test_dataset = test_datagen.flow_from_directory(
    test_dir,
    batch_size=custom_batch,
    target_size=(512, 512),
    class_mode='categorical',
    classes = None
    )
#================================================================================================
#모델설정
model = load_model(model_dir)
#전체 모델 학습가능
model.trainable = True
#==================================================================================
# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate), #최적화 설정
    metrics=['accuracy']
    )

# 모델 요약 출력
model.summary()

#Slack 알람 설정
webhook_url = "https://hooks.slack.com/services/xxxxxxx"
@slack_sender(webhook_url=webhook_url, channel="#training")

#학습정의
def Fine_tuning_Sono_Axial_classification(your_nicest_parameters='hist'):
    #학습
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epochs,
    validation_data=validation_dataset,
    verbose=1,
    workers=16
    )
    model.save(f"{save_dir}{model_name}_Fine_tuning.h5")
    #학습 결과 시각화
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    #학습 손실 시각화
    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f'{save_dir}loss_{model_name}_Fine_tuning.png')
    #학습 정확도 시각화
    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.savefig(f'{save_dir}accuracy_{model_name}_Fine_tuning.png')
    #마지막 가중치 테스트
    f_loss, f_accuracy = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=16,
        use_multiprocessing=False,
        return_dict=False,
        )
    model.close()
    # 테스트 결과 출력
    print(f'Fine_tuning loss : {f_loss}')
    print(f'Fine_tuning accuracy : {f_accuracy}')
    time.sleep(3)
    return f'\n {model_name} Fine_tuning_Train accuracy : {max(acc)}\n{model_name} Fine_tuning_test_loss : {f_loss}\n{model_name} Fine_tuning_test_accuracy : {f_accuracy}'

# 실행
Fine_tuning_Sono_Axial_classification()
#다음 슬랙을 위한 대기시간 설정
time.sleep(61)
print("Done!")