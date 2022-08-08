import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from knockknock import slack_sender
from keras.models import load_model
import time
#========================================================
#세팅값 입력
webhook_slack = "https://hooks.slack.com/services/xxxxxxx"
slack_channel = "#xxxxxxx"
base_dir = '/xxxxxxxx/'
model_dir = "/xxxxx/xxxxx.h5"
model_name = "DenseNet201" # DenseNet201, InceptionResNetV2, InceptionV3, Xception, ResNet50, ResNetRS50 ,ResNet50V2, RegNetY320, NASNetLarge, VGG16, VGG19
custom_batch = 16
class_num = 2
tuning_learning_rate = 0.00001 #tunning 학습률
tuning_epoch = 100
cpu_core = 16
base_model = tf.keras.applications.densenet.DenseNet201(
    # tf.keras.applications.densenet.DenseNet201
    # tf.keras.applications.inception_resnet_v2.InceptionResNetV2
    # tf.keras.applications.inception_v3.InceptionV3
    # tf.keras.applications.xception.Xception
    # tf.keras.applications.resnet50.ResNet50
    # tf.keras.applications.resnet_rs.ResNetRS50
    # tf.keras.applications.resnet_v2.ResNet50V2
    # tf.keras.applications.regnet.RegNetY320
    # tf.keras.applications.nasnet.NASNetLarge
    # tf.keras.applications.vgg16.VGG16
    # tf.keras.applications.vgg19.VGG19
    include_top=False,
    weights='imagenet',#전이학습 가중치
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation='softmax'
    )
#========================================================
# 데이터셋 구성
# dataset
#   train
#       -----/class1
#       -----/class2
#       -----/class3
#   val
#      -----/class1
#      -----/class2
#      -----/class3
#   test
#      -----/class1
#      -----/class2
#      -----/class3
#   results
#========================================================
#저장위치
save_dir = f"{base_dir}results/"

#학습 데이터셋
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )
#========================================================
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
#========================================================
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

#모델 정의
model = load_model(model_dir)

#튜닝 실행
model.trainable = True

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=tuning_learning_rate), #최적화 설정
    metrics=['accuracy']
    )

# 모델 요약 출력
model.summary()

#튜닝학습 정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Fine_tuning_CT_Resolution(your_nicest_parameters='f_hist'):
    f_hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=tuning_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core
    )
    
    #학습모델 저장
    model.save(f"{save_dir}{model_name}_Fine_tuning.h5")
    
    #학습 결과 시각화
    acc = f_hist.history['accuracy']
    val_acc = f_hist.history['val_accuracy']
    loss = f_hist.history['loss']
    val_loss = f_hist.history['val_loss']
    
    #학습 손실 시각화
    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tunning Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f'{save_dir}loss_{model_name}_Fine_tuning.png')
    
    #학습 정확도 시각화
    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tunning Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.savefig(f'{save_dir}accuracy_{model_name}_Fine_tuning.png')
    
    #테스트
    f_loss, f_accuracy = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    
    # 테스트 결과 출력
    print(f'Fine tuning loss : {f_loss}')
    print(f'Fine tuning accuracy : {f_accuracy}')
    time.sleep(3)
    
    #slack에 출력
    return f'\n {model_name} Fine tuning Train accuracy : {max(acc)}\n{model_name} Fine tuning test loss : {f_loss}\n{model_name} Fine tuning test accuracy : {f_accuracy}'


#실행
Fine_tuning_CT_Resolution()   

#다음 슬랙을 위한 대기시간 설정
print("Waiting for next trainning")
time.sleep(61)
print("Done!")