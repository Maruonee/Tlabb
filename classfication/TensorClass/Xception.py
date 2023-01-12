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
webhook_slack = "https://hooks.slack.com/services/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"
slack_channel = "#chest_train"
base_dir = '/home/tlab1004/dataset/images'#운영체제에 맞추어 변경해야함
#                                      'Sono_Penet' 를 원하는 이름으로 변경
custom_batch = 32
custom_epochs = 1000
custom_learning_rate = 0.0001
class_num = 2
cpu_core = 16
custom_image_size = (512, 512)
model_name = "Xception" # DenseNet201, InceptionResNetV2, InceptionV3, Xception, ResNet50, ResNetRS50 ,ResNet50V2, RegNetY320, NASNetLarge, VGG16, VGG19
base_model = tf.keras.applications.xception.Xception(
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
    weights='imagenet',#전이학습 가중치 imagenet or None
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation='softmax'
    )
#========================================================
#call back 모니터링 세팅
monitor_factor = 'val_accuracy'
monitor_epochs = 3 #call back에서 에포크설정동안 정확도가 향상되지 않으면 훈련 중지
#========================================================
#튜닝세팅
tuning_learning_rate = 0.00000000001
tuning_epoch = 1000
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
save_dir = os.path.join(base_dir,"results")
best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
last_model_dir = os.path.join(save_dir,f"{model_name}_Last.h5")
fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")


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
        filepath=best_model_dir,
        monitor='accuracy',
        mode='auto',
        save_best_only=True #가장 좋았던값으로 가중치를 저장
        ),
    #텐서보드 저장
    keras.callbacks.TensorBoard(
        log_dir=save_dir, 
        histogram_freq=monitor_epochs, 
        write_graph=True, 
        write_images=True,
        write_steps_per_second=True
        )
    ]
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
    target_size=custom_image_size,
    batch_size=custom_batch,
    class_mode='categorical'
    )

#검증 데이터셋
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    )
validation_dir = os.path.join(base_dir,'val')
validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=custom_image_size,
    batch_size=custom_batch,
    class_mode='categorical'
    )

#테스트 데이터 셋
test_dir = os.path.join(base_dir,'test')
test_dataset = validation_datagen.flow_from_directory(
    test_dir,
    batch_size=custom_batch,
    target_size=custom_image_size,
    class_mode='categorical',
    classes = None
    )

#기본모델 멈춤
base_model.trainable = False

#아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(units = class_num, activation='softmax')(out_layer)

#레이어 추가된 모델 설정
model = tf.keras.models.Model(base_model.input, out_layer)

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=custom_learning_rate), #최적화 설정
    metrics=['accuracy']
    )

# 모델 요약 출력
model.summary()

#튜닝학습 정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
#모델이름 설정
def Fine_tuning_Sono_Penet_DenseNet201(your_nicest_parameters='f_hist'):
    model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=tuning_learning_rate), #최적화 설정
    metrics=['accuracy']
    )
    f_hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=tuning_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core
    )
    
    #학습모델 저장
    model.save(fine_tuning_model_dir)
    
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
    plt.title('Fine tuning Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'loss_{model_name}_Fine_tuning.png'))
    
    #학습 정확도 시각화
    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'accuracy_{model_name}_Fine_tuning.png'))
    
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

#학습정의

@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Sono_Penet_DenseNet201(your_nicest_parameters='hist'):
    #학습
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epochs,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core,
    callbacks=[callback_list]
    )
    hist.model.save(last_model_dir)
    
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
    plt.savefig(os.path.join(save_dir,f'loss_{model_name}.png'))
    
    #학습 정확도 시각화
    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'accuracy_{model_name}.png'))
    
    #마지막 가중치 테스트
    l_loss, l_accuracy = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    
    #Callback 가중치 테스트
    b_model = load_model(best_model_dir)
    b_loss, b_accuracy = b_model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    
    # 테스트 결과 출력
    print(f'Lest loss : {l_loss}')
    print(f'Lest accuracy : {l_accuracy}')  
    print(f'Best loss : {b_loss}')
    print(f'Best accuracy : {b_accuracy}')

    #튜닝 실행
    model.trainable = True
    model.summary()
    try:
        Fine_tuning_Sono_Penet_DenseNet201()
    
    #slack에 출력
    finally:
        time.sleep(3)
        return f'\n {model_name} Train accuracy : {max(acc)}\n{model_name} Best accuracy : {b_accuracy}\n{model_name} Best loss : {b_loss}\n{model_name} Last accuracy : {l_accuracy}\n{model_name} Last loss : {l_loss}'


# 실행
Sono_Penet_DenseNet201()

#다음 슬랙을 위한 대기시간 설정
print("Waiting for next trainning")
time.sleep(61)
print("Done!")