import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from knockknock import slack_sender
from keras.models import load_model
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#GPU오류 수정
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print(session)
fix_gpu()
#====================================하단 수정
#슬랙 webhook주소
webhook_slack = "@@"
slack_channel = "#@@"

#데이터 및 컴퓨터 설정
base_dir = '/home/@@'
class_num = 1 # binary is 1
cpu_core = 16
custom_class_mode = 'binary'
#참고 https://keras.io/ko/preprocessing/image/
#"categorical", "binary", "sparse", "input", "other",'None' 종류로 모델 설정
#모델설정
#https://www.tensorflow.org/api_docs/python/tf/keras/applications참고
model_name = "DenseNet201"
custom_learning_rate = 0.001
model_loss_function = 'binary_crossentropy'
#mse, categorical_crossentropy, binary_crossentropy등 사용 가능
custom_metrics = 'binary_accuracy'
# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy등 사용가능
custom_activation_function = 'sigmoid'#binary 는 sigmoid 나머지는 softmax
custom_transfer_weight = 'imagenet'# imagenet이나 None
custom_classifier_activation = 'softmax' # None or softmax
custom_batch = 16
custom_epoch = 1000
custom_learning_rate = 0.001
custom_image_size = (512, 512)

#세부 튜닝설정
tuning_learning_rate = 0.00001
tuning_epoch = 500

#call back 모니터링 설정
monitor_factor = f'val_{custom_metrics}'
monitor_epoch = 100 #call back에포크

#기본 모델 설정
base_model = tf.keras.applications.densenet.DenseNet201(
    include_top=False,
    weights=custom_transfer_weight
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=class_num,
    classifier_activation='softmax'
    )

#데이터 증강 옵션
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
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
    )
# 최적화
# https://keras.io/api/optimizers/ 참고
model_optimizer = tf.keras.optimizers.Adam(
    learning_rate=custom_learning_rate,
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
    # name="Adam",
    )

#====================================================
#검증세트 생성
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    )
#저장위치
save_dir = os.path.join(base_dir,"results")
best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
last_model_dir = os.path.join(save_dir,f"{model_name}_Last.h5")
fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
#콜벡
callback_list=[
    #조기종료 옵션
    keras.callbacks.EarlyStopping(
        monitor=monitor_factor,
        patience=monitor_epoch, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
        mode='auto',
        restore_best_weights=True #가장 좋았던값으로 가중치를 저장
        ),
    #에포크마다 현재 가중치를 저장
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_dir,
        monitor=custom_metrics,
        mode='auto',
        save_best_only=True #가장 좋았던값으로 가중치를 저장
        ),
    #텐서보드 저장
    # keras.callbacks.TensorBoard(
    #     log_dir=save_dir, 
    #     histogram_freq=monitor_epoch, 
    #     write_graph=True, 
    #     write_images=True,
    #     write_steps_per_second=True
    #     )
    ]
#학습 데이터 셋
train_dir = os.path.join(base_dir,'train')
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=custom_image_size,
    batch_size=custom_batch,
    class_mode=custom_class_mode
    )
#검증 데이터 셋
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
#기본모델 멈춤
base_model.trainable = False
#아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
# out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='sigmoid')(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(class_num, activation=custom_activation_function)(out_layer)
model = tf.keras.models.Model(base_model.input, out_layer)

# 모델 컴파일
model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics]
    )
# 모델 요약 출력
model.summary()


#튜닝학습
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Fine_tuning(your_nicest_parameters='f_hist'):
    model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics]
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
    f_acc = f_hist.history[custom_metrics]
    f_val_acc = f_hist.history[f'val_{custom_metrics}']
    f_loss = f_hist.history['loss']
    f_val_loss = f_hist.history['val_loss']
    #학습 정확도 시각화
    plt.figure()
    plt.plot(f_acc, label=f'Training {custom_metrics}')
    plt.plot(f_val_acc, label=f'Validation {custom_metrics}')
    plt.legend(loc='lower right')
    plt.ylabel(custom_metrics)
    plt.ylim([min(plt.ylim()),1])
    plt.title(f'Fine tuning Training and Validation {custom_metrics}')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}_Fine_tuning.png'))
    #학습 손실 시각화
    plt.figure()
    plt.plot(f_loss, label='Training Loss')
    plt.plot(f_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'loss_{model_name}_Fine_tuning.png'))
    #테스트
    f_test_loss, f_test_accuracy = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    time.sleep(3)
    #slack에 출력
    return f'\n {model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n {model_name} Fine_tuning_Train_loss : {min(f_loss)}\n {model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)}\n {model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}\n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss}'

#학습정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Training(your_nicest_parameters='hist'):
    #학습
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core,
    callbacks=[callback_list]
    )
    hist.model.save(last_model_dir)
    #학습 결과 시각화
    acc = hist.history[custom_metrics]
    val_acc = hist.history[f'val_{custom_metrics}']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    #학습 손실 시각화
    plt.figure()
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'loss_{model_name}.png'))
    #학습 정확도 시각화
    plt.figure()
    plt.plot(acc, label=f'Training {custom_metrics}')
    plt.plot(val_acc, label=f'Validation {custom_metrics}')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title(f'Training and Validation {custom_metrics}')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}.png'))
    #마지막 가중치 테스트
    l_test_loss, l_test_accuracy = model.evaluate(
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
    b_test_loss, b_test_accuracy = b_model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    
    #튜닝 실행
    model.trainable = True
    model.summary()
    try:
        Fine_tuning()
    
    #slack에 출력
    finally:
        time.sleep(3)
        return f'\nDatasets \nTraining data : {len(train_dataset)} \nValidation data : {len(validation_dataset)} \nTest data : {len(test_dataset)} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n Best_Test_accuracy : {b_test_accuracy}\n{model_name} Best_Test_loss : {b_test_loss} \n{model_name} Last_Test_accuracy : {l_test_accuracy} \n{model_name} Last_Test_loss : {l_test_loss}'  
# 실행
Training()

@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Test_Predict(your_nicest_parameters='custom_prediction'):
    #학습
    model = load_model(fine_tuning_model_dir)
    custom_prediction = model.predict(
    test_dataset,
    verbose='auto',
    workers=cpu_core
    )
    # print(type(prediction))
    # print(len(prediction))
    # print(prediction)
    # print(np.argmax(prediction))
    return f'Check time : {model_name}_based_on{fine_tuning_model_dir}'

Test_Predict()

#다음 슬랙을 위한 대기시간 설정
print("Waiting for next trainning")
time.sleep(10)
print("Done!")