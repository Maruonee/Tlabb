import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from knockknock import slack_sender
from keras.models import load_model
import time

from utils.score import recall_m, precision_m, f1_score_m
from utils.etc import fix_gpu

fix_gpu()
#====================================================================================
#슬랙
webhook_slack = "https://hooks.slack.com/services/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"
slack_channel = "ct_train"
#데이터 및 컴퓨터 설정
base_dir = '/home/tlab1004/datasets/Class/Contrast'
class_num = 2 #binary is 1
cpu_core = 16
#하이퍼파라미터 설정
#참고 https://keras.io/ko/preprocessing/image/
custom_class_mode = 'categorical'#"categorical", "binary", "sparse", "input", "other",'None'
custom_batch = 16
custom_epoch = 300
custom_learning_rate = 0.001
custom_image_size = (512, 512)
tuning_learning_rate = 0.00001
tuning_epoch = 100
# monitor_epoch = 100 #call back에포크

# #====================================================================================
# #1 모델설정
# #https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
# model_name = "VGG19"
# custom_learning_rate = 0.001
# custom_learning_rate = 0.001
# model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
# custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
# base_model = tf.keras.applications.VGG19(
#     include_top=False,
#     weights='imagenet',# 전이학습 가중치 imagenet or None
#     input_shape=None,
#     pooling=None,
#     classes=class_num,
#     classifier_activation='softmax'# None or "softmax"
#     )
# #데이터 생성
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     )
# #   추가 이미지 증강 옵션
# #   rotation_range=40,
# #   width_shift_range=0.2,
# #   height_shift_range=0.2,
# #   shear_range=0.2,
# #   zoom_range=0.2,
# #   horizontal_flip=True,
# #   fill_mode=`nearest`
# #   featurewise_center=False,
# #   samplewise_center=False,
# #   featurewise_std_normalization=False,
# #   samplewise_std_normalization=False,
# #   zca_whitening=False,
# #   zca_epsilon=1e-06,
# #   rotation_range=0,
# #   width_shift_range=0.0,
# #   height_shift_range=0.0,
# #   brightness_range=None,
# #   shear_range=0.0,
# #   zoom_range=0.0,
# #   channel_shift_range=0.0,
# #   fill_mode='nearest',
# #   cval=0.0,
# #   horizontal_flip=False,
# #   vertical_flip=False,
# #   rescale=None,
# #   preprocessing_function=None,
# #   data_format=None,
# #   validation_split=0.0,
# #   dtype=None
# # 최적화 https://keras.io/api/optimizers/ 참고
# model_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=custom_learning_rate
#     )
#     # beta_1=0.9,
#     # beta_2=0.999,
#     # epsilon=1e-07,
#     # amsgraid=False,
#     # weight_decay=None,
#     # clipnorm=None,
#     # clipvalue=None,
#     # global_clipnorm=None,
#     # use_ema=False,
#     # ema_momentum=0.99,
#     # ema_overwrte_frequency=None,
#     # jit_compile=True,
#     # name="Adam"
# #====================================================================================
# validation_datagen = ImageDataGenerator(
#     rescale=1./255,
#     )
# #저장위치
# save_dir = os.path.join(base_dir,"results")
# best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
# last_model_dir = os.path.join(save_dir,f"{model_name}.h5")
# fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
# # # 조기종료 옵션
# # callback_list=[
# #     keras.callbacks.EarlyStopping(
# #         monitor=f'val_{custom_metrics}',
# #         patience=monitor_epoch, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
# #         mode='auto',
# #         restore_best_weights=True #가장 좋았던값으로 가중치를 저장
# #     ),
# #     #에포크마다 현재 가중치를 저장
# #     keras.callbacks.ModelCheckpoint(
# #         filepath=best_model_dir,
# #         monitor=custom_metrics,
# #         mode='auto',
# #         save_best_only=True #가장 좋았던값으로 가중치를 저장
# #     ),
# # ]
# #데이터 셋
# train_dir = os.path.join(base_dir,'train')
# train_dataset = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# validation_dir = os.path.join(base_dir,'val')
# validation_dataset = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# #테스트 데이터 셋
# test_dir = os.path.join(base_dir,'test')
# test_dataset = validation_datagen.flow_from_directory(
#     test_dir,
#     batch_size=custom_batch,
#     target_size=custom_image_size,
#     class_mode=custom_class_mode,
#     )
# #기본모델 멈춤
# base_model.trainable = False
# #아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
# out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
# out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
# out_layer = tf.keras.layers.ReLU()(out_layer) 
# out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
# out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
# model = tf.keras.models.Model(base_model.input, out_layer)
# # 모델 컴파일
# model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
# # 모델 요약 출력
# model.summary()
# #검출시간 계산
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Test_Predict(your_nicest_parameters='custom_prediction'):
#     #학습
#     custom_prediction = model.predict(
#     test_dataset,
#     verbose='auto',
#     workers=cpu_core
#     )
#     return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

# #튜닝학습
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Fine_tuning(your_nicest_parameters='f_hist'):
#     model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
#     f_hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=tuning_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core
#     )
#     #학습모델 저장
#     f_hist.model.save(fine_tuning_model_dir)
#     #학습 결과 시각화
#     f_acc = f_hist.history[custom_metrics]
#     f_val_acc = f_hist.history[f'val_{custom_metrics}']
#     f_loss = f_hist.history['loss']
#     f_val_loss = f_hist.history['val_loss']
#     f_recall = f_hist.history['recall_m']
#     f_val_recall = f_hist.history['val_recall_m']
#     f_precision = f_hist.history['precision_m']
#     f_val_precision = f_hist.history['val_precision_m']
#     f_f1_score = f_hist.history['f1_score_m']
#     f_val_f1_score = f_hist.history['val_f1_score_m']
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(f_loss, label='Training Loss')
#     plt.plot(f_val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}_Fine_tuning.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(f_acc, label=f'Training {custom_metrics}')
#     plt.plot(f_val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel(custom_metrics)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Fine tuning Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}_Fine_tuning.png'))
#     # 학습 recall 시각화
#     plt.figure()
#     plt.plot(f_recall, label=f'Training Recall')
#     plt.plot(f_val_recall, label=f'Validation Recall')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(f_precision, label=f'Training Precision')
#     plt.plot(f_val_precision, label=f'Validation Precision')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f_f1_score, label='Training F1 score')
#     plt.plot(f_val_f1_score, label='Validation F1 score')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning F1 score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
#     # #테스트
#     f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     #slack에 출력
#     return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

# #학습정의
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Training(your_nicest_parameters='hist'):
#     hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=custom_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core,
#     # callbacks=[callback_list]
#     )
#     hist.model.save(last_model_dir)
    
#     #학습 결과 시각화
#     acc = hist.history[custom_metrics]
#     val_acc = hist.history[f'val_{custom_metrics}']
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#     recall = hist.history['recall_m']
#     val_recall = hist.history['val_recall_m']
#     precision = hist.history['precision_m']
#     val_precision = hist.history['val_precision_m']
#     f1_score = hist.history['f1_score_m']
#     val_f1_score = hist.history['val_f1_score_m']
    
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(loss, label='Training Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(acc, label=f'Training {custom_metrics}')
#     plt.plot(val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel('Accuracy')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}.png'))
#     #학습 recall 시각화
#     plt.figure()
#     plt.plot(recall, label='Training Recall')
#     plt.plot(val_recall, label='Validation Recall')
#     plt.legend(loc='lower right')
#     plt.ylabel('Recall')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(precision, label='Training Precision')
#     plt.plot(val_precision, label='Validation Precision')
#     plt.legend(loc='lower right')
#     plt.ylabel('Precision')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f1_score, label='Training F1_score')
#     plt.plot(val_f1_score, label='Validation F1_score')
#     plt.legend(loc='lower right')
#     plt.ylabel('F1_score')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation F1_score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
#     #마지막 가중치 테스트
#     l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     print('!!!!Training done!!!!')
    
#     print('Test start')
#     Test_Predict()
#     print('!!!!Test done!!!!')
    
#     print('Fine tuning start')
#     Fine_tuning()    
#     print('!!!!FineTraining done!!!!')
    
#     return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# # 실행
# Training()
# time.sleep(10)


# #====================================================================================
# #1 모델설정
# #https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
# model_name = "DenseNet201"
# custom_learning_rate = 0.001
# custom_learning_rate = 0.001
# model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
# custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
# base_model = tf.keras.applications.densenet.DenseNet201(
#     include_top=False,
#     weights='imagenet',# 전이학습 가중치 imagenet or None    input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=class_num,
#     classifier_activation='softmax'# None or "softmax"
#     )
# #데이터 생성
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     )
# #   추가 이미지 증강 옵션
# #   rotation_range=40,
# #   width_shift_range=0.2,
# #   height_shift_range=0.2,
# #   shear_range=0.2,
# #   zoom_range=0.2,
# #   horizontal_flip=True,
# #   fill_mode=`nearest`
# #   featurewise_center=False,
# #   samplewise_center=False,
# #   featurewise_std_normalization=False,
# #   samplewise_std_normalization=False,
# #   zca_whitening=False,
# #   zca_epsilon=1e-06,
# #   rotation_range=0,
# #   width_shift_range=0.0,
# #   height_shift_range=0.0,
# #   brightness_range=None,
# #   shear_range=0.0,
# #   zoom_range=0.0,
# #   channel_shift_range=0.0,
# #   fill_mode='nearest',
# #   cval=0.0,
# #   horizontal_flip=False,
# #   vertical_flip=False,
# #   rescale=None,
# #   preprocessing_function=None,
# #   data_format=None,
# #   validation_split=0.0,
# #   dtype=None
# # 최적화 https://keras.io/api/optimizers/ 참고
# model_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=custom_learning_rate
#     )
#     # beta_1=0.9,
#     # beta_2=0.999,
#     # epsilon=1e-07,
#     # amsgraid=False,
#     # weight_decay=None,
#     # clipnorm=None,
#     # clipvalue=None,
#     # global_clipnorm=None,
#     # use_ema=False,
#     # ema_momentum=0.99,
#     # ema_overwrte_frequency=None,
#     # jit_compile=True,
#     # name="Adam"
# #====================================================================================
# validation_datagen = ImageDataGenerator(
#     rescale=1./255,
#     )
# #저장위치
# save_dir = os.path.join(base_dir,"results")
# best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
# last_model_dir = os.path.join(save_dir,f"{model_name}.h5")
# fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
# # # 조기종료 옵션
# # callback_list=[
# #     keras.callbacks.EarlyStopping(
# #         monitor=f'val_{custom_metrics}',
# #         patience=monitor_epoch, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
# #         mode='auto',
# #         restore_best_weights=True #가장 좋았던값으로 가중치를 저장
# #     ),
# #     #에포크마다 현재 가중치를 저장
# #     keras.callbacks.ModelCheckpoint(
# #         filepath=best_model_dir,
# #         monitor=custom_metrics,
# #         mode='auto',
# #         save_best_only=True #가장 좋았던값으로 가중치를 저장
# #     ),
# # ]
# #데이터 셋
# train_dir = os.path.join(base_dir,'train')
# train_dataset = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# validation_dir = os.path.join(base_dir,'val')
# validation_dataset = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# #테스트 데이터 셋
# test_dir = os.path.join(base_dir,'test')
# test_dataset = validation_datagen.flow_from_directory(
#     test_dir,
#     batch_size=custom_batch,
#     target_size=custom_image_size,
#     class_mode=custom_class_mode,
#     )
# #기본모델 멈춤
# base_model.trainable = False
# #아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
# out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
# out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
# out_layer = tf.keras.layers.ReLU()(out_layer) 
# out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
# out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
# model = tf.keras.models.Model(base_model.input, out_layer)
# # 모델 컴파일
# model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
# # 모델 요약 출력
# model.summary()
# #검출시간 계산
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Test_Predict(your_nicest_parameters='custom_prediction'):
#     #학습
#     custom_prediction = model.predict(
#     test_dataset,
#     verbose='auto',
#     workers=cpu_core
#     )
#     return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

# #튜닝학습
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Fine_tuning(your_nicest_parameters='f_hist'):
#     model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
#     f_hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=tuning_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core
#     )
#     #학습모델 저장
#     f_hist.model.save(fine_tuning_model_dir)
#     #학습 결과 시각화
#     f_acc = f_hist.history[custom_metrics]
#     f_val_acc = f_hist.history[f'val_{custom_metrics}']
#     f_loss = f_hist.history['loss']
#     f_val_loss = f_hist.history['val_loss']
#     f_recall = f_hist.history['recall_m']
#     f_val_recall = f_hist.history['val_recall_m']
#     f_precision = f_hist.history['precision_m']
#     f_val_precision = f_hist.history['val_precision_m']
#     f_f1_score = f_hist.history['f1_score_m']
#     f_val_f1_score = f_hist.history['val_f1_score_m']
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(f_loss, label='Training Loss')
#     plt.plot(f_val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}_Fine_tuning.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(f_acc, label=f'Training {custom_metrics}')
#     plt.plot(f_val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel(custom_metrics)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Fine tuning Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}_Fine_tuning.png'))
#     # 학습 recall 시각화
#     plt.figure()
#     plt.plot(f_recall, label=f'Training Recall')
#     plt.plot(f_val_recall, label=f'Validation Recall')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(f_precision, label=f'Training Precision')
#     plt.plot(f_val_precision, label=f'Validation Precision')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f_f1_score, label='Training F1 score')
#     plt.plot(f_val_f1_score, label='Validation F1 score')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning F1 score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
#     # #테스트
#     f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     #slack에 출력
#     return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

# #학습정의
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Training(your_nicest_parameters='hist'):
#     hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=custom_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core,
#     # callbacks=[callback_list]
#     )
#     hist.model.save(last_model_dir)
    
#     #학습 결과 시각화
#     acc = hist.history[custom_metrics]
#     val_acc = hist.history[f'val_{custom_metrics}']
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#     recall = hist.history['recall_m']
#     val_recall = hist.history['val_recall_m']
#     precision = hist.history['precision_m']
#     val_precision = hist.history['val_precision_m']
#     f1_score = hist.history['f1_score_m']
#     val_f1_score = hist.history['val_f1_score_m']
    
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(loss, label='Training Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(acc, label=f'Training {custom_metrics}')
#     plt.plot(val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel('Accuracy')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}.png'))
#     #학습 recall 시각화
#     plt.figure()
#     plt.plot(recall, label='Training Recall')
#     plt.plot(val_recall, label='Validation Recall')
#     plt.legend(loc='lower right')
#     plt.ylabel('Recall')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(precision, label='Training Precision')
#     plt.plot(val_precision, label='Validation Precision')
#     plt.legend(loc='lower right')
#     plt.ylabel('Precision')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f1_score, label='Training F1_score')
#     plt.plot(val_f1_score, label='Validation F1_score')
#     plt.legend(loc='lower right')
#     plt.ylabel('F1_score')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation F1_score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
#     #마지막 가중치 테스트
#     l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     print('!!!!Training done!!!!')
    
#     print('Test start')
#     Test_Predict()
#     print('!!!!Test done!!!!')
    
#     print('Fine tuning start')
#     Fine_tuning()    
#     print('!!!!FineTraining done!!!!')
    
#     return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# # 실행
# Training()
# time.sleep(10)

# #====================================================================================
# #1 모델설정
# #https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
# model_name = "EfficientNetB2"
# custom_learning_rate = 0.001
# model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
# custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
# base_model = tf.keras.applications.efficientnet.EfficientNetB2(
#     include_top=False,
#     weights='imagenet',# 전이학습 가중치 imagenet or None    input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=class_num,
#     classifier_activation='softmax'# None or "softmax"
#     )
# #데이터 생성
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     )
# #   추가 이미지 증강 옵션
# #   rotation_range=40,
# #   width_shift_range=0.2,
# #   height_shift_range=0.2,
# #   shear_range=0.2,
# #   zoom_range=0.2,
# #   horizontal_flip=True,
# #   fill_mode=`nearest`
# #   featurewise_center=False,
# #   samplewise_center=False,
# #   featurewise_std_normalization=False,
# #   samplewise_std_normalization=False,
# #   zca_whitening=False,
# #   zca_epsilon=1e-06,
# #   rotation_range=0,
# #   width_shift_range=0.0,
# #   height_shift_range=0.0,
# #   brightness_range=None,
# #   shear_range=0.0,
# #   zoom_range=0.0,
# #   channel_shift_range=0.0,
# #   fill_mode='nearest',
# #   cval=0.0,
# #   horizontal_flip=False,
# #   vertical_flip=False,
# #   rescale=None,
# #   preprocessing_function=None,
# #   data_format=None,
# #   validation_split=0.0,
# #   dtype=None
# # 최적화 https://keras.io/api/optimizers/ 참고
# model_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=custom_learning_rate
#     )
#     # beta_1=0.9,
#     # beta_2=0.999,
#     # epsilon=1e-07,
#     # amsgraid=False,
#     # weight_decay=None,
#     # clipnorm=None,
#     # clipvalue=None,
#     # global_clipnorm=None,
#     # use_ema=False,
#     # ema_momentum=0.99,
#     # ema_overwrte_frequency=None,
#     # jit_compile=True,
#     # name="Adam"
# #====================================================================================
# validation_datagen = ImageDataGenerator(
#     rescale=1./255,
#     )
# #저장위치
# save_dir = os.path.join(base_dir,"results")
# best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
# last_model_dir = os.path.join(save_dir,f"{model_name}.h5")
# fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
# # 조기종료 옵션
# # callback_list=[
# #     keras.callbacks.EarlyStopping(
# #         monitor=f'val_{custom_metrics}',
# #         patience=monitor_epoch, #에포크설정동안 정확도가 향상되지 않으면 훈련 중지
# #         mode='auto',
# #         restore_best_weights=True #가장 좋았던값으로 가중치를 저장
# #     ),
# #     #에포크마다 현재 가중치를 저장
# #     keras.callbacks.ModelCheckpoint(
# #         filepath=best_model_dir,
# #         monitor=custom_metrics,
# #         mode='auto',
# #         save_best_only=True #가장 좋았던값으로 가중치를 저장
# #     ),
# # ]
# #데이터 셋
# train_dir = os.path.join(base_dir,'train')
# train_dataset = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# validation_dir = os.path.join(base_dir,'val')
# validation_dataset = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=custom_image_size,
#     batch_size=custom_batch,
#     class_mode=custom_class_mode
#     )
# #테스트 데이터 셋
# test_dir = os.path.join(base_dir,'test')
# test_dataset = validation_datagen.flow_from_directory(
#     test_dir,
#     batch_size=custom_batch,
#     target_size=custom_image_size,
#     class_mode=custom_class_mode,
#     )
# #기본모델 멈춤
# base_model.trainable = False
# #아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
# out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
# out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
# out_layer = tf.keras.layers.ReLU()(out_layer) 
# out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
# out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
# model = tf.keras.models.Model(base_model.input, out_layer)
# # 모델 컴파일
# model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
# # 모델 요약 출력
# model.summary()
# #검출시간 계산
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Test_Predict(your_nicest_parameters='custom_prediction'):
#     #학습
#     custom_prediction = model.predict(
#     test_dataset,
#     verbose='auto',
#     workers=cpu_core
#     )
#     return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

# #튜닝학습
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Fine_tuning(your_nicest_parameters='f_hist'):
#     model.compile(
#     loss=model_loss_function,
#     optimizer=model_optimizer,
#     metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
#     )
#     f_hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=tuning_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core
#     )
#     #학습모델 저장
#     f_hist.model.save(fine_tuning_model_dir)
#     #학습 결과 시각화
#     f_acc = f_hist.history[custom_metrics]
#     f_val_acc = f_hist.history[f'val_{custom_metrics}']
#     f_loss = f_hist.history['loss']
#     f_val_loss = f_hist.history['val_loss']
#     f_recall = f_hist.history['recall_m']
#     f_val_recall = f_hist.history['val_recall_m']
#     f_precision = f_hist.history['precision_m']
#     f_val_precision = f_hist.history['val_precision_m']
#     f_f1_score = f_hist.history['f1_score_m']
#     f_val_f1_score = f_hist.history['val_f1_score_m']
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(f_loss, label='Training Loss')
#     plt.plot(f_val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}_Fine_tuning.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(f_acc, label=f'Training {custom_metrics}')
#     plt.plot(f_val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel(custom_metrics)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Fine tuning Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}_Fine_tuning.png'))
#     # 학습 recall 시각화
#     plt.figure()
#     plt.plot(f_recall, label=f'Training Recall')
#     plt.plot(f_val_recall, label=f'Validation Recall')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(f_precision, label=f'Training Precision')
#     plt.plot(f_val_precision, label=f'Validation Precision')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f_f1_score, label='Training F1 score')
#     plt.plot(f_val_f1_score, label='Validation F1 score')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Fine tuning F1 score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
#     # #테스트
#     f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     #slack에 출력
#     return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

# #학습정의
# @slack_sender(webhook_url=webhook_slack, channel=slack_channel)
# def Training(your_nicest_parameters='hist'):
#     hist = model.fit(
#     train_dataset,
#     batch_size=custom_batch,
#     epochs=custom_epoch,
#     validation_data=validation_dataset,
#     verbose=1,
#     workers=cpu_core,
#     # callbacks=[callback_list]
#     )
#     hist.model.save(last_model_dir)
    
#     #학습 결과 시각화
#     acc = hist.history[custom_metrics]
#     val_acc = hist.history[f'val_{custom_metrics}']
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#     recall = hist.history['recall_m']
#     val_recall = hist.history['val_recall_m']
#     precision = hist.history['precision_m']
#     val_precision = hist.history['val_precision_m']
#     f1_score = hist.history['f1_score_m']
#     val_f1_score = hist.history['val_f1_score_m']
    
#     #학습 손실 시각화
#     plt.figure()
#     plt.plot(loss, label='Training Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel(model_loss_function)
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'loss_{model_name}.png'))
#     #학습 정확도 시각화
#     plt.figure()
#     plt.plot(acc, label=f'Training {custom_metrics}')
#     plt.plot(val_acc, label=f'Validation {custom_metrics}')
#     plt.legend(loc='lower right')
#     plt.ylabel('Accuracy')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title(f'Training and Validation {custom_metrics}')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'{custom_metrics}_{model_name}.png'))
#     #학습 recall 시각화
#     plt.figure()
#     plt.plot(recall, label='Training Recall')
#     plt.plot(val_recall, label='Validation Recall')
#     plt.legend(loc='lower right')
#     plt.ylabel('Recall')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Recall')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
#     #학습 precision 시각화
#     plt.figure()
#     plt.plot(precision, label='Training Precision')
#     plt.plot(val_precision, label='Validation Precision')
#     plt.legend(loc='lower right')
#     plt.ylabel('Precision')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation Precision')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
#     #학습 f1score 시각화
#     plt.figure()
#     plt.plot(f1_score, label='Training F1_score')
#     plt.plot(val_f1_score, label='Validation F1_score')
#     plt.legend(loc='lower right')
#     plt.ylabel('F1_score')
#     plt.ylim([min(plt.ylim()),1])
#     plt.title('Training and Validation F1_score')
#     plt.xlabel('epoch')
#     plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
#     #마지막 가중치 테스트
#     l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
#         test_dataset,
#         batch_size=custom_batch,
#         verbose=1,
#         steps=None,
#         workers=cpu_core,
#         use_multiprocessing=False,
#         return_dict=False,
#         )
#     print('!!!!Training done!!!!')
    
#     print('Test start')
#     Test_Predict()
#     print('!!!!Test done!!!!')
    
#     print('Fine tuning start')
#     Fine_tuning()    
#     print('!!!!FineTraining done!!!!')
    
#     return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# # 실행
# Training()
# time.sleep(10)

#====================================================================================
#1 모델설정
#https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
model_name = "inception_resnet_v2"
custom_learning_rate = 0.001
custom_learning_rate = 0.001
model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights='imagenet',# 전이학습 가중치 imagenet or None    input_tensor=None,
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
# 조기종료 옵션
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
#기본모델 멈춤
base_model.trainable = False
#아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
model = tf.keras.models.Model(base_model.input, out_layer)
# 모델 컴파일
model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
    )
# 모델 요약 출력
model.summary()
#검출시간 계산
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Test_Predict(your_nicest_parameters='custom_prediction'):
    #학습
    custom_prediction = model.predict(
    test_dataset,
    verbose='auto',
    workers=cpu_core
    )
    return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

#튜닝학습
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Fine_tuning(your_nicest_parameters='f_hist'):
    model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
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
    f_hist.model.save(fine_tuning_model_dir)
    #학습 결과 시각화
    f_acc = f_hist.history[custom_metrics]
    f_val_acc = f_hist.history[f'val_{custom_metrics}']
    f_loss = f_hist.history['loss']
    f_val_loss = f_hist.history['val_loss']
    f_recall = f_hist.history['recall_m']
    f_val_recall = f_hist.history['val_recall_m']
    f_precision = f_hist.history['precision_m']
    f_val_precision = f_hist.history['val_precision_m']
    f_f1_score = f_hist.history['f1_score_m']
    f_val_f1_score = f_hist.history['val_f1_score_m']
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
    # 학습 recall 시각화
    plt.figure()
    plt.plot(f_recall, label=f'Training Recall')
    plt.plot(f_val_recall, label=f'Validation Recall')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(f_precision, label=f'Training Precision')
    plt.plot(f_val_precision, label=f'Validation Precision')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f_f1_score, label='Training F1 score')
    plt.plot(f_val_f1_score, label='Validation F1 score')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning F1 score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
    # #테스트
    f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    #slack에 출력
    return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

#학습정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Training(your_nicest_parameters='hist'):
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core,
#    callbacks=[callback_list]
    )
    hist.model.save(last_model_dir)
    
    #학습 결과 시각화
    acc = hist.history[custom_metrics]
    val_acc = hist.history[f'val_{custom_metrics}']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    recall = hist.history['recall_m']
    val_recall = hist.history['val_recall_m']
    precision = hist.history['precision_m']
    val_precision = hist.history['val_precision_m']
    f1_score = hist.history['f1_score_m']
    val_f1_score = hist.history['val_f1_score_m']
    
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
    #학습 recall 시각화
    plt.figure()
    plt.plot(recall, label='Training Recall')
    plt.plot(val_recall, label='Validation Recall')
    plt.legend(loc='lower right')
    plt.ylabel('Recall')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(precision, label='Training Precision')
    plt.plot(val_precision, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.ylabel('Precision')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f1_score, label='Training F1_score')
    plt.plot(val_f1_score, label='Validation F1_score')
    plt.legend(loc='lower right')
    plt.ylabel('F1_score')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation F1_score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
    #마지막 가중치 테스트
    l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    print('!!!!Training done!!!!')
    
    print('Test start')
    Test_Predict()
    print('!!!!Test done!!!!')
    
    print('Fine tuning start')
    Fine_tuning()    
    print('!!!!FineTraining done!!!!')
    
    return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# 실행
Training()
time.sleep(10)

#====================================================================================
#1 모델설정
#https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
model_name = "ResNet50V2"
custom_learning_rate = 0.001
custom_learning_rate = 0.001
model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
base_model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=False,
    weights='imagenet',# 전이학습 가중치 imagenet or None    input_tensor=None,
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
# 조기종료 옵션
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
#기본모델 멈춤
base_model.trainable = False
#아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
model = tf.keras.models.Model(base_model.input, out_layer)
# 모델 컴파일
model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
    )
# 모델 요약 출력
model.summary()
#검출시간 계산
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Test_Predict(your_nicest_parameters='custom_prediction'):
    #학습
    custom_prediction = model.predict(
    test_dataset,
    verbose='auto',
    workers=cpu_core
    )
    return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

#튜닝학습
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Fine_tuning(your_nicest_parameters='f_hist'):
    model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
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
    f_hist.model.save(fine_tuning_model_dir)
    #학습 결과 시각화
    f_acc = f_hist.history[custom_metrics]
    f_val_acc = f_hist.history[f'val_{custom_metrics}']
    f_loss = f_hist.history['loss']
    f_val_loss = f_hist.history['val_loss']
    f_recall = f_hist.history['recall_m']
    f_val_recall = f_hist.history['val_recall_m']
    f_precision = f_hist.history['precision_m']
    f_val_precision = f_hist.history['val_precision_m']
    f_f1_score = f_hist.history['f1_score_m']
    f_val_f1_score = f_hist.history['val_f1_score_m']
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
    # 학습 recall 시각화
    plt.figure()
    plt.plot(f_recall, label=f'Training Recall')
    plt.plot(f_val_recall, label=f'Validation Recall')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(f_precision, label=f'Training Precision')
    plt.plot(f_val_precision, label=f'Validation Precision')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f_f1_score, label='Training F1 score')
    plt.plot(f_val_f1_score, label='Validation F1 score')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning F1 score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
    # #테스트
    f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    #slack에 출력
    return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

#학습정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Training(your_nicest_parameters='hist'):
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core,
#    callbacks=[callback_list]
    )
    hist.model.save(last_model_dir)
    
    #학습 결과 시각화
    acc = hist.history[custom_metrics]
    val_acc = hist.history[f'val_{custom_metrics}']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    recall = hist.history['recall_m']
    val_recall = hist.history['val_recall_m']
    precision = hist.history['precision_m']
    val_precision = hist.history['val_precision_m']
    f1_score = hist.history['f1_score_m']
    val_f1_score = hist.history['val_f1_score_m']
    
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
    #학습 recall 시각화
    plt.figure()
    plt.plot(recall, label='Training Recall')
    plt.plot(val_recall, label='Validation Recall')
    plt.legend(loc='lower right')
    plt.ylabel('Recall')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(precision, label='Training Precision')
    plt.plot(val_precision, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.ylabel('Precision')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f1_score, label='Training F1_score')
    plt.plot(val_f1_score, label='Validation F1_score')
    plt.legend(loc='lower right')
    plt.ylabel('F1_score')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation F1_score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
    #마지막 가중치 테스트
    l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    print('!!!!Training done!!!!')
    
    print('Test start')
    Test_Predict()
    print('!!!!Test done!!!!')
    
    print('Fine tuning start')
    Fine_tuning()    
    print('!!!!FineTraining done!!!!')
    
    return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# 실행
Training()
time.sleep(10)

#====================================================================================
#1 모델설정
#https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
model_name = "Xception"
custom_learning_rate = 0.001
custom_learning_rate = 0.001
model_loss_function = 'categorical_crossentropy'# mse, categorical_crossentropy, binary_crossentropy
custom_metrics = 'categorical_accuracy'# binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
base_model = tf.keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',# 전이학습 가중치 imagenet or None    input_tensor=None,
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
# 조기종료 옵션
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
#기본모델 멈춤
base_model.trainable = False
#아웃레이어 세팅(기본 모델에 계속 적층하는 구조)
out_layer = tf.keras.layers.Conv2D(custom_image_size[1], (1, 1), padding='SAME', activation='softmax')(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(base_model.output)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)#binary is sigmoid and other is softmax and #1 or class_num
model = tf.keras.models.Model(base_model.input, out_layer)
# 모델 컴파일
model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
    )
# 모델 요약 출력
model.summary()
#검출시간 계산
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Test_Predict(your_nicest_parameters='custom_prediction'):
    #학습
    custom_prediction = model.predict(
    test_dataset,
    verbose='auto',
    workers=cpu_core
    )
    return f'Training duration is Predict time {model_name}\n Test data :{test_dataset.samples}\n Training summary is under this message\n.\n.\n.\n,'

#튜닝학습
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Fine_tuning(your_nicest_parameters='f_hist'):
    model.compile(
    loss=model_loss_function,
    optimizer=model_optimizer,
    metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
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
    f_hist.model.save(fine_tuning_model_dir)
    #학습 결과 시각화
    f_acc = f_hist.history[custom_metrics]
    f_val_acc = f_hist.history[f'val_{custom_metrics}']
    f_loss = f_hist.history['loss']
    f_val_loss = f_hist.history['val_loss']
    f_recall = f_hist.history['recall_m']
    f_val_recall = f_hist.history['val_recall_m']
    f_precision = f_hist.history['precision_m']
    f_val_precision = f_hist.history['val_precision_m']
    f_f1_score = f_hist.history['f1_score_m']
    f_val_f1_score = f_hist.history['val_f1_score_m']
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
    # 학습 recall 시각화
    plt.figure()
    plt.plot(f_recall, label=f'Training Recall')
    plt.plot(f_val_recall, label=f'Validation Recall')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}_Fine_tuning.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(f_precision, label=f'Training Precision')
    plt.plot(f_val_precision, label=f'Validation Precision')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}_Fine_tuning.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f_f1_score, label='Training F1 score')
    plt.plot(f_val_f1_score, label='Validation F1 score')
    plt.legend(loc='upper right')
    plt.ylabel(model_loss_function)
    plt.ylim([min(plt.ylim()),1])
    plt.title('Fine tuning F1 score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'f1score_{model_name}_Fine_tuning.png'))
    # #테스트
    f_test_loss, f_test_accuracy, f_test_f1_score, f_test_precision, f_test_recall = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    #slack에 출력
    return f'\n{model_name} Fine_tuning_Train_{custom_metrics} : {max(f_acc)}\n{model_name} Fine_tuning_Train_loss : {min(f_loss)}  \n{model_name} Fine_tuning_Train_Recall : {max(f_recall)} \n{model_name} Fine_tuning_Train_Precision : {max(f_precision)} \n{model_name} Fine_tuning_Train_F1_Score : {max(f_f1_score)} \n{model_name} Fine_tuning_Validation_{custom_metrics} : {max(f_val_acc)} \n{model_name} Fine_tuning_Validation_loss : {min(f_val_loss)}  \n{model_name} Fine_tuning_Validation_Recall : {max(f_val_recall)} \n{model_name} Fine_tuning_Validation_Precision : {max(f_val_precision)} \n{model_name} Fine_tuning_Validation_F1_Score : {max(f_val_f1_score)} \n{model_name} Fine_tuning_Test_{custom_metrics} : {f_test_accuracy} \n{model_name} Fine_tuning_Test_loss : {f_test_loss} \n{model_name} Fine_tuning_Test_Recall : {f_test_recall} \n{model_name} Fine_tuning_Test_Precision : {f_test_precision} \n{model_name} Fine_tuning_Test_F1_Score : {f_test_f1_score}'

#학습정의
@slack_sender(webhook_url=webhook_slack, channel=slack_channel)
def Training(your_nicest_parameters='hist'):
    hist = model.fit(
    train_dataset,
    batch_size=custom_batch,
    epochs=custom_epoch,
    validation_data=validation_dataset,
    verbose=1,
    workers=cpu_core,
#    callbacks=[callback_list]
    )
    hist.model.save(last_model_dir)
    
    #학습 결과 시각화
    acc = hist.history[custom_metrics]
    val_acc = hist.history[f'val_{custom_metrics}']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    recall = hist.history['recall_m']
    val_recall = hist.history['val_recall_m']
    precision = hist.history['precision_m']
    val_precision = hist.history['val_precision_m']
    f1_score = hist.history['f1_score_m']
    val_f1_score = hist.history['val_f1_score_m']
    
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
    #학습 recall 시각화
    plt.figure()
    plt.plot(recall, label='Training Recall')
    plt.plot(val_recall, label='Validation Recall')
    plt.legend(loc='lower right')
    plt.ylabel('Recall')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Recall')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'recall_{model_name}.png'))
    #학습 precision 시각화
    plt.figure()
    plt.plot(precision, label='Training Precision')
    plt.plot(val_precision, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.ylabel('Precision')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Precision')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'precision_{model_name}.png'))
    #학습 f1score 시각화
    plt.figure()
    plt.plot(f1_score, label='Training F1_score')
    plt.plot(val_f1_score, label='Validation F1_score')
    plt.legend(loc='lower right')
    plt.ylabel('F1_score')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation F1_score')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(save_dir,f'F1_score_{model_name}.png'))
    
    #마지막 가중치 테스트
    l_test_loss, l_test_accuracy, l_test_f1_score, l_test_precision, l_test_recall  = model.evaluate(
        test_dataset,
        batch_size=custom_batch,
        verbose=1,
        steps=None,
        workers=cpu_core,
        use_multiprocessing=False,
        return_dict=False,
        )
    print('!!!!Training done!!!!')
    
    print('Test start')
    Test_Predict()
    print('!!!!Test done!!!!')
    
    print('Fine tuning start')
    Fine_tuning()    
    print('!!!!FineTraining done!!!!')
    
    return f'\nDatasets \nTraining data : {train_dataset.samples} \nValidation data : {validation_dataset.samples} \nTest data :{test_dataset.samples} \n{model_name}_Train_{custom_metrics} : {max(acc)} \n{model_name}_Train_loss : {min(loss)} \n{model_name}_Train_Recall : {max(recall)} \n{model_name}_Train_Precision : {max(precision)} \n{model_name}_Train_F1_score : {max(f1_score)} \n{model_name}_Validation_{custom_metrics} : {max(val_acc)} \n{model_name}_Validation_loss : {min(val_loss)} \n{model_name}_Validation_Recall : {max(val_recall)} \n{model_name}_Validation_Precision : {max(val_precision)} \n{model_name}_Validation_F1_score : {max(val_f1_score)} \n{model_name} Test_accuracy :{l_test_accuracy} \n{model_name} Test_loss : {l_test_loss} \n{model_name} Test_Recall : {l_test_recall} \n{model_name} Test_Precision : {l_test_precision} \n{model_name} Test_F1Score : {l_test_f1_score}'  

# 실행
Training()
time.sleep(10)
