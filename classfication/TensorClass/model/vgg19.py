import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import os
from keras import backend as K
from keras.layers import Dense

#Recall Precision F1score
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_score_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1score = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1score

#====================================================================================
#https://www.tensorflow.org/api_docs/python/tf/keras/applications 참고
def class_vgg19(
    class_num, 
    custom_batch = 16, 
    monitor_epoch = 100, 
    base_dir,
    trans_learing = None, 
    augmantaion_option, 
    optimizers_option, 
    custom_class_mode, 
    custom_image_size=(256,256), 
    custom_kernel_size=3
    ): 
    #출력 함수 정의
    activation_function = ""
    model_loss_function = ""
    custom_metrics = ""
    if class_num == 2:
        activation_function = "sigmoid"
        model_loss_function = "binary_crossentropy"
        custom_metrics = "binary_accuracy"
    else:
        activation_function = "softmax"
        model_loss_function = "categorical_crossentropy"
        custom_metrics = "categorical_accuracy"
        
    model_name = "VGG19"
    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights=trans_learing,
        input_tensor=None,
        pooling=None,
        classes=class_num,
        #input_shape=None,
        #classifier_activation='softmax'
        )
    #데이터 생성
    train_datagen = ImageDataGenerator(augmantaion_option)
    model_optimizer = tf.keras.optimizers.Adam(optimizers_option)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    #저장위치
    save_dir = os.path.join(base_dir,model_name,"results")
    best_model_dir = os.path.join(save_dir,f"{model_name}_Best.h5")
    last_model_dir = os.path.join(save_dir,f"{model_name}_Last.h5")
    fine_tuning_model_dir = os.path.join(save_dir,f"{model_name}_Fine_tuning.h5")
    # 조기종료 옵션
    callback_list=[
        keras.callbacks.EarlyStopping(
            monitor=f'val_{custom_metrics}',
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
    ]
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
    #아웃레이어 세팅    
    out_layer = tf.keras.layers.Conv2D(
        custom_image_size[1], kernel_size = custom_kernel_size, strides=1, padding='SAME', activation='sigmoid')(base_model.output) #2D화(이미지 영상에 적용)
    out_layer = tf.keras.layers.BatchNormalization()(out_layer) #배치 정규화(loss감소를 위함)
    out_layer = Dense(class_num, activation=activation_function)(out_layer)
    model = tf.keras.models.Model(input = base_model.input, outputs = out_layer)
    # 모델 컴파일
    model.compile(
        loss=model_loss_function,
        optimizer=model_optimizer,
        metrics=[custom_metrics,f1_score_m,precision_m, recall_m]
        )
    # 모델 요약 출력
    model.summary()
    #검출시간 계산