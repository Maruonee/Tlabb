import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from tensorflow.compat.v1.keras.backend import set_session


# Dataset setting

# train
# -----/class1
# -----/class2
# -----/class3
# test
# -----/class1
# -----/class2
# -----/class3

base_dir = '/media/tlab/Data/Dataset/Sono/Axial/'
train_dir = os.path.join(base_dir,'train')
validation_dir= os.path.join(base_dir,'val')

# batchsize and epochs setting
CustomBatch = 16
Customepochs = 2000
Class_num = 2

# train image set
train_datagen = ImageDataGenerator(
    rescale=1./255
    )
# train image set Augmentation function
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode=`nearest`)
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

# validation image set
validation_datagen = ImageDataGenerator(
    rescale=1./255
    )

# Data Set setting
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=CustomBatch,
    class_mode='categorical',
#categorical : 2D one-hot 부호화된 라벨이 반환
#binary : 1D 이진 라벨이 반환
#sparse : 1D 정수 라벨이 반환
#None : 라벨이 반환되지 않음
    )

validation_dataset = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(512, 512),
    batch_size=CustomBatch,
    class_mode='categorical'
    )
#Class name and visualize dataset

###############Transfer Learning base model set#################

base_model = tf.keras.applications.densenet.DenseNet201(
    include_top=False,
    weights='imagenet',
    )
#   input_tensor=None
#   input_shape=None
#   pooling=None
#   classes=1000
base_model.trainable = False

# Outlayer set
out_layer = tf.keras.layers.Conv2D(512, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) 
out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
out_layer = tf.keras.layers.Dense(Class_num, activation='softmax')(out_layer)

# Outlayer Set and model
model = tf.keras.models.Model(base_model.input, out_layer)
# model compile
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
    )

model.summary()

# Trainning Record
history = model.fit(
    train_dataset,
    epochs=Customepochs,
    validation_data=validation_dataset,
    verbose=1
    )

# model save location
model.save("dencenet201.h5")

#visualize Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.savefig('dencenet201 Accuracy.png')

plt.figure()
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('dencenet201 Loss.png')
