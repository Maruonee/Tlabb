import os
from keras_preprocessing.image import ImageDataGenerator
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
#데이터위치
train_dir = '/home/tlab/train'
custom_batch = 16
#학습 데이터셋
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )

i=0

for batch in train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=custom_batch,
    class_mode='categorical',
    save_to_dir='/home/tlab/sss'
    ):
    i+= 1
    if i>custom_batch:
        break