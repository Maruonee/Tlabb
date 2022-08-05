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
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
#================================================================================================
#세팅
base_dir = '/home/tlab/sono/'
model_name = "DenseNet201"
custom_batch = 16
model_dir = f"{base_dir}results/{model_name}_Best.h5"
#================================================================================================
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

#모델불러오기
model = load_model(model_dir)

#모델요약
model.summary()

#테스트
loss, accuracy = model.evaluate(
    test_dataset,
    batch_size=custom_batch,
    verbose=1,
    steps=None,
    workers=16,
    use_multiprocessing=False,
    return_dict=False,
    )

print('Test loss :', loss)
print('Test accuracy :', accuracy)