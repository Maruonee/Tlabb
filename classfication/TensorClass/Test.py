""" 데이터셋 구성
# train
#  -----/class1
#  -----/class2
#  -----/class3
# val
#  -----/class1
#  -----/class2
#  -----/class3
# test
#  -----/class1
#  -----/class2
#  -----/class3
"""
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
#========================================================
#세팅
base_dir = '/home/tlab/sono/' #운영체제에 맞추어 변경해야함
model_name = "DenseNet201"
model_dir = "XXXXXX.h5"
custom_batch = 16
custom_image_size = (512, 512)
cpu_core = 16
#========================================================

#테스트 데이터 셋
test_datagen = ImageDataGenerator(
    rescale=1./255,
)
test_dir = os.path.join(base_dir,'test')
test_dataset = test_datagen.flow_from_directory(
    test_dir,
    batch_size=custom_batch,
    target_size=custom_image_size,
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
    workers=cpu_core,
    use_multiprocessing=False,
    return_dict=False,
    )

print('Test Loss :', loss)
print('Test Accuracy :', accuracy)