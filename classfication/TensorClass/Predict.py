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
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

#================================================================================================
#세팅
predict_imgdir = "/home/tlab/sono/test/Fail"
custom_batch = 16
model_dir = "/home/tlab/Results/Classification/DenseNet201_Best.h5"
#예측 이미지 생성

for fn in uploaded.keys():
  path = predict_imgdir
  img=image.load_img(path, target_size=(150, 150))
  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=custom_batch)
  print(classes[0])

if classes[0]>0:
    print(fn + " is a dog")
else:
    print(fn + " is a cat")
    
# predict_datagen = ImageDataGenerator(
#     rescale=1./255,
#     )
# predict_path = os.path.join(predict_dir,class_name)
# predict_dataset = predict_datagen.flow_from_directory(
#     predict_path,
#     batch_size=custom_batch,
#     target_size=(512, 512),
#     class_mode='categorical',
#     )

plt.figure(figsize=(6,6))

for index in range(25):
    plt.subplot(5,5,index + 1)  
    plt.imshow(predict_dataset)
    plt.axis('off')
    
plt.savefig(f'{predict_dir}{model_name}.png')

# #모델선정
# model = load_model(f"{model_dir}")

# def printmd(string):
#     display(Markdown(string))
# class_dictionary = {'airplane': 0,
#                     'car': 1,
#                     'cat': 2,
#                     'dog': 3,
#                     'flower': 4,
#                     'fruit': 5,
#                     'motorbike': 6,
#                     'person': 7}
# IMAGE_SIZE    = (224, 224)

# test_image = image.load_img(test_df.iloc[number_1, 0]
#                             ,target_size =IMAGE_SIZE )
# test_image = image.img_to_array(test_image)
# plt.imshow(test_image/255.);

# test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1], test_image.shape[2]))
# test_image = preprocess_input(test_image)
# prediction = model.predict(test_image)

# df = pd.DataFrame({'pred':prediction[0]})
# df = df.sort_values(by='pred', ascending=False, na_position='first')
# printmd(f"## 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")

# for x in class_dictionary:
#   if class_dictionary[x] == (df[df == df.iloc[0]].index[0]):
#     printmd(f"### Class prediction = {x}")
#     break



# fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
#                         subplot_kw={'xticks': [], 'yticks': []})

# for i, ax in enumerate(axes.flat):
#     ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
#     ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred[i].split('_')[0]}", fontsize = 15)
# plt.tight_layout()
# plt.show()












































predict_img = image.load_img(predict_dir, target_size=(150, 150))

#   x=image.img_to_array(img)
#   x=np.expand_dims(x, axis=0)
#   images = np.vstack([x])

predictions = model.predict(predict_dir)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print(predictions[0])

if predictions[0]>0:
    print(fn + " is a pass")
else:
    print(fn + " is a fail")