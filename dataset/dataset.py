#데이터 세트 나누기
from glob import glob

#이미지 리스트 불러오기
img_list = glob("/home/tlab/dataset/image/*.tif")
print(len(img_list))

#8:2로 나누기
from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=120)
Q
print(len(train_img_list), len(val_img_list))
with open("/home/tlab/dataset/train.txt", 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')
with open("/home/tlab/dataset/val.txt", 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')
