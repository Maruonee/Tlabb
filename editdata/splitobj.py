#데이터 세트 나누기
from glob import glob
from sklearn.model_selection import train_test_split
import os
#데이터셋 위치
data_dir = '/home/tlab/dataset'

#이미지 리스트 불러오기
img_list = glob(os.path.join(data_dir,"images","*.jpg"))

# 트레이닝셋, 벨류, 테스트 6:2:2
train_img_list, val_test_img_list = train_test_split(
    img_list,
    test_size=0.4,
    shuffle=True,
    stratify=None,#Classification할대는 class로 나눠야함
    random_state=34)
val_img_list, test_img_list = train_test_split(
    val_test_img_list,
    test_size=0.5,
    shuffle=True,
    stratify=None,#Classification할대는 class로 나눠야함
    random_state=34)

#저장
with open(os.path.join(data_dir,"train.txt"), 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')
with open(os.path.join(data_dir,"val.txt"), 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')
with open(os.path.join(data_dir,"test.txt"), 'w') as f:
  f.write('\n'.join(test_img_list) + '\n')
  
#이미지수
print("Total image : ",len(img_list),"\n"
      "Train image : ",len(train_img_list),"\n"
      "Validation image : ",len(val_img_list),"\n"
      "Test image : ",len(test_img_list))