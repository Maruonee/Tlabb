#데이터 세트 나누기
from glob import glob
from sklearn.model_selection import train_test_split
import os
#데이터셋 위치
data_dir = '/home/tlab4090/datasets/images/'
save_dir = '/home/tlab4090/datasets/opt/'
#이미지 리스트 불러오기
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
img_list = []
for file_name in os.listdir(data_dir):
    if os.path.splitext(file_name)[1].lower() in image_extensions:
        img_list.append(os.path.join(data_dir, file_name))

# # 전체 데이터셋을 6:2:2 비율로 분할 20%는 테스트 세트
# train_val_img_list, test_img_list = train_test_split(
#     img_list,
#     test_size=0.2,
#     shuffle=True,
#     stratify=None,
#     random_state=34
# )
# # 남은 80%의 데이터셋을 다시 6:2 비율로 학습 및 검증 세트로 분할
# train_img_list, val_img_list = train_test_split(
#     train_val_img_list,
#     test_size=0.25,  # 80% 중 25%는 전체의 20%에 해당
#     shuffle=True,
#     stratify=None,
#     random_state=34
# )

# 전체 데이터셋을 50%는 테스트 세트
train_val_img_list, test_img_list = train_test_split(
    img_list,
    test_size=0.5,
    shuffle=True,
    stratify=None,  # 분류 문제의 경우, 클래스 비율을 유지하려면 적절히 설정
    random_state=34
)

# 남은 50%의 데이터셋을 3:2 비율로 학습 및 검증 세트로 분할
train_img_list, val_img_list = train_test_split(
    train_val_img_list,
    test_size=2/5,  # 50% 중 2/5는 전체의 20%에 해당
    shuffle=True,
    stratify=None,  # 분류 문제의 경우, 클래스 비율을 유지하려면 적절히 설정
    random_state=34
)

#저장
with open(os.path.join(save_dir,"train.txt"), 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')
with open(os.path.join(save_dir,"val.txt"), 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')
with open(os.path.join(save_dir,"test.txt"), 'w') as f:
  f.write('\n'.join(test_img_list) + '\n')
with open(os.path.join(save_dir, "datasets.txt"), 'w') as f:
    f.write("Total image : " + str(len(img_list)) + "\n" +
            "Train image : " + str(len(train_img_list)) + "\n" +
            "Validation image : " + str(len(val_img_list)) + "\n" +
            "Test image : " + str(len(test_img_list)))

print("Total image : " + str(len(img_list)) + "\n" +
            "Train image : " + str(len(train_img_list)) + "\n" +
            "Validation image : " + str(len(val_img_list)) + "\n" +
            "Test image : " + str(len(test_img_list)))
    
print("save done")