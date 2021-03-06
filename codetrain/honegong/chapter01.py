#%%
from ast import Break
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#데이터 베이스
bream_length = [
    25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0
    ]
bream_weight = [
    242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0
    ] 
smelt_length = [
    9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
    ]
smelt_weight = [
    6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
    ]

#%%
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show
# %%
#합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
#배열 속 배열구조로 쌍으로 만들기
fish_data = [[l,w] for l, w in zip(length, weight)]
#앞 35과 뒤 14 구분
fish_target = [1] * 35 + [0]* 14
print(f"데이터셋입니다.\n{fish_data}\n데이터셋 끝")

while True:
    K_point = int(input(f"K값을 입력하세요({len(fish_data)}이하로 입력하세요) "))
    if K_point < len(fish_data):
        break
#학습 k값 정하기
kn = KNeighborsClassifier(n_neighbors=K_point)
#kn 트레이닝
kn.fit(fish_data, fish_target)
print("학습완료!")
#%%
#트레이닝 결과(정확도)
A_fish = kn.score(fish_data, fish_target)
print(f"{int((A_fish)*1000)/10}%의 정확도 입니다.")
#%%
#예측
F_length = int(input("생선 길이를 입력하세요. "))
F_weight = int(input("생선 무게를 입력하세요. "))

if str(kn.predict([[F_length,F_weight]])) == "[1]":
     print("도미입니다.(bream)")
else:
     print("빙어입니다.(smlet)")
     
# %%
print(kn._fit_X,"\n", kn._y)

# %%
