abs(-5) # 5
pow(4,2) # 4^2
max(4,23) # 23
minI(4,23) # 4
round(3.14) # 3
randomrange(1,46)  #1~46미만의 임의 값 생성
############################################################
number_f_day = (int(randrange(3,29))) # 3~29

def maru_plus(a,b=0):
    return float(a) + float(b) # a + b

plu_s = maru_plus(1,3)

print(f"your number is {plu_s}")
############################################################

def say_hello(name, age, output_day, favoite_num):
    return f"hello {name} you are {age} years old today is {output_day}, your favoite number is {favoite_num}"
hello = say_hello("hoseong", "32", day, str(favo_num))
print(hello)
############################################################        
input_days = ("Mom", "Tue","Wed", "Thu", "Fri")

for output_day in input_days:
    print(output_day)
for output_day in input_days:
    if output_day is "Wed":
        break
    else:
        print(output_day)
 
###########################################################
adress_weburl = ("http://yo.com")

remove_http = adress_weburl.replace("http://","")
number_sitename = remove_http[:remove_http.find(".")]
make_pass_two = len(number_sitename)
if make_pass_two >= 3:
    make_pass_one = number_sitename[:3]
else:
    make_pass_one = number_sitename

make_pass_three = adress_weburl.count("e")
maked_pass = print(f"{adress_weburl} 에서 생성된 비밀번호는 {make_pass_one}{make_pass_two}{make_pass_three}!입니다.")
############################################################

weather = input("오늘 날씨 어때요?")
if weather == "비" or weather == "눈":
    print("우산을 챙기세요")
elif weather == "미세먼지":
    print("마스크를 쓰세요")
else : 
    print("그냥 나가세요")
weather = float(input("오늘 온도는 어때요?"))
if 30 <= weather:
     print("더워요")
elif 10<= weather and weather < 30:
     print("좋은 날씨")
elif 0<= weather < 10:
     print("쌀쌀")
else : 
     print("개추워요")  
    

##################################################
name_list = ["황호성","김호철","김동현"]
for waiting_no in range(1,30):
    for list in name_list:
        print("%s 대기번호 %d" % (list, waiting_no))

 
###################################################
absent = [2,5]
no_book = [7]
for student in range(1,11):
    if student in absent:
        continue
    elif student in no_book:
        print("오늘수업은 여기까지 %d는 책사와" % (student))
        break
    print("%s 책 읽어봐"%(student))

############################################################

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))

student = [1,2,3,4,5]
print(student)

student = [i+100 for i in student]
print(student)

############################################################

import random
# tim = random.shuffle(range(1,51))
ind_num = int(input("손님수를 넣으세요."))
lis = list(range(1,ind_num))
tim = [random.randrange(1,61) for i in range(ind_num)]

tim = []
lis = []
for p_list in range(1,51):  
    rd_time = int(randrange(5,51))
    tim.append(rd_time)
    lis.append(p_list)
    if 5<= rd_time <=15: #탑승가능 손님
        print(f"{p_list}번째 손님 (소요시간 : {rd_time})")
    else:
        print(f"{p_list}번째 손님 (소요시간 : {rd_time})")


str_lits = list(map(str, lis)) #문자열 변환
name_list = ["택시 손님"+i for i in str_lits] # 택시손님 문자열 추가
texi_list = dict(zip(name_list,tim)) # 손님 : 시간
print(f"손님 {ind_num}명이 탑승 대기중 입니다.")
for key, value in texi_list.items():
    if value <= 15:
        print(f"하지만 탑승 가능한 손님은 {key}입니다.")

early_time = min(texi_list.values())

r_texi_list = {v:k for k,v in texi_list.items()} #key value 뒤집기
early_list=r_texi_list.get(early_time)

print(f"가장 가까이 가는 손님은 {early_list}이며 소요시간은 {early_time}분 입니다.").

############################################################

def checkpoint(gun, soldiers, optiom=False):
    gun = gun - soldiers
    print(f"부대내 남은 총은 {gun}입니다.") 
    return gun, soldiers

gun_num = int(input("부대 총의 수를 입력하시오."))
soldiers_num = int(input("경계초소 인원을 입력하시오."))

gun, soldiers = checkpoint(gun_num, soldiers_num)

print(gun, soldiers)
############################################################
def std_weight(c_height, c_gender):
    if c_gender =="":
        normal_wei = round((c_height)*(c_height) * 22 / 10000,2)
        print(f"키 {c_height}, 남자의 표준 체중은 {normal_wei}Kg 입니다.")
        return normal_wei
    else:
        normal_wei = round((c_height)*(c_height) * 21 / 10000)
        print(f"키 {c_height}, 여자의 표준 체중은 {normal_wei}Kg 입니다.")
        return normal_wei

height = float(input("키를 입력하시오, (cm단위) "))
gender = input("여자라면 아무키나 입력하세요. 아니라면 엔터치세요 ")

normal_wei = std_weight (height, gender)

############################################################
score_file = open("./score.txt","w", encoding = "utf8")
score_test = {'수학': 1, '영어' : 50}

print(score_test, file=score_file)