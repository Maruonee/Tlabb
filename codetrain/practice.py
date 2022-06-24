#definition
def maru_plus(a,b=0):
    return float(a) + float(b)

def maru_minus(a,b=0):
    return float(a) - float(b)

def maru_times(a,b=1):
    return float(a) * float(b)

def maru_division(a,b=1):
    return float(a) / float(b)

def maru_negation(a):
    return -float(a)

def maru_power(a,b):
    return float(a)**float(b)

def maru_remainder(a,b):
    return float(a) % float(b)

print("favoite number plus")
p_a = input()
p_b = input()
favo_num = maru_plus(p_a,p_b)

print(f"your favoite number is {favo_num}")

# print("minus")
# m_a = input()
# m_b = input()
# print(maru_minus(m_a,m_b))

# print("times")
# t_a = input()
# t_b = input()
# print(maru_times(t_a,t_b))

# print("division")
# d_a = input()
# d_b = input()
# print(maru_division(d_a,d_b))

# print("negation")
# n_a = input()
# print(maru_negation(n_a))

# print("power")
# p_a = input()
# p_b = input()
# print(maru_power(p_a,p_b))

# print("remainder")
# r_a = input()
# r_b = input()
# print(maru_remainder(r_a,r_b))

days = ("Mom", "Tue","Wed", "Thu", "Fri")

for day in days:
    print(day)
    
for day in days:
    if day is "Wed":
        break
    else:
        print(day)
        
def say_hello(name, age, day, favoite_num):
    return f"hello {name} you are {age} years old today is {day}, your favoite number is {favoite_num}"


hello = say_hello("hoseong", "32", day, str(favo_num))

print(hello)