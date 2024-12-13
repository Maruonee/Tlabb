import numpy as np
# NPY 파일에서 데이터 로드
loaded_array = np.load('/home/tlab4090/testmask.npy')

print(loaded_array)
print(type(loaded_array))
print(np.sum(np.sum(loaded_array, axis = 2)))