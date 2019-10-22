import os
import shutil

path = ['./data/train/', './data/val/', './data/test/']
new_path = ['./data/mini_train/', './data/mini_val/', './data/mini_test/']
for i in new_path:
    if not os.path.exists(i):
        os.makedirs(i)
for i in range(len(path)):
    data_list = os.listdir(path[i])
    length = len(data_list)
    length = length // 100
    for j in range(length):
        shutil.copy(path[i] + data_list[j], new_path[i])
    print('finish the %s copy' % i)
