import os
import shutil

j = 0
path = './data/new_unlabeled/'
target_path = ['./data/new_train/', './data/new_val/', './data/new_test/']
per = [0, 0.72, 0.75, 1.0]
image_list = os.listdir(path)
image_list.sort(key=lambda x: int(x[:-4]))

for i in target_path:
    if not os.path.exists(i):
        os.makedirs(i)
# 将指定的文件file复制到file_dir的文件夹里面
for i in range(len(image_list)):
    if round(per[j] * len(image_list)) <= i < round(per[j + 1] * len(image_list)):
        shutil.copy(path + image_list[i], target_path[j])
    else:
        print(round(per[j] * len(image_list)))
        print(round(per[j + 1] * len(image_list)))
        j += 1
        shutil.copy(path + image_list[i], target_path[j])

