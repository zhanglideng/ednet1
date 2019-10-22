import cv2
import os

path = ['./data/train/', './data/val/', './data/test/']
min_length = 10000000
m_width = 0
min_width = 10000000
m_length = 0
name1 = ''
name2 = ''
count = 0
count_min512 = 0
for i in path:
    # print(i)
    data_list = os.listdir(i)
    for j in data_list:
        count = count + 1
        if count % 200 == 0:
            print(count)
        image = cv2.imread(i + '/' + j)
        # print(image.shape)
        if image.shape[0] < 512 or image.shape[1] < 512:
            count_min512 = count_min512 + 1
        if image.shape[0] < min_length:
            min_length = image.shape[0]
            m_width = image.shape[1]
            name1 = i + '/' + j
            print(name1)
            print('min_length=' + str(min_length))
        if image.shape[1] < min_width:
            min_width = image.shape[1]
            m_length = image.shape[0]
            name2 = i + '/' + j
            print(name2)
            print('min_width=' + str(min_width))
    print('count=' + str(count))

print('the min length image is %s.the size is %d x %d' % (name1, min_length, m_width))
print('the min width image is %s.the size is %d x %d' % (name2, m_length, min_width))
print('the num of image small than 512 is %d' % count_min512)
