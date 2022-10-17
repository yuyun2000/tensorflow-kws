import os
from copy import deepcopy
from random import randint
import cv2

def shuffle(lst):
  temp_lst = deepcopy(lst)
  m = len(temp_lst)
  while (m):
    m -= 1
    i = randint(0, m)
    temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
  return temp_lst

file1 = open('train.txt', mode='a+')
file2 = open('test.txt', mode='a+')
list = os.listdir('./pic/')
list = shuffle(list)
num = 0
for i in range(len(list)):
    if list[i][0] == 'b':
        label = '0'
    else:
        label = '1'

    if i <len(list)*0.8:
        file1.write(label + '\t' + '%s\n' % (list[i]))
    else:
        file2.write(label + '\t' + '%s\n' % (list[i]))

file1.close()
file2.close()


