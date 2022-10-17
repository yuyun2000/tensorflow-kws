'''
把带有关键词的音频取前三分之一和随机噪声音频，再取后三分之一和随机噪声拼接
'''
import os
import scipy.io.wavfile as wavfile
import numpy as np

listall = os.listdir('./wav')
list1 = [] #关键词列表
list2 = os.listdir('./bg') #背景音列表,使用原来的长音频

for i in listall:
    if 'kword-org' in i:
        list1.append(i)

for id,i in enumerate(list1):
    sample_rate, signal = wavfile.read('./wav/%s'%i)
    idx = np.random.randint(0,len(list2))
    rate , bgsig = wavfile.read('./bg/%s'%list2[idx])

    mix1 = bgsig[:32000]
    oft = np.random.randint(0,23000) #这个偏移控制在背景音上关键词的位置
    oft2 = np.random.randint(1000, len(signal)-6500) #这个偏移控制关键词的起点位置
    oft3 = np.random.randint(5000,6000)#这个偏移控制截取关键词的长度 大概是一个字或接近两个字
    mix1[oft:oft+oft3] = signal[oft2:oft2+oft3]

    wavfile.write('./wav/bg-cut-%s.wav'%(id),16000,mix1)

