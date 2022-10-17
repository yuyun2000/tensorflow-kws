import matplotlib.pyplot as plt
from vad import VoiceActivityDetector
import wave
import numpy as np
import scipy.io.wavfile as wavfile

# -*- coding: utf-8 -*-
# 读Wave文件并且绘制波形

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

# 打开WAV音频
f = wave.open('./ze.wav', "rb")

# 读取格式信息
# (声道数、量化位数、采样频率、采样点数、压缩类型、压缩类型的描述)
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# nchannels通道数 = 2
# sampwidth量化位数 = 2
# framerate采样频率 = 22050
# nframes采样点数 = 53395

# 读取nframes个数据，返回字符串格式
str_data = f.readframes(nframes)

f.close()

# 将字符串转换为数组，得到一维的short类型的数组
wave_data = np.fromstring(str_data, dtype=np.short)

# 赋值的归一化
wave_data = wave_data * 1.0 / (max(abs(wave_data)))

# 整合左声道和右声道的数据
wave_data = np.reshape(wave_data, [nframes, nchannels])
# wave_data.shape = (-1, 2)   # -1的意思就是没有指定,根据另一个维度的数量进行分割

# 最后通过采样点数和取样频率计算出每个取样的时间
time = np.arange(0, nframes) * (1.0 / framerate)

plt.figure()

plt.subplot(1, 1, 1)
plt.plot(time, wave_data[:])
plt.xlabel("时间/s",fontsize=14)
plt.ylabel("幅度",fontsize=14)
plt.grid()  # 标尺
plt.tight_layout()  # 紧密布局



#vad 切割

load_file = "ze.wav"
sample_rate, signal = wavfile.read(load_file)
# 获取vad分割节点
v = VoiceActivityDetector(load_file)
raw_detection = v.detect_speech()
speech_time = v.convert_windows_to_readible_labels(raw_detection)
print(speech_time)
for i, time in enumerate(speech_time):
    s = time['speech_begin']
    e = time['speech_end']
    plt.axvline(s,color = 'red')
    plt.axvline(e,color = 'red')
    print(s, e)

plt.show()


