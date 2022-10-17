import json,os,time,pickle,time
import numpy as np
import scipy.io.wavfile as wavfile

def compute_log_mel_fbank_fromsig(signal, sample_rate,n=80):
    # 2.预增强
    # signal = signal[:512]
    # print(signal)
    MEL_N = n

    # 3.分帧
    frame_size, frame_stride = 0.032, 0.032
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,1)
    frames = pad_signal[indices]


    pre_emphasis = 0.97
    for i in range(frames.shape[0]):
        frames[i] = np.append(frames[i][0], frames[i][1:] - pre_emphasis * frames[i][:-1])

    # 4.加窗
    # hamming = np.hamming(frame_length)
    # frames *= hamming

    # 5.N点快速傅里叶变换（N-FFT）
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((mag_frames ** 2))  # 获取能量谱  (1.0 / NFFT) *
    # 6.提取mel Fbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    n_filter = MEL_N  # mel滤波器组的个数, 影响每一帧输出维度，通常取40或80个
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((n_filter, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, n_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    # 7.提取log mel Fbank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 1 * np.log(filter_banks) -3  # dB
    filter_banks[np.where(filter_banks<0)]=0
    filter_banks=filter_banks*10.0
    # print(filter_banks.astype(np.uint8))
    filter_banks = filter_banks.astype(np.uint8)

    # filter_banks = (filter_banks - 127.5) * 0.0078125   #归一化

    # filter_banks = filter_banks *0.0039215686  # 归一化

    # print(filter_banks.dtype)
    return filter_banks

import librosa
def compute_log_mel_fbank(wav_file,n=80,noisy = 0):
    """
    计算音频文件的fbank特征
    :param wav_file: 音频文件
    :return:
    """
    if 'wav' in wav_file:
        sample_rate, signal = wavfile.read(wav_file)
        # print(signal)
    elif 'flac' in wav_file:
        signal, sample_rate = librosa.load(wav_file, sr=None)
        signal = signal * 32768
        signal = signal.astype(np.int16)

    if noisy == 1:
        fbank = compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)
        out = []
        out.append(fbank)
        for i in range(9):
            signal = add_noise(signal, w=0.001*(i+1))
            fbank = compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)
            out.append(fbank)

        return np.array(out)
    else:
        return compute_log_mel_fbank_fromsig(signal, sample_rate, n=n)

def add_noise(x, w=0.008):
    # w：噪声因子
    x = x/32768
    output = x + w * np.random.normal(loc=0, scale=1, size=len(x))
    return output*32768


import os
import cv2
filelist = os.listdir('./wav')
for i in filelist:
    out = compute_log_mel_fbank('./wav/%s'%i,40)
    out = cv2.resize(out,(40,64))
    cv2.imwrite('./pic/%s.jpg'%i[:-4],out)

