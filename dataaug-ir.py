'''
读取flac音频并进行ir效果
保存为wav
'''
import os
import numpy as np
import cv2
from tensorflow.keras import layers, Model, optimizers
import scipy.io.wavfile as wavfile

ir_wav = "./IR_0.25_sweep_old.wav"
fs,irdata=wavfile.read(ir_wav)

inputs = layers.Input(shape=(None,1))
x = layers.Conv1D(1, fs, name="ir_conv")(inputs)  #设定ir长度为1s
model = Model(inputs, x, name="ir")
a=model.get_layer(index=1)
w = a.get_weights()
#print(w[0].shape)
w[0][:,0,0] = irdata/32768
a.set_weights(w)



listwav = os.listdir('./trainwav')
for i in range(len(listwav)):

    sample_rate, signal = wavfile.read('./trainwav/%s'%listwav[i])

    res = model.predict(signal.reshape((1, -1, 1))/32768)
    res = (res[0, :, 0] * 32767).astype(np.int16)

    wavfile.write("./irdata/%s"%listwav[i], 16000, res)

