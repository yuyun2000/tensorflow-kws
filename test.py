import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataloader import test_iterator
from utils import  correct_num_batch, l2_loss
from model import kwsmodel
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    cross_entropy = tf.keras.losses.categorical_crossentropy(labels, prediction)

    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy, prediction

def test(model, log_file):
    data_iterator = test_iterator()

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(int(33/1))):
        images, labels = data_iterator.next()
        ce, prediction = test_step(model, images, labels)
        correct_num = correct_num_batch(labels, prediction)

        sum_ce += ce * 1
        sum_correct_num += correct_num
        # print('ce: {:.4f}, accuracy: {:.4f}'.format(ce, correct_num / 1))

    log_file.write('test: cross entropy loss: {:.4f}, l2 loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / 33,
                                                                                                  l2_loss(model),
                                                                                                  sum_correct_num / 33))

if __name__ == '__main__':
    # gpu config
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model

    # model = kwsmodel(dim0=4)  # 随即杀死神经元的概率
    # model.build(input_shape=(None,) + (256,40,1))
    # model.load_weights('./h5/kws0930.h5')

    model = tf.keras.models.load_model('./h5/kws-7k-mixup-64-3x3-94.h5')
    # model = tf.keras.models.load_model('./h5/kws-l2-mixup-14.h5')
    # model.summary()

    import scipy.io.wavfile as wavfile
    from prepro import compute_log_mel_fbank_fromsig

    filename = './t2.wav'

    sample_rate, signal = wavfile.read(filename)
    # print(signal.shape)
    out = compute_log_mel_fbank_fromsig(signal, sample_rate,n=40) #128*s,40
    out = cv2.resize(out,(40,64))
    # # out = np.pad(out,((0,128-112),(0,0)))
    # # out = cv2.imread('./train/0-0-original.jpg',0)
    out = model(out.reshape(1,64,40,1)/255,training=False)
    print(out)

    # test
    # with open('test_log.txt', 'a') as f:
    #     test(model, f)
