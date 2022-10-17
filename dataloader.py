import tensorflow as tf
import numpy as np
import os
import cv2
from prepro import compute_log_mel_fbank

tmask_max = 64 // 10
fmask_max = 40 // 10


def add_mask(data, cnt=2):
    tlen, flen = data.shape
    for i in range(cnt):
        toft = np.random.randint(0, tlen - tmask_max)
        tmlen = np.random.randint(0, tmask_max)
        data[toft:toft + tmlen] = 0
    for i in range(cnt):
        foft = np.random.randint(0, flen - fmask_max)
        fmlen = np.random.randint(0, fmask_max)
        data[:, foft:foft + fmlen] = 0
    return data


def load_list(list_path='./train.txt', root_path='./pic/'):
    images = []
    labels = []
    outimg = []
    outlabel = []
    with open(list_path, 'r') as f:
        for line in f:
            label, name = line.split('\t')
            label = int(label)
            images.append(root_path + name[:-1])
            labels.append(label)
    f.close()
    #     for i in range(len(images)):
    #         if i %10000 ==0:
    #             print(i)

    #         data = cv2.imread(images[i], 0)
    #         data = cv2.resize(data,(40,256))
    #         data = np.array(data).reshape((256, 40, 1))
    #         outimg.append(data)

    #         label_one_hot = np.zeros(3)
    #         label_one_hot[labels[i]] = 1.0
    #         outlabel.append(label_one_hot)
    #     outimg = np.array(outimg)
    #     outlabel = np.array(outlabel)
    # return outimg, outlabel
    return images, labels


import tensorflow_probability as tfp

tfd = tfp.distributions
def mixup(a, b):
    (image1, label1), (image2, label2) = a, b
    alpha = [0.2]
    beta = [0.2]
    dist = tfd.Beta(alpha, beta)
    l = dist.sample(1)[0][0]
    # l = np.random.random(1)*0.5+0.25
    # l = l.astype(np.float32)
    img = l * image1 + (1 - l) * image2
    lab = l * label1 + (1 - l) * label2

    return img, lab


def load_list2(list_path='./test.txt', root_path='./pic/'):
    images = []
    labels = []
    outimg = []
    outlabel = []
    with open(list_path, 'r') as f:
        for line in f:
            label, name = line.split('\t')
            label = int(label)
            images.append(root_path + name[:-1])
            labels.append(label)
    f.close()

    return images, labels


def load_image(image_path, label):
    image = cv2.imread(image_path.numpy().decode(), 0)
    if image.shape[0] != 64:
        image = cv2.resize(image, (40, 64))
    image = add_mask(image)
    image = image.reshape((64, 40, 1)) / 255

    label_one_hot = np.zeros(2)
    label_one_hot[label] = 1.0

    return image, label_one_hot


def load_image_test(image_path, label):
    image = image_path.numpy().decode()
    image = cv2.imread(image, 0)
    # image = compute_log_mel_fbank(image, n=40)
    if image.shape[0] != 64:
        image = cv2.resize(image, (40, 64))
    image = image.reshape((64, 40, 1)) / 255

    label_one_hot = np.zeros(2)
    label_one_hot[label] = 1.0

    return image, label_one_hot


def train_iterator():
    images, labels = load_list()
    dataset1 = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images), seed=1).map(
        lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset2 = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images), seed=2).map(
        lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    dataset = (
        dataset
        # .shuffle(len(images))
        .map(mixup, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(4)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    dataset = dataset.repeat()

    it = dataset.__iter__()
    return it


def test_iterator():
    images, labels = load_list2()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_test, inp=[x, y], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(1).prefetch(1)
    it = dataset.__iter__()
    return it


if __name__ == '__main__':
    it = train_iterator()
    images, labels = it.next()
    print(labels[0])

    # wav,lable = load_list()
    # print(lable)




