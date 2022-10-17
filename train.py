from model import kwsmodel
import os
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras import optimizers
from dataloader import train_iterator
from utils import *


def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)

        cross_entropy = tf.keras.losses.categorical_crossentropy(labels, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        l2 = l2_loss(model)
        loss = cross_entropy + l2

        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction


def train(model, data_iterator, optimizer):
    bs = 4
    for i in tqdm(range(int(132 / bs))):
        images, labels = data_iterator.next()
        ce, prediction = train_step(model, images, labels, optimizer)
        correct_num = correct_num_batch(labels, prediction)
        print('loss: {:.6f}, accuracy: {:.4f}'.format(ce, correct_num / bs))


class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)


if __name__ == '__main__':
    train_data_iterator = train_iterator()

    model = kwsmodel(dim0=4)
    model.build(input_shape=(None,) + (64, 40, 1))
    # model.load_weights('./h5/kws-l2-mixup-14.h5')

    # model = tf.keras.models.load_model("./h5/hand-198.h5")

    model.summary()

    # optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    optimizer = optimizers.Adam()

    from test import test
    for epoch_num in range(100):
        train(model, train_data_iterator, optimizer)
        if epoch_num % 1 == 0:
            with open('test_log_3x3_7k.txt', 'a') as f:
                test(model, f)
            model.save('./h5/kws-7k-mixup-64-3x3-%s.h5' % epoch_num, save_format='h5')




