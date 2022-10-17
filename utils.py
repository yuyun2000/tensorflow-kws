import tensorflow as tf


def l2_loss(model, weights=1e-6):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights

def correct_num_batch(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num