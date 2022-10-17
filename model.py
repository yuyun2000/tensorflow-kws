import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# （1）标准卷积模块
def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    # 超参数alpha控制卷积核个数
    filters = int(filters * alpha)

    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,  # 步长
                      padding='same',  # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置
    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # relu6激活函数
    return x  # 返回一次标准卷积后的结果

# （2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters * alpha)

    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),  # 卷积核size默认3*3
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1, 1),  # 卷积核默认1*1
                      padding='same',  # 卷积过程中特征图size不变
                      strides=(1, 1),  # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # 激活函数

    return x  # 返回深度可分离卷积结果


def conv_block_withoutrelu(
        inputs,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def mobileinvertedblock(inputs,inc,midc,outc,midkernelsize=(5,5)):
    x = conv_block(inputs,midc,1,kernel_size=(1,1))

    if inc >= outc:
        strides = (1,1)
    else:
        strides = (2,2)
    x = layers.DepthwiseConv2D(kernel_size=midkernelsize,
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=1,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(x)  # 有BN层就不需要偏置
    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # relu6激活函数
    x = conv_block_withoutrelu(x,outc,kernel_size=(1,1))
    if inc == outc:
        return x+inputs
    else:
        return x

# def kwsmodel( input_shape,classes=3):
#     # 创建输入层
#     inputs = layers.Input(shape=input_shape)
#     x = conv_block(inputs, 16, 1, strides=(2, 2))  # 步长为2，压缩宽高，提升通道数
#     x = conv_block(x, 32, 1)
#     x = mobileinvertedblock(x,32,32,64)
#     x = mobileinvertedblock(x,64,80,64,midkernelsize=(3,3))
#     x = mobileinvertedblock(x, 64, 80, 128)
#     x = mobileinvertedblock(x, 128, 128, 128)
#     x = mobileinvertedblock(x, 128, 96, 128, midkernelsize=(3, 3))
#     x = mobileinvertedblock(x,128,168,168,midkernelsize=(3,3))
#     x = mobileinvertedblock(x, 168, 196, 168)
#     x = mobileinvertedblock(x, 168, 168, 168, midkernelsize=(3, 3))
#     # x = mobileinvertedblock(x, 128, 168, 168, midkernelsize=(3, 3))
#     x = mobileinvertedblock(x, 168, 256, 168, midkernelsize=(3, 3))
#     # x = mobileinvertedblock(x, 168, 256, 256, midkernelsize=(3, 3))
#     x = mobileinvertedblock(x, 168, 256, 128, midkernelsize=(3, 3))
#     # 调整输出特征图x的特征图个数
#     # 卷积层，将特征图x的个数转换成分类数
#     x = layers.GlobalAveragePooling2D()(x)  # 通道维度上对size维度求平均
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(classes*20)(x)
#     x = layers.ReLU()(x)
#     x = layers.Dense(classes * 10)(x)
#     x = layers.Dense(classes)(x)
#     x = layers.Softmax()(x)
#     # 构建模型
#     model = Model(inputs, x)
#     # 返回模型结构
#     return model

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Softmax, Activation, BatchNormalization, Flatten, Dropout, DepthwiseConv2D
from tensorflow.keras.layers import MaxPool2D, AvgPool2D, AveragePooling2D, GlobalAveragePooling2D,ZeroPadding2D,Input,Embedding,PReLU
def kwsmodel(dim0=16):
    dst_h = 64
    dst_w = 40
    dst_ch = 1;
    model = Sequential()
    model.add(Conv2D(dim0, (3, 3), padding='same', strides=(2, 2), input_shape=(dst_h, dst_w, dst_ch), name='ftr00'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 32x32
    model.add(DepthwiseConv2D((3, 3), padding='same', name='ftr01'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 32x32

    model.add(Conv2D(dim0 * 2, (3, 3), padding='same', strides=(2, 2), name='ftr10'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 16x16
    model.add(DepthwiseConv2D((5, 5), padding='same', name='ftr11'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 32x32

    model.add(Conv2D(dim0 * 4, (3, 3), padding='same', strides=(2, 2), name='ftr20'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 8x8
    model.add(DepthwiseConv2D((3, 3), padding='same', name='ftr21'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 32x32

    model.add(Conv2D(dim0 * 8, (3, 3), padding='same', strides=(2, 2), name='ftr30'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 8x8
    model.add(DepthwiseConv2D((3, 3), padding='same', name='ftr31'));
    model.add(BatchNormalization());
    model.add(Activation('relu'));  # 32x32

    # model.add(DepthwiseConv2D((16,3), padding = 'valid', name='ftr40'));model.add(BatchNormalization());model.add(Activation('relu')); #8x8

    # model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D(name='GAP'))
    # model.add(AveragePooling2D(10,7))
    model.add(Dropout(0.5))
    model.add(Dense(dim0 * 4, name="fc0"))
    model.add(Dense(dim0 * 2, name="fc1"))
    model.add(Dense(2, name="fc2"))
    model.add(Activation('softmax', name="sm"))
    return model



if __name__ == '__main__':
    # 获得模型结构
    model = kwsmodel()
    # # 查看网络模型结构
    model.summary()
    # model.save("./mbtest.h5", save_format="h5")

    # print(model.layers[-3])

    # model = tf.keras.models.load_model("./mbtest.h5")
    # model.summary()