# coding: utf-8
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import tensorflow as tf
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')

    plt.savefig('feature_map.png')
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


if __name__ == "__main__":
    img_path = './001.png'
    weights_path = '../model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # base_model = VGG19("G:heaodi/PythonCode/CNNVisual/model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)
    base_model = VGG19(weights_path, include_top=False)         # VGG net
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block_pool_features = model.predict(x)
    print(block_pool_features.shape)

    feature = block_pool_features.reshape(block_pool_features.shape[1:])

    visualize_feature_map(feature)
