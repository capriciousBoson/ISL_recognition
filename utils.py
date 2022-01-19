import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import numpy as np
import glob
from PIL import Image
from tensorflow.keras.preprocessing import image
from random import shuffle

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation


def load_image():
    img_path = "test_images\1199.jpg"
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch

