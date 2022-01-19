import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from utils import data_augmenter
#from Load_dataset import *
#X_train, X_test, Y_train, Y_test = Load_dataset()

BATCH_SIZE = 128
IMG_SIZE = (128, 128)
directory = r"C:/Users/avina/Documents/Deep Learning/ISL_resnet50/Dataset/"

# Load the image directory and divide into train and validation sets
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='training',
                                             seed=34)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='validation',
                                             seed=34)

class_names = train_dataset.class_names

# prefetch() prevents a memory bottleneck that can occur when reading from disk.
# set the number of elements to prefetch manually,
# use tf.data.experimental.AUTOTUNE to choose the parameters automatically
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input



def isl_model(image_shape=IMG_SIZE):
    input_shape = image_shape + (3,)
    #removing the top layer of mobilenet
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # apply data augmentation to the inputs
    data_augmentation = data_augmenter()
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    #x = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(x)

    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D(name="Avg_Pool")(x)
    # include dropout with probability of 0.2 to avoid overfitting.
    x = tfl.Dropout(0.2)(x)

    # use a prediction layer
    outputs = tfl.Dense(units=35, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

model_2 = isl_model(IMG_SIZE)
base_learning_rate = 0.001
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#print(len(model_2.layers))

initial_epochs = 10
history = model_2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

#save the trained model
model_2.save('trained_model/isl_model_v1.2')


acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
