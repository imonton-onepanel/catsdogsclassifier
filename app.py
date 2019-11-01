#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE

p = subprocess.Popen('onepanel download inohmonton/datasets/catdogdata input', shell=True)
p.wait()
if p.returncode != 0:
    sys.exit(p.returncode)

train_dir = "./input/dataset/train/"
train_dir = pathlib.Path(train_dir)

valid_dir = "./input/dataset/validation/"
valid_dir = pathlib.Path(valid_dir)

image_count = len(list(train_dir.glob('*/*.jpg')))

CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

BATCH_SIZE = 32  # Typical values are 8, 16 or 32
IMG_HEIGHT = 255
IMG_WIDTH = 255
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)
IMG_SIZE = 75  # Replace with the size of your images
NB_CHANNELS = 3  # 3 for RGB images or 1 for grayscale images
NB_TRAIN_IMG = 6000  # Replace with the total number training images
NB_VALID_IMG = 2000  # Replace with the total number validation images

model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(2, 2),
                 strides=(1, 1),
                 padding='same',
                 input_shape=(IMG_SIZE, IMG_SIZE, NB_CHANNELS),
                 data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=2))
model.add(Conv2D(filters=64,
                 kernel_size=(2, 2),
                 strides=(1, 1),
                 padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = img_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE)

valid_generator = img_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE)

print("Training...")
start = time.time()
model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=20)
end = time.time()
print('Processing time:{:.2}'.format((end - start) / 60))

model.save('../output/model_baseline.h5')
