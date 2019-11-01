from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess
import sys
import pathlib

p = subprocess.Popen('onepanel download inohmonton/datasets/catdogdata input', shell=True)
p.wait()
if p.returncode != 0:
    sys.exit(p.returncode)

test_data = "./input/dataset/test/"
test_dir = pathlib.Path(test_data)
IMG_SIZE = 75
NB_CHANNELS = 3
BATCH_SIZE = 32

img_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_generator = img_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE)


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

# load model
model.load_weights('./static/model_baseline.h5')

prediction = model.predict(test_generator[:10])