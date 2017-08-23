#!/usr/bin/env python

import numpy as np
import os
import glob
import fnmatch
import random

from keras import applications
from keras.preprocessing.image   import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

img_train_dir = "/home/aicg2/datasets/aic540/train/images"
label_train_dir = "/home/aicg2/vgg_aic/aic540_vgg_train_labels"
img_val_dir = "/home/aicg2/datasets/aic540/val/images"
label_val_dir = "/home/aicg2/vgg_aic/aic540_vgg_val_labels"

img_width, img_height = 224, 224
nb_classes = 15

def preprocess_input(x, dim_ordering='default'):
    """ preprocess image data for vgg model
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

def gen_data_file(img_files, label_files):
    """ Gerantes data using img and label files
    """
    X = []
    y = []
    for idx, img_file in enumerate(img_files):
        label_file = label_files[idx]
        y1 = np.load(label_file)
        img = load_img(img_file, target_size = (img_width, img_height))
        x1 = img_to_array(img)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)
        X.append(x1)
        y.append(y1)

    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

def gen_img_label_file(img_dir, label_dir, shuffle=True):
    """ Generates list of image and corresponding label files
    """
    label_file_path = []
    img_file_path = []
    img_files = fnmatch.filter(os.listdir(img_dir),  '*.jpeg')
    if shuffle is True:
        random.shuffle(img_files)
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.split(".jpeg")[0] + ".npy")
        if os.path.exists(label_path) is False:
            print 'WARNING: label not found for '+ img_path + ' at ' + label_path
            continue
        else:
            print 'INFO" label found for '+ img_path + ' at ' + label_path

        img_file_path.append(img_path)
        label_file_path.append(label_path)

    return img_file_path, label_file_path

def gen_data_dir(img_dir, label_dir, shuffle=True):
    """ Takes img and label directory
    """
    img_file_path, label_file_path = gen_img_label_file(img_dir, label_dir, shuffle)
    return gen_data_file(img_file_path, label_file_path)

def gen_batch(img_dir, label_dir, batch_size, shuffle=True):
    """ Generates batches of data
    """
    img_file_path, label_file_path = gen_img_label_file(img_dir, label_dir, shuffle)
    num_images = len(img_file_path)
    while True:
        for i in range(0, num_images-batch_size, batch_size):
            X, y = gen_data_file(img_file_path[i:i+batch_size], label_file_path[i:i+batch_size])
            yield X, y

def get_model():
    """ Get vgg19 model for regression experiment
    """
    # import VGG model and use it
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers:
        layer.trainable = False

    #Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(nb_classes, activation="linear")(x)

    # creating the final model
    model = Model(input = model.input, output = predictions)

    # compile the model
    model.compile(loss = "mean_squared_error", optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=["accuracy"])
    print model.summary()

    return model

def _main():
    batch_size = 16
    epochs = 50
    model = get_model()
    checkpoint = ModelCheckpoint('aic.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # get validation data
    print 'generating validation data'
    x_val, y_val = gen_data_dir(img_val_dir, label_val_dir)
    xy_val = (x_val, y_val)
    img_files, label_files = gen_img_label_file(img_train_dir, label_train_dir)
    num_train_samples = len(img_files)
    print 'num train images', num_train_samples, ' and labels',len(label_files)

    # Train the model
    print 'fitting model'
    model.fit_generator(
    generator = gen_batch(img_train_dir, label_train_dir, batch_size),
    steps_per_epoch = int(num_train_samples / batch_size) + 100,
    epochs = epochs,
    validation_data = xy_val,
    callbacks = [checkpoint, early])

    score = model.evaluate(x_val, y_val)
    print model.predict(x_val)

_main()
