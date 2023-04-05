from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import argparse
import cv2
import pickle
import os
import tensorflow as tf

SIZE = 50

imagePaths = sorted(list(paths.list_images("dogs-cats-mini")))

def label_img(img):
    word_label = img.split('.')[-3]
    word_label = word_label[-3:]

    if word_label == 'cat': return [1,0]

    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for path in imagePaths:
        label = label_img(path[15:])
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (SIZE, SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()

train, test = train_test_split(train_data, train_size=0.9)

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# convnet = input_data(shape=[None, SIZE, SIZE, 1], name='input')
#
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
#
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
#
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
#
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
#
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
#
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
#
# convnet = fully_connected(convnet, 2, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
#
# model = tflearn.DNN(convnet)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(SIZE, SIZE, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = np.array([i[0] for i in train]).reshape(-1, SIZE, SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, SIZE, SIZE,1)
test_y = [i[1] for i in test]

history = model.fit(X, Y, n_epoch=1, validation_set=(test_x, test_y), batch_size=32)

# from sklearn.metrics import accuracy_score
# predictions_train = model.predict(X)
# predictions_train = np.argmax(predictions_train, axis=1)
# train_acc = accuracy_score(predictions_train, Y)
# print("Train acc:", train_acc)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig("adam_relu.png")
# plt.show()

import matplotlib
matplotlib.use('TkAgg')

fig = plt.figure()
for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(SIZE, SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
