# example of loading the mnist dataset
import os

import cv2
import numpy as np
from keras.datasets.cifar10 import load_data
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

# load the images into memory
# TEST_DATASET_DIR = 'pokemon_jpg/pokemon_jpg'
# X = np.array([])
# for img_name in os.listdir(TEST_DATASET_DIR):
#     img = cv2.imread(os.path.join(TEST_DATASET_DIR, img_name))
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = (img.astype('float32') - 127.5) / 127.5
#     X = np.append(X, img)
#
# a, b = train_test_split(X, train_size=0.7)
#
# print(a.shape)
# print(b.shape)
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

# plot images from the training dataset
for i in range(25):
 # define subplot
 pyplot.subplot(5, 5, 1 + i)
 # turn off axis
 pyplot.axis('off')
 # plot raw pixel data
 pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()
