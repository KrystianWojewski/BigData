import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
df = pd.read_csv("diabetes.csv")

df["class"] = df["class"].map(lambda x: 1 if x=="tested_positive" else 0)

inputs = df[["pregnant-times","glucose-concentr","blood-pressure","skin-thickness","insulin","mass-index","pedigree-func","age"]].values
classes = df[["class"]].values

train_inputs, test_inputs, train_classes, test_classes = train_test_split(inputs, classes, train_size=0.7, random_state=23215)

from keras.utils import np_utils
train_classes_utl = np_utils.to_categorical(train_classes)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = Sequential()

model.add(Dense(6, activation="relu", input_dim=8))
model.add(Dense(3, activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_inputs, train_classes_utl, validation_split=0.1, epochs=100, batch_size=10, callbacks=[callback])

from sklearn.metrics import accuracy_score
print("---------------------------Results---------------------------")
predictions_train = model.predict(train_inputs)
predictions_train = np.argmax(predictions_train, axis=1)
train_acc = accuracy_score(predictions_train, train_classes)
print("Train acc:", train_acc)
predictions_test = model.predict(test_inputs)
predictions_test = np.argmax(predictions_test, axis=1)
test_acc = accuracy_score(predictions_test, test_classes)
print("Test acc:", test_acc)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_classes.ravel(), predictions_test))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("adam_relu.png")
plt.show()

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# from ann_visualizer.visualize import ann_viz

# ann_viz(model, filename="network", title="Zadanie 5")
