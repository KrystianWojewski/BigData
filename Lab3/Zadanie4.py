import pandas as pd

from sklearn.model_selection import train_test_split
df = pd.read_csv("diabetes.csv")

df["class"] = df["class"].map(lambda x: 1 if x=="tested_positive" else 0)

inputs = df[["pregnant-times","glucose-concentr","blood-pressure","skin-thickness","insulin","mass-index","pedigree-func","age"]].values
classes = df[["class"]].values

train_inputs, test_inputs, train_classes, test_classes = train_test_split(inputs, classes, train_size=0.7, random_state=23215)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(hidden_layer_sizes=(8,6,3), max_iter=500)
mlp.fit(train_inputs, train_classes.ravel())

print("-------------------------------- 8,6,3 --------------------------------")
predictions_train = mlp.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_classes.ravel(), predictions_test))

mlp = MLPClassifier(hidden_layer_sizes=(8,12,6,3), max_iter=10000)
mlp.fit(train_inputs, train_classes.ravel())

print("-------------------------------- 8,12,6,3 --------------------------------")
predictions_train = mlp.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_classes.ravel(), predictions_test))
