import pandas as pd

from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")

df["Setosa"] = df["variety"].apply(lambda x: 1 if x == "Setosa" else 0)
df["Virginica"] = df["variety"].apply(lambda x: 1 if x == "Virginica" else 0)
df["Versicolor"] = df["variety"].apply(lambda x: 1 if x == "Versicolor" else 0)

inputs = df[["sepal.length","sepal.width","petal.length","petal.width"]].values
classes = df[["Setosa","Virginica","Versicolor"]].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(inputs, classes, train_size=0.7)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=5000)
mlp.fit(train_inputs, train_classes)

from sklearn.metrics import accuracy_score

predictions_train = mlp.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))
