from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.model_selection import train_test_split
datasets = train_test_split(iris.data, iris.target, test_size=0.3)

train_data, test_data, train_labels, test_labels = datasets

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(4, 2), max_iter=3000)
mlp.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score

print("------------------------------- 4,2,1 -------------------------------")
predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=3000)
mlp.fit(train_data, train_labels)

print("------------------------------- 4,3,1 -------------------------------")
predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

mlp = MLPClassifier(hidden_layer_sizes=(4, 3, 3), max_iter=3000)
mlp.fit(train_data, train_labels)

print("------------------------------- 4,3,3,1 -------------------------------")
predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))




