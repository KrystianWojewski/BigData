import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("iris.csv")
# print(df.values)

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=23215)

# print(test_set)
# print(test_set.shape[0])

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

def classify_iris(sl, sw, pl, pw):
    if pw  < 1:
        return("Setosa")
    elif pl > 5:
        return("Virginica")
    else:
        return ("Versicolor")

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, 4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions/len*100, "%")

from sklearn import tree

print("Decision Tree:")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)

tree.plot_tree(clf)
plt.savefig("decisionTree.png")

y_pred = clf.predict(test_inputs)
acc = clf.score(test_inputs, test_classes)
print(acc)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_classes, y_pred))

from sklearn.neighbors import KNeighborsClassifier

print("3NN:")
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn = knn.fit(train_inputs, train_classes)

y_pred = knn.predict(test_inputs)
acc = knn.score(test_inputs, test_classes)
print(acc)

print(confusion_matrix(test_classes, y_pred))

print("5NN:")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn = knn.fit(train_inputs, train_classes)

y_pred = knn.predict(test_inputs)
acc = knn.score(test_inputs, test_classes)
print(acc)

print(confusion_matrix(test_classes, y_pred))

print("11NN:")
knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn = knn.fit(train_inputs, train_classes)

y_pred = knn.predict(test_inputs)
acc = knn.score(test_inputs, test_classes)
print(acc)

print(confusion_matrix(test_classes, y_pred))

from sklearn.naive_bayes import GaussianNB

print("GNB:")
gnb = GaussianNB()
gnb = gnb.fit(train_inputs, train_classes)

y_pred = gnb.predict(test_inputs)
acc = gnb.score(test_inputs, test_classes)
print(acc)

print(confusion_matrix(test_classes, y_pred))
