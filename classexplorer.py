import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = [[181, 80, 44], [177, 70, 43],
     [160, 60, 38], [154, 54, 37],
     [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37],
     [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Classifiers
clf = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier()
clf3 = svm.SVC()
clf4 = RandomForestClassifier()

#Training data fit
clf = clf.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

#New Data Fit
pred_new_data_tree = clf.predict([[190, 70, 43]])
print(pred_new_data_tree)

pred_new_data_neighbors = clf2.predict([[190, 70, 43]])
print(pred_new_data_neighbors)

pred_new_data_svm = clf3.predict([[190, 70, 43]])
print(pred_new_data_svm)

pred_new_data_forest = clf4.predict([[190, 70, 43]])
print(pred_new_data_forest)

#Dataset accuracy
prediction_tree = clf.predict(X)
acc_tree = accuracy_score(Y, prediction_tree)
print('{}'.format(acc_tree))

prediction_neighbors = clf2.predict(X)
acc_neighbors = accuracy_score(Y, prediction_neighbors)
print('{}'.format(acc_neighbors))

prediction_svm = clf3.predict(X)
acc_svm = accuracy_score(Y, prediction_svm)
print('{}'.format(acc_svm))


prediction_forest = clf4.predict(X)
acc_forest = accuracy_score(Y, prediction_forest)
print('{}'.format(acc_forest))


#Best Classifier

index = np.argmax([acc_neighbors, acc_forest, acc_svm])
classifiers = {0: 'Neighbors', 1: 'forest', 2: 'svm'}
print('{}'.format(classifiers[index]))
