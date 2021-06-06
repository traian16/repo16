# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:19:34 2020

@author: traian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import rcParams
from matplotlib.cm import rainbow
from matplotlib import pyplot
'exec(%matplotlib inline)'
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import all the Machine Learning algorithms 
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('D:\MACHINE LEARNING\Heart Disease Prediction Project\heart.csv')
dataset.info()
dataset.describe()
dataset.head()
rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
#plt.imshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
dataset.hist()
pyplot.show()
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'blue'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
#
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
knn_scores = []
pyplot.show()
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {}% with {} neighbors.".format(knn_scores[7]*100, 8))
###Training and Predictions
from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5)
#classifier = KNeighborsClassifier(n_neighbors=8)
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
#
y_pred = classifier.predict(X_test)
print(y_pred)
#
print(X_test)
print(y_test)
print(X_train)
print(y_train)

# test in production
#dataset.info()
#dataset.describe()
#dataset.hist()
print(dataset)

#target=0
#x = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
x = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 0, 1, 2, 1, 0, 1, 2, 1, 1, 0, 0, 1, 2, 0, 1, 0, 1]]

print(x)
y_pred=classifier.predict(x)
print(y_pred)

x = [[60, 1, 0, 130, 306, 0, 0, 332, 1, 4.4, 1, 20, 30, 100, 100, 200, 100, 0, 10, 20, 100, 100, 0, 0, 10, 20, 0, 100, 0, 10]]
y_pred=classifier.predict(x)
print(y_pred)

x = [[1.062485, 2.191778, -0.372287,  0.234095, 0, 0, 0, 0, -1, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]]
y_pred=classifier.predict(x)
print(y_pred)

# target=1
x = [[0.290464, 0.478391, -0.101730, -1.165281, 0, 0, 0, 0, -1, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
y_pred=classifier.predict(x)
print(y_pred)

x = [[-0.812425, -0.092738,  0.130176,  1.283627, 0, 0, 0, 0, -1, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
y_pred=classifier.predict(x)
print(y_pred)

print(X_train)
print(X_test)
##
print(60, 1, 0, 130, 306, 0, 0, 332)
#
import numpy as np
#z = standardScaler.fit_transform([60, 1, 0, 130, 306, 0, 0, 332])

#a = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
#b = a.reshape(1,-1)

#a = np.matrix([[60, 1, 0, 130, 306, 0, 0, 332]])
a = np.matrix([[60, 1, 0, 130, 306, 0, 0, 332, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
b = a.reshape(1,-1)
print(b)
z = standardScaler.fit_transform(b)
print(z)
y_pred=classifier.predict(z)
print(y_pred)
