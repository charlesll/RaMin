# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:12:53 2017

@author: charles
"""

import pickle
import numpy as np

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier

#
# Data importation
#

X = np.load('../data/excellent_unoriented/obs.npy')
y = pickle.load( open( "../data/excellent_unoriented/labels.pkl", "rb" ) )


#
# TRain/Test split
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

#
# Initiate classifiers
#

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(max_depth=15),
    RandomForestClassifier(max_depth=15, n_estimators=5, max_features=2),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('\n- Classifier:'+name+' is scoring '+str(score)+'.')

#
# Keeping the algo that outperforms the other without tweeking
#

print('#########################################################')
print("Now getting the best classifiers and training them again")
print('#########################################################')

names_good = ["Nearest Neighbors", "Linear SVM", "Neural Net", "Naive Bayes"]

classifiers_good = [
    KNeighborsClassifier(6),
    SVC(kernel="linear", C=0.02),
    MLPClassifier(alpha=0.1,hidden_layer_sizes=600),
    GaussianNB()]

# iterate over good classifiers
for name, clf in zip(names_good, classifiers_good):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('\n- Classifier:'+name+' is scoring '+str(score)+'.\n')
