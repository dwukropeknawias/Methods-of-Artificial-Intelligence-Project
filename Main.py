# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from diversified_classifier import DiversifiedClassifier

for i in range(1, 11):
    data = genfromtxt('./data/classification_%d.csv' % i, delimiter=",")

    d_clf = DiversifiedClassifier()
    k_clf = KNeighborsClassifier()
    preds = d_clf.predict(data, k_clf, 3)
    print(preds)
