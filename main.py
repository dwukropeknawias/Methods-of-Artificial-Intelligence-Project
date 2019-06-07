# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from diversified_classifier import DiversifiedClassifier
from standard_classifier import StandardClassifier


for i in range(1, 11):
    data = genfromtxt('./data/classification_%d.csv' % i, delimiter=",")

    std_clf = StandardClassifier(KNeighborsClassifier())
    div_clf = DiversifiedClassifier(KNeighborsClassifier(), 3)

    print(std_clf.predict(data))
    print(div_clf.predict(data))



