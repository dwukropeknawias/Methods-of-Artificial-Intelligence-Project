# Estimators
# Others
from numpy import genfromtxt
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from diversified_classifier import DiversifiedClassifier

n_splits = 5  # configure n of folds
div_clf = DiversifiedClassifier(KNeighborsClassifier(), n_splits)
std_clf = KNeighborsClassifier()
for i in range(1, 11):
    data = genfromtxt('./data/classification_%d.csv' % i, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    folds = KFold(n_splits=n_splits)

    for train_index, test_index in folds.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        div_clf.fit(X_train, y_train)
        div_pred = div_clf.predict(X_test)
        div_score = precision_score(y_test, div_pred, average='micro')

        std_clf.fit(X_train, y_train)
        std_pred = std_clf.predict(X_test)
        std_score = precision_score(y_test, std_pred, average='micro')

        #tutaj porownanie jakie nam bedzie potrzebne
        if std_score > div_score:
            print('STD!')
        else:
            print('DIV!')
