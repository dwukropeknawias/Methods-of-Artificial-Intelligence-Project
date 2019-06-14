# Estimators
# Others
from numpy import genfromtxt
from numpy import concatenate
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

    std_preds = []
    div_preds = []
    for train_index, test_index in folds.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        div_clf.fit(X_train, y_train)
        div_preds.append( div_clf.predict(X_test))


        std_clf.fit(X_train, y_train)
        std_preds.append( std_clf.predict(X_test))

    std_pred = concatenate(std_preds, axis=0)
    div_pred = concatenate(div_preds, axis=0)

    div_score = precision_score(y, div_pred, average='micro')
    std_score = precision_score(y, std_pred, average='micro')

    print(f'Dataset: {i}. \n STD precisions score:{std_score} \n DIV Precision Score:{div_score}\n')