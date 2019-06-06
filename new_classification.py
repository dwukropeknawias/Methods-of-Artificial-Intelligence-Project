# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from numpy import mean,std



for i in range (1, 11):
    data = genfromtxt('./data/classification_%d.csv' %i, delimiter=",")
    X = data [:,:-1]
    y = data [:,-1]

    scaler = StandardScaler()
    Xscaler = scaler.fit_transform(X)
    pca = PCA()
    Xpca = pca.fit_transform(Xscaler)
    kf = KFold(n_splits=5)
    accuracies = []


    classifiers = [KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier()]

    print("Score from the file number %d " %i)

    for j in classifiers:
        clf = j
        index=1
        for train, test in kf.split(X):
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print("Fold number %.d - accuracy of " %index + str(clf.__class__.__name__) + " = %.3f" % score, sep="")
            accuracies.append(score)
            index+=1
        print("Mean accuracy: " + str(mean(accuracies)) + ". Standard deviation: %.4f " % std(accuracies))
        print("\n")

