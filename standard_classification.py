# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





for i in range (1, 11):
    data = genfromtxt('./data/classification_%d.csv' %i, delimiter=",")
    X = data [:,:-1]
    y = data [:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

    classifiers = [KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier()]

    print("Score from the file number %d " %i)

    for j in classifiers:
        clf = j
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("Accuracy of " + j.__class__.__name__ + " = %.3f " %score)

    print("\n")

