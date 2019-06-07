# Estimators
# Others
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score


class StandardClassifier:
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        print(len(X))
        # X = self.pre_process_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=4)  # fair dice roll
        self.estimator.fit(X_train, y_train)
        y_prediction = self.estimator.predict(X_test)
        p_score = precision_score(y_test, y_prediction, average='micro')
        return p_score

    def pre_process_data(self, data):
        return PCA().fit_transform(StandardScaler().fit_transform(data))
