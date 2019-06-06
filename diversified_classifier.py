# Estimators
# Others
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DiversifiedClassifier:
    def predict(self, data, clf, n_splits):
        X = data[:, :-1]
        y = data[:, -1]
        # X = self.pre_process_data(X)
        folds = KFold(n_splits=n_splits, random_state=4)  # fair dice roll
        scores = []
        pred_vectors = []

        for train_index, test_index in folds.split(X):
            X_test, y_test = X[test_index], y[test_index]

            for train_index2, test_index2 in folds.split(X_test):
                X_train2, y_train2 = X[train_index2], y[train_index2]
                X_test2, y_test2 = X[test_index2], y[test_index2]
                clf.fit(X_train2, y_train2)
                y_pred = clf.predict(X_test2)
                pred_vectors.append(y_pred)
                score = precision_score(y_test2, y_pred, average='micro')
                scores.append(score)
        decisions = []
        max_size = 0
        for vector in pred_vectors:
            if max_size < len(vector):
                max_size = len(vector)

        for i in range(max_size):
            decisions.append(DecisionData())
            for k, vector in enumerate(pred_vectors):
                if len(vector) > 0:
                    pred, vector = vector[0], vector[1:]
                    if pred in decisions[i].class_occurrence:
                        decisions[i].class_occurrence[pred] += 1
                    else:
                        decisions[i].class_occurrence[pred] = 1
                    if pred in decisions[i].class_precision:
                        decisions[i].class_precision[pred] += scores[k]
                    else:
                        decisions[i].class_precision[pred] = scores[k]
        results = []

        for dec in decisions:
            classes = []
            max_value = 0
            for key in dec.class_occurrence.keys():
                if (dec.class_occurrence[key] * dec.class_precision[key]) > max_value:
                    classes.clear()
                    classes.append(key)
                    max_value = (dec.class_occurrence[key] * dec.class_precision[key])
                elif (dec.class_occurrence[key] * dec.class_precision[key]) == max:
                    classes.append(key)
            results.append(classes)

        return results;

    def pre_process_data(self, data):
        return PCA().fit_transform(StandardScaler().fit_transform(data))


class DecisionData:
    def __init__(self):
        self.class_occurrence = {}
        self.class_precision = {}
