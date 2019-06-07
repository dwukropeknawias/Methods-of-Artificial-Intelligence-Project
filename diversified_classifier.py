# Estimators
# Others
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler


class DiversifiedClassifier:

    def __init__(self, estimator, n_splits):
        self.estimator = estimator
        self.n_splits = n_splits

    def predict(self, data):
        X = data[:, :-1]
        y = data[:, -1]
        # X = self.pre_process_data(X)

        prediction_vectors, p_scores = self.estimate_individual_predictions(X, y)
        print(len(prediction_vectors))
        max_size = self.get_max_size(prediction_vectors)
        decision_data_sets = self.accumulate_predictions(prediction_vectors, p_scores, max_size)
        results = self.calculate_final_decisions(decision_data_sets)
        print(

        )
        print(len(y))
        return precision_score(y, results)

    def estimate_individual_predictions(self, X, y):
        folds = KFold(n_splits=self.n_splits)
        prediction_vectors = []
        p_scores = []

        for train_index, test_index in folds.split(X):  # first folding
            X_test, y_test = X[test_index], y[test_index]

            for train_index2, test_index2 in folds.split(X_test):  # second folding
                X_train2, y_train2 = X[train_index2], y[train_index2]
                X_test2, y_test2 = X[test_index2], y[test_index2]

                self.estimator.fit(X_train2, y_train2)
                y_pred = self.estimator.predict(X_test2)
                prediction_vectors.append(y_pred)
                score = precision_score(y_test2, y_pred, average='micro')
                p_scores.append(score)

        return prediction_vectors, p_scores

    def get_max_size(self, data):
        max = 0
        for vector in data:
            if max < len(vector):
                max = len(vector)
        return max

    def pre_process_data(self, data):
        return PCA().fit_transform(StandardScaler().fit_transform(data))

    def accumulate_predictions(self, prediction_vectors, p_scores, max_size):
        decision_data_sets = []

        for i in range(max_size):
            decision_data_sets.append(DecisionDataSet())

            for k, vector in enumerate(prediction_vectors):
                if len(vector) > 0:
                    prediction, vector = vector[0], vector[1:]  # pop first element
                    if prediction in decision_data_sets[i].class_occurrence:  # check if class is in class occurrence
                        decision_data_sets[i].class_occurrence[prediction] += 1
                    else:
                        decision_data_sets[i].class_occurrence[prediction] = 1
                    if prediction in decision_data_sets[i].class_precision:  # check if class is in class precision
                        decision_data_sets[i].class_precision[prediction] += p_scores[k]
                    else:
                        decision_data_sets[i].class_precision[prediction] = p_scores[k]

        return decision_data_sets

    def calculate_final_decisions(self, decision_data_sets):
        results = []

        for dds in decision_data_sets:
            classes = []
            max_value = 0

            for key in dds.class_occurrence.keys():
                if (dds.class_occurrence[key] * dds.class_precision[key]) > max_value:
                    classes.clear()
                    classes.append(key)
                    max_value = (dds.class_occurrence[key] * dds.class_precision[key])
                elif (dds.class_occurrence[key] * dds.class_precision[key]) == max:
                    classes.append(key)

            results.append(classes)

        return results


class DecisionDataSet:
    def __init__(self):
        self.class_occurrence = {}
        self.class_precision = {}
