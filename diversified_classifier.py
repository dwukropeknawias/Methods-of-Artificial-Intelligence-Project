# Estimators
# Others
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler


class DiversifiedClassifier:
    classifiers_group = []
    precision_scores = []

    def __init__(self, estimator, n_splits):
        self.estimator = estimator.__class__ #store class to later create new instances of this estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        self.classifiers_group.clear()
        self.precision_scores.clear()
        folds = KFold(n_splits=self.n_splits)

        for train_index, test_index in folds.split(X): #for each fold create new clf, train it, append it and its precision score
            clf = self.estimator()
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            clf.fit(X_train, y_train)
            y_prediction = clf.predict(X_test)
            score = precision_score(y_test, y_prediction, average='micro')

            self.classifiers_group.append(clf)
            self.precision_scores.append(score)

    def predict(self, X):
        prediction_vectors = self.estimate_individual_predictions(X)
        max_size = self.get_max_size(prediction_vectors)
        decision_data_sets = self.accumulate_predictions(prediction_vectors, self.precision_scores, max_size)
        results = self.calculate_final_decisions(decision_data_sets)
        return results

    def estimate_individual_predictions(self, X):
        prediction_vectors = []

        for clf in self.classifiers_group:
            y_prediction = clf.predict(X)
            prediction_vectors.append(y_prediction)

        return prediction_vectors

    def get_max_size(self, data): # get maxsize (because data not always splits evenly)
        max = 0
        for vector in data:
            if max < len(vector):
                max = len(vector)
        return max

    def accumulate_predictions(self, prediction_vectors, p_scores, max_size):
        decision_data_sets = []

        for i in range(max_size): #for each element in vectors
            decision_data_sets.append(DecisionDataSet())

            for k, vector in enumerate(prediction_vectors): #for each vector
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
            max_value = 0 #the biggest class occurence * class precision score

            for key in dds.class_occurrence.keys(): #for each class in the voting
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
