from sklearn import tree 

import numpy as np

from data_loader import load_txt
from metrics import get_performance_measure, print_metric_score

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for iter in range(self.n_trees):
            # choose a training set by choosing N times with replacement from training set 
            # (N is the number of training example)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            dtree = tree.DecisionTreeClassifier(random_state=0, 
                                               max_depth=self.max_depth,
                                               min_samples_split=self.min_samples_split,
                                               max_features=self.max_features)
            
            dtree.fit(X_sample, y_sample)
            self.trees.append(dtree)


    def predict(self, X, probability=False):
        if probability:
            # print('pred proba')
            tree_predictions_proba = []
            for dtree in self.trees:
                tree_predictions_proba.append(dtree.predict_proba(X)[:, 1])
            sample_predictions_proba = np.array(tree_predictions_proba).transpose().tolist()

            predictions_proba = []
            for preds in sample_predictions_proba:
                # print(np.array(preds).shape)
                avg_proba = sum(preds) / len(preds) 
                predictions_proba.append(avg_proba)
            predictions_proba = np.array(predictions_proba)
            return predictions_proba
        
        else:
            # print('pred')
            tree_predictions = []
            for dtree in self.trees:
                tree_predictions.append(dtree.predict(X))
            sample_predictions = np.array(tree_predictions).transpose().tolist()
            # print(np.array(tree_predictions).shape)
            # print(np.array(sample_predictions).shape)

            # use majority voting among all trees
            predictions = []
            for preds in sample_predictions:
                most_frequent_label = max(preds, key=preds.count)
                predictions.append(most_frequent_label)
            predictions = np.array(predictions)
            return predictions
        
# X_train, X_test, y_train, y_test = load_txt(1) 

# clf = RandomForest()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict(X_test, probability=True)
# # y_pred_proba = clf.predict_proba(X_test)

# accuracy, precison, recall, f1, auc = get_performance_measure(y_test, y_pred, y_pred_proba)
# print_metric_score('random forest', accuracy, precison, recall, f1, auc)