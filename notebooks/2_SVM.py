import numpy as np
from data_loader import load_txt
from metrics import get_performance_measure, print_metric_score

import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    # print('linear_kernel')
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=3):
    return np.power(1 + np.dot(x1, x2), p)

def gaussian_kernel(x1, x2, gamma=0.02):
    return np.exp(-gamma * np.power(np.linalg.norm(x1 - x2), 2))

class SVM:
    def __init__(self, kernel='linear', C=1.0, learning_rate=1e-3, mylambda=1e-2, n_iter_=1000):
        self.kernel = 'linear'
        self.C = C
        self.learning_rate = learning_rate
        self.n_iter_ = n_iter_
        self.mylambda = mylambda
        self.w = None
        self.b = None

    def fit(self, X=None, y=None):
        # initializing weights and bias
        # print(X.shape)
        n_features = X.shape[1]
        # print('n_features', n_features)
        self.w = np.zeros(n_features)
        self.b = 0

        # changing_lable_from_0_to_minus_1
        y = np.where(y == 0, -1, 1)

        if self.kernel == 'linear':
            for iter in range(self.n_iter_):
                for i in range(len(X)):
                    wx_plus_b = linear_kernel(X[i], self.w) + self.b

                    # calculating gradients
                    if y[i] * wx_plus_b >= 1:
                        dw = self.mylambda * self.w
                        db = 0
                    else:
                        dw = self.mylambda * self.w - self.C * np.dot(y[i], X[i])
                        db = - self.C * y[i]

                    self.w = self.w - self.learning_rate * dw
                    self.b = self.b - self.learning_rate * db
        else:
            pass

    def predict(self, X,  probability=False):
        prediction_probability = np.dot(X, self.w) + self.b
        if probability:
            return prediction_probability
        else:
            prediction = np.sign(prediction_probability)
            prediction = np.where(prediction == -1, 0, 1)
            return prediction
        
X_train, X_test, y_train, y_test = load_txt(2) 

clf = SVM(n_iter_=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict(X_test, probability=True)
# y_pred_proba = clf.predict_proba(X_test)

accuracy, precison, recall, f1, auc = get_performance_measure(y_test, y_pred, y_pred_proba)
print_metric_score('SVM linear', accuracy, precison, recall, f1, auc)
        