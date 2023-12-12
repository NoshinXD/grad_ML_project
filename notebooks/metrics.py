import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_K_fold_cv(clf, X_train, y_train):
    kf = KFold(n_splits=10)  # 10-fold cross-validation

    # accs = []
    # precs = []
    # recs = []
    # f1s = []
    # aucs = []

    # best_acc = -999999
    # best_prec = -999999
    # best_rec = -999999
    # best_f1 = -999999
    # best_auc = -999999

    # best_clf = 0 

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        clf.fit(X_train_fold, y_train_fold)  # Train the model on the training fold
        y_pred = clf.predict(X_val_fold)  # Predict on the validation fold
        # y_pred_proba = clf.predict_proba(X_val_fold)

    #     accuracy, precison, recall, f1, auc = get_performance_measure(y_val_fold, y_pred, y_pred_proba)
    #     accs.append(accuracy)
    #     # precs.append(precison)
    #     # recs.append(recall)
    #     # f1s.append(f1)
    #     # aucs.append(auc)
    
    # average_score = np.mean(accs)
    return clf

def get_performance_measure(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_proba)

    return accuracy, precison, recall, f1, roc

def print_metric_score(model_name, accuracy, precison, recall, f1, auc):
    print(model_name)
    print('accuracy:', accuracy)
    print('precison:', precison)
    print('recall:', recall)
    print('f1:', f1)
    print('auc:', auc)