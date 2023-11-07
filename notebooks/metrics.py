from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_performance_measure(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    return accuracy, precison, recall, f1, auc

def print_metric_score(model_name, accuracy, precison, recall, f1, auc):
    print(model_name)
    print('accuracy:', accuracy)
    print('precison:', precison)
    print('recall:', recall)
    print('f1:', f1)
    print('auc:', auc)