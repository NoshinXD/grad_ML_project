import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_txt(file_no):
    path_name = '/u/epw9kz/academic/first_year/ML/project/'
    
    if file_no == 1:
        file = path_name + 'dataset/project3_dataset1.txt'
    elif file_no == 2:
        file = path_name + 'dataset/project3_dataset2.txt'
    else:
        pass
        
    df = pd.read_csv(file, sep='\t', header=None)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    return X_train, X_test, y_train, y_test 

# X_train, X_test, y_train, y_test = load_txt(1) 
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    