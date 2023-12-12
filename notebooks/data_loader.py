import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def preprocessing(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            labelencoder = LabelEncoder()
            df[col] = labelencoder.fit_transform(df[col])

    # print(df.info())
    # normalizing
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)

    return df
            

def load_txt(file_no):
    print('dataset ' + str(file_no))
    path_name = '/u/epw9kz/academic/first_year/ML/project/'
    
    if file_no == 1:
        file = path_name + 'dataset/project3_dataset1.txt'
    elif file_no == 2:
        file = path_name + 'dataset/project3_dataset2.txt'
    else:
        pass
        
    df = pd.read_csv(file, sep='\t', header=None)

    df = preprocessing(df)

    X = df[:, :-1]
    y = df[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    return X_train, X_test, y_train, y_test 

# X_train, X_test, y_train, y_test = load_txt(1) 
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    