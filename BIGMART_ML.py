#!/usr/bin/env python
# coding: utf-8

# In[130]:


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox
import category_encoders as ce
train_data = pd.read_csv('trainbigmart.csv')



def clean_data(data):
    columns = data.columns
    null_cols = [col for col in columns if data[col].isnull().any()]
    num_null_cols= [col for col in null_cols if data[col].dtype in ['float', 'int']]
    cat_null_cols = list(set(null_cols) - set(num_null_cols))
    for col in num_null_cols:
        data[col] = data[col].fillna(data[col].mean())
    for col in cat_null_cols:
        data[col] = data[col].fillna('small')
    return data
    
        
def preprocess_test(data):
    column = ['Item_Fat_Content', 'Item_Type', 
              'Item_MRP', 'Outlet_Identifier',]
    test_pro = data.loc[:,column]
    enc = ce.OneHotEncoder()
    cat_cols = [col for col in test_pro.columns if test_pro[col].dtype == 'object' and col != 'Outlet_Identifier']
    enc.fit(test_pro[cat_cols])

    test_clean = test_pro.join(enc.transform(test_pro[cat_cols]))

    test_clean = test_clean.drop(cat_cols, axis = 1)
    id_list =list(test_clean.Outlet_Identifier.unique())
    X_test = [test_clean[test_clean['Outlet_Identifier']==id_].drop('Outlet_Identifier', axis = 1) for id_ in id_list]
    return X_test

def preprocess_train(data):
    column = ['Item_Fat_Content', 'Item_Type', 
              'Item_MRP', 'Outlet_Identifier', 'Item_Outlet_Sales']
    train_pro = data.loc[:,column]
    enc = ce.OneHotEncoder()
    cat_cols = [col for col in train_pro.columns if train_pro[col].dtype == 'object' and col != 'Outlet_Identifier']
    enc.fit(train_pro[cat_cols])

    train_clean = train_pro.join(enc.transform(train_pro[cat_cols]))

    train_clean = train_clean.drop(cat_cols, axis = 1)
    id_list =list(train_clean.Outlet_Identifier.unique())
    X_train = [train_clean[train_clean['Outlet_Identifier']==id_].drop(['Outlet_Identifier', 'Item_Outlet_Sales'], axis = 1) for id_ in id_list]
    y_train = [train_clean[train_clean['Outlet_Identifier']==id_]['Item_Outlet_Sales'] for id_ in id_list]
    train = [X_train, y_train]
    return train

class BigMartModel():

    def fit_predict(self, X):
        y_pred = []
        index = []
        for i in range(10):
            train = preprocess_train(clean_data(train_data))
            xtrain =train[0]
            ytrain =train[1]
            model = LassoCV(n_alphas = 100, cv = 3, max_iter= 1000,  random_state = 0)
            model.fit(xtrain[i],ytrain[i])
            test_model = X[i]
            index.append(test_model.index)
            pred = model.predict(test_model)
            y_pred.append(pred)
        sub = [item for sublist in y_pred for item in sublist]
        indices = [item1 for sublist in index for item1 in sublist] 
        predictions =pd.DataFrame(sub, columns = ['Item_Outlet_Sales'], index = indices)
        predictions = predictions.sort_index()
        return predictions


# In[ ]:




