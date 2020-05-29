#!/usr/bin/env python
# coding: utf-8

# In[62]:


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
import pandas as pd

import category_encoders as ce
train_data = pd.read_csv('trainbigmart.csv')

class BigMart:
    def __init__(self, data):
        """ initializes the class with the data attribute"""
        self.data = data
        self.models = {}
        
    def clean_data(self):
        """ 
        cleans the data and removes missing
        values
        
        """
        columns = self.data.columns
        null_cols = [col for col in columns if self.data[col].isnull().any()]
        num_null_cols= [col for col in null_cols if self.data[col].dtype in ['float', 'int']]
        cat_null_cols = list(set(null_cols) - set(num_null_cols))
        for col in num_null_cols:
            self.data[col] = self.data[col].fillna(self.data[col].mean())
        for col in cat_null_cols:
            self.data[col] = self.data[col].fillna('small')
        
    def preprocess_data(self):
        """
        preprocesses the data to better fit model
        """
        column = ['Item_Fat_Content', 'Item_Type', 
              'Item_MRP', 'Outlet_Identifier', 'Item_Outlet_Sales']
        train_pro = self.data.loc[:,column]
        enc = ce.OneHotEncoder()
        cat_cols = [col for col in train_pro.columns if train_pro[col].dtype == 'object' and col != 'Outlet_Identifier']
        enc.fit(train_pro[cat_cols])

        train_clean = train_pro.join(enc.transform(train_pro[cat_cols]))

        train_clean = train_clean.drop(cat_cols, axis = 1)
        id_list =list(train_clean.Outlet_Identifier.unique())
        X_train = [train_clean[train_clean['Outlet_Identifier']==id_].drop(['Outlet_Identifier', 'Item_Outlet_Sales'], axis = 1) for id_ in id_list]
        y_train = [train_clean[train_clean['Outlet_Identifier']==id_]['Item_Outlet_Sales'] for id_ in id_list]
        train = [X_train, y_train]
        self.data = train
        
    def train_model(self):
        """ trains the preprocessed data with algorithm """
        
        for i in range(10):
            model = LassoCV(n_alphas = 100, cv = 3, max_iter= 1000,  random_state = 0)
            xtrain =self.data[0]
            ytrain =self.data[1]
            model.fit(xtrain[i],ytrain[i])
            self.models[i] = model

            
    def predict(self, data):
        """ 
        gets the predicted value of
        a dataframe passed to it as an argument
        """
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
        y_pred = []
        index = []
        for i in range(10):
            test_model = X_test[i]
            index.append(test_model.index)
            pred = self.models[i].predict(test_model)
            y_pred.append(pred)
        sub = [item for sublist in y_pred for item in sublist]
        indices = [item1 for sublist in index for item1 in sublist] 
        predictions =pd.DataFrame(sub, columns = ['Item_Outlet_Sales'], index = indices)
        predictions = predictions.sort_index()
        return predictions



# In[ ]:




