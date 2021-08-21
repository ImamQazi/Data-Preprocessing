# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:31:37 2021

@author: Imam Qazi
"""
#imoport libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Data.csv')

#one way if you have alot features
#create dependent (X) & independent variables (Y)
#x = dataset[['ENGINESIZE']]
#y = dataset[['Purchased']]

#other way if you have just two fetaure 
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

#taking care of missing data (for missing data we use the mean of whole column)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding categorical values 
#for Country column
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
x[:, 0] = labelencoder.fit_transform(x[:,0])
onehotencoder = ColumnTransformer(
    [('OHE', OneHotEncoder(),[0])],     remainder = 'passthrough'
    )
x = onehotencoder.fit_transform(x)

#for Purchase column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#spliting the data set into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

