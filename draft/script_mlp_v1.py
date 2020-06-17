#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:08:39 2020

@author: datascience
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

pd.get_option("display.max_rows")
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)


data = pd.read_csv('diabetes.csv') 
data.describe().transpose()
print(data.shape)

# Data transformations
target_column = ['Outcome'] 
predictors = list(set(list(data.columns))-set(target_column))
data[predictors] = data[predictors]/data[predictors].max()
data.describe().transpose()


#X = data.data[:, (2,3)]
#y = (data.target==0).astype(np.int8)
X = data[predictors].values
y = data[target_column].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

model = Pipeline(steps=[
        ('one-hot encoder', OneHotEncoder()),
        ('mlp', MLPClassifier(solver='adam', alpha=1e-5, max_iter=500,
                              hidden_layer_sizes=(5, 8), random_state=1))
])

model.fit(X_train, y_train)

predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))


print("weights between input and first hidden layer:")
print(mlp.coefs_[0])
print("\nweights between first hidden and second hidden layer:")
print(mlp.coefs_[1])

print("Bias values for first hidden layer:")
print(mlp.intercepts_[0])
print("\nBias values for second hidden layer:")
print(mlp.intercepts_[1])