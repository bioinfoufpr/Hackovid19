#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:08:39 2020

@author: datascience
"""
#import matplotlib.pyplot as plt
#import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

#from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score

#from sklite import LazyExport
from joblib import dump

#def multi2_binclass(y, target=1):
#    new_y = y.copy()
#    orig_target = y==target
#    new_y[new_y!=target] = 0
#    new_y[orig_target] = 1
#    return new_y'macro'
#/home/datascience/Documents/IA_Wonders/Trekkers/draft/output/RxCov10_model_01_07_2020_09_59_10.joblib
#def mult2_binmat(y):
#    y_mult = np.zeros(shape=(len(np.unique(y)), len(y)), dtype='uint8')
#    for i in y:
#        new_y = y.copy()
#        orig_target = y==i
#        new_y[new_y!=i] = 0
#        new_y[orig_target] = 1
#        y_mult[i,:] = new_y
#    return y_mult

#target = 0
#y_bin = multi2_binclass(y, target)

def generate_model(X, y, prefix, param):
    ''' Runs MLP with softmax output activation method, which allows multiclass
    classification. We will need to test All VS One to check performance.'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    print(X_train.shape); print(X_test.shape)
    
    ppl_model = Pipeline(steps=[
            ('one-hot encoder', OneHotEncoder()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(5,), activation='relu',
                                  solver='lbfgs', alpha=1e-5, max_iter=500,
                                  random_state=1, verbose=True)) ])
    ppl_model[1].out_activation_ = 'softmax'
    
    ppl_model.fit(X_train, y_train)
    
    predict_train = ppl_model.predict(X_train)
    predict_test = ppl_model.predict(X_test)
    
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    
    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))
    
#    print("weights between input and first hidden layer:")
#    print(ppl_model1[1].coefs_[0])
#    print("\nweights between first hidden and second hidden layer:")
#    print(ppl_model1[1].coefs_[1])
#    
#    print("Bias values for first hidden layer:")
#    print(ppl_model1[1].intercepts_[0])
#    print("\nBias values for second hidden layer:")
#    print(ppl_model1[1].intercepts_[1])
    
    #sklite_file = "draft/mlp_sweep_model.json"
    #lazy = LazyExport(ppl_model1)
    nameout = 'output/'+prefix+'_model'+param.curr_datetime+'.joblib'
    dump(ppl_model[1], nameout)

models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(),
    'MLP': MLPClassifier()
}
models1 = {
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
}

params1 = {
        'RandomForestClassifier': { 'n_estimators': [4, 8] },
        'AdaBoostClassifier':  { 'n_estimators': [4, 8] },
        'GradientBoostingClassifier': { 'n_estimators': [4, 8], 'learning_rate': [0.8, 1.0] },
        'SVC': [
                {'kernel': ['linear'], 'C': [1, 10]},
                {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
                ],
        'MLP': 
            { 'hidden_layer_sizes': [(5,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['lbfgs', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive']
                    }
}

params1 = {
        'RandomForestClassifier': { 'n_estimators': [4, 8] },
        'AdaBoostClassifier':  { 'n_estimators': [4, 8] },
}
            
import EstimatorSwitcher as ESw

pd.options.display.max_columns = 22

helper1 = ESw.EstimatorSwitcher(models1, params1)
helper1.fit(X, y, n_jobs=2)
df = helper1.score_summary()
