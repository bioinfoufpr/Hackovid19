#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:08:39 2020

@author: datascience
"""
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score

class EstimatorSwitcher:
    '''This code is based on the suggested by David Batista in this post:
    http://www.davidsbatista.net/blog/2018/02/23/model_optimization/.
    I added more validators and the possibility of run it without GridSearch and
    cross-validation. In my experience we don't need these expensive resources 
    when we are focused in the pre-processing step.
    '''
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.set_scores()
   
    def set_scores(self):
        f1 = make_scorer(f1_score , average='micro')
        accuracy = make_scorer(accuracy_score)
        self.default_scores = {'Accuracy': accuracy, 'F1_score': f1}
    

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=self.default_scores, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs
    
    def score_summary(self, sort_by = 'mean_score0'):
        def row(key, scores, params, n_scores):
            new_d = dict()
            for s in range(n_scores):
                d = {
                     'min_score{}'.format(s): min(scores),
                     'max_score{}'.format(s): max(scores),
                     'mean_score{}'.format(s): np.mean(scores)
                }
                if len(new_d)==0:
                    new_d['estimator'] = key
                    new_d.update(d)
                else:
                    new_d.update(d)
                print(new_d)
                
            return pd.Series({**params,**new_d})
        
        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                for sc in self.default_scores.keys():
                    key = "split{}_test_{}".format(i, sc)
                    r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))
        
            all_scores = np.hstack(scores)
            n_scores = len(self.default_scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p, n_scores)))
        
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        columns = ['estimator']+[i for i in df.columns if 'score' in i]
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]