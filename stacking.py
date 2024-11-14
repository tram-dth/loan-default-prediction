# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:02:14 2024

@author: dthtr
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor, 
                              AdaBoostRegressor)
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBRegressor
pd.set_option("display.precision", 3)


#my own module
import myFuncs as mf


def train_stacking_model(x_train, y_train, over_under_sampling):
    
    # Initialize the base models
    base_models = [
        XGBRegressor(n_estimators= ),
        KNeighborsClassifier(n_neighbors= )
        ]

    # Initialize the meta model
    meta_model = LogisticRegression()

    # Train the base models
    base_model_predictions = []
    for base_model in base_models:
        base_model.fit(x_train, y_train)
        base_model_predictions.append(base_model.predict(x_train))

    # Stack the predictions
    stacked_predictions = np.column_stack(base_model_predictions)

    # Train the meta model
    meta_model.fit(stacked_predictions, y_train)

    return base_models, meta_model




def stacking_predict(base_models, meta_model, x):

    # Generate predictions from base models
    base_model_predictions = []
    for base_model in base_models:
        base_model_predictions.append(base_model.predict(x))

    # Stack the predictions
    stacked_predictions = np.column_stack(base_model_predictions)

    # Generate predictions from meta model
    y_pred = meta_model.predict(stacked_predictions)
    y_pred_p = meta_model.predict_proba(x_test)[:, -1]

    return y_pred, y_pred_p



def trials(base_models, meta_model, x_train, y_train, x_test, y_test):
    
    #generating prediction for both train and test set  
    y_train_pred, y_train_pred_p = stacking_predict(base_models, meta_model, x_train)
    y_test_pred, y_test_pred_p = stacking_predict(base_models, meta_model, x_test)
    
    #evaluate performance
    results_train = mf.classification_performance(y_train, y_train_pred, all_indicators=True)
    results_test = mf.classification_performance(y_test, y_test_pred, all_indicators=True)
    
    
    #draw the test prediction's ROC
    mf.draw_roc(y_test, y_test_pred)
    
    return results_train, results_test
    
    
    
    
def running(base_models, meta_model, x_train, y_train, x_test, y_test, max_exprun, over_under_sampling):
    auc_all_train = [0]*max_expruns
    acc_all_train = [0]*max_expruns
    f1_all_train = [0]*max_expruns
    precision_all_train = [0]*max_expruns
    recall_all_train = [0]*max_expruns
   
    auc_all_test = [0]*max_expruns
    acc_all_test = [0]*max_expruns
    f1_all_test = [0]*max_expruns
    precision_all_test = [0]*max_expruns
    recall_all_test = [0]*max_expruns
    
    
    for run_num in range(0, max_expruns): 
        x_train, x_test, y_train, y_test = mf.data_split_random(df, x_vars, y_vars, run_num)
        
        if over_under_sampling == True:
            resampler = mf.over_under_sampler()
            x_train, y_train = resampler.fit_resample(x_train, y_train)
            
        train_stacking_model(x_train, y_train, over_under_sampling)
            
        results_train, result_test = trials(base_models, 
                                            meta_model, 
                                            x_train, 
                                            y_train.values.ravel(), 
                                            x_test, 
                                            y_test.values.ravel()) 
        
     auc_all_train[run_num] = result_train[0]
     acc_all_train[run_num] = result_train[1]
     f1_all_train[run_num] = result_train[2]
     precision_all_train[run_num] = result_train[3] 
     recall_all_train[run_num] = result_train[4]
     
     auc_all_test[run_num] = result_test[0]
     acc_all_test[run_num] = result_test[1]
     f1_all_test[run_num] = result_test[2]
     precision_all_test[run_num] = result_test[3]
     recall_all_test[run_num] = result_test[4]
     
     train = (auc_all_train, acc_all_train, f1_all_train, precision_all_train, recall_all_train)
     test = (auc_all_test, acc_all_test, f1_all_test, precision_all_test, recall_all_test) 
     
     return  train, test   
 
    
 def print_results(results, file_name, no_indi = 3):
     cols = ['pos_weight',
             'mean_auc', 'sd_auc', 
             'mean_acc', 'sd_acc', 
             'mean_f1_score', 'sd_f1_score',
             'mean_precision', 'sd_precision',
             'mean_recall', 'sd_recall']
     indis = ['auc', 'acc', 'f1_score', 'precision', 'recall']
     
     L = []
     for k in results.keys():
         row = [k]
         print('\n\nStacking model performance')
         for i in range(0, no_indi):
             mean_indicator = np.mean(results[k][i]).round(4)
             sd_indicator = np.std(results[k][i], axis = 0).round(2)
             row.append(mean_indicator)
             row.append(sd_indicator)
             print(f'the mean {indis[i]}  =  {mean_indicator}')
             print(f'the sd {indis[i]}  =  {sd_indicator}')
                
         L.append(row)
     L = pd.DataFrame(L, columns= cols[: (1+no_indi*2)])
     
     #print to csv
     L.to_csv(file_name, sep = ',', header=True)
    
