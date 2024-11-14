# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:54:56 2024

@author: dthtr
"""

import pandas as pd
import numpy as np

from sklearn import set_config
set_config(transform_output = "pandas")
from sklearn.neighbors import KNeighborsClassifier


#___my own modules_______
import myFuncs as mf


def knn(x_train, x_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model using the training sets
    model.fit(x_train, y_train)
    
    # Make predictions on training and testing sets
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    
    #performance indicators on training and testing sets
    #auc, acc, f1
    perf_train = mf.classification_performance(y_train, y_pred_train, all_indicators = True)
    perf_test = mf.classification_performance(y_test, y_pred_test, all_indicators = True)
    
    return perf_train, perf_test



def trials(df, x_vars, y_vars, k, max_expruns, over_under_sampling = False): 
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
        
        x_train, x_test, y_train, y_test = mf.data_split_random(df, x_vars, y_vars, 50 - run_num)
        
        if over_under_sampling == True:
            resampler = mf.over_under_sampler()
            x_train, y_train = resampler.fit_resample(x_train, y_train)
            
        
        result_train, result_test = knn(x_train, 
                                        x_test, 
                                        y_train.values.ravel(), 
                                        y_test.values.ravel(), 
                                        k)
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



def k_experiment(df, x_vars, y_vars, k_range, max_expruns, over_under_sampling = False):
    results_train = {}
    results_test = {}
    for k in k_range:
        r_train, r_test = trials(df, x_vars, y_vars, k, max_expruns, over_under_sampling)
        results_train[k] = r_train
        results_test[k] = r_test
    
    return results_train, results_test


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
        print('\n\KNN performance at the number of neighbour k =  : ', k)
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
   