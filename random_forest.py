# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:56:04 2024

@author: dthtr
"""


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import random

from sklearn import set_config
set_config(transform_output = "pandas")

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

import imblearn 
from imblearn.over_sampling import RandomOverSampler


import myFuncs as mf



def random_forest(run_num, x_train, x_test, y_train, y_test, n_trees, tree_depth, weight):
    model = RandomForestClassifier(n_estimators=n_trees, 
                                   max_depth=tree_depth, 
                                   random_state=run_num,
                                   class_weight=weight)
    
    model.fit(x_train, y_train)    
    
    # Make predictions on training and testing sets
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    
    #performance indicators on training and testing sets
    #auc, acc, f1
    perf_train = mf.classification_performance(y_train, y_pred_train, all_indicators= True)
    perf_test = mf.classification_performance(y_test, y_pred_test, all_indicators= True)
    
    
    return perf_train, perf_test





def trials(df, x_vars, y_vars, n_trees, tree_depth, weight, resampling, max_expruns): 
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
        
        if resampling:
            if resampling == 'smote':
                print('doing smote')
                sampler = imblearn.over_sampling.SMOTE(random_state = 50-run_num)
            elif resampling == 'smoteenn':
                print('doing smoteenn')
                sampler = imblearn.combine.SMOTEENN(random_state=50-run_num)
            
            print(f'resampled random forest at N trees = {n_trees}, depth = {tree_depth}')
            x_train, y_train = sampler.fit_resample(x_train, y_train)
            
            
        result_train, result_test = random_forest(run_num,
                                                  x_train, 
                                                  x_test, 
                                                  y_train.values.ravel(),
                                                  y_test.values.ravel(), 
                                                  n_trees,
                                                  tree_depth, 
                                                  weight)
        
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



def class_weight_eperiment(df, x_vars, y_vars, n_trees, tree_depth, class_weight_range, resampling, max_expruns):
    results_train = {}
    results_test = {}
    for w in class_weight_range:
        r_train, r_test = trials(df, 
                                 x_vars, 
                                 y_vars, 
                                 n_trees, 
                                 tree_depth, 
                                 w, 
                                 resampling,
                                 max_expruns)
        pos_w = w[1]
        results_train[pos_w] = r_train
        results_test[pos_w] = r_test
    return results_train, results_test


        
    

def n_trees_experiment(df, x_vars, y_vars, n_trees_range, tree_depth, max_expruns):
    results_train = {}
    results_test = {}
    for n_trees in n_trees_range:
        r_train, r_test = trials(df, x_vars, y_vars, n_trees, tree_depth, max_expruns)
        results_train[n_trees] = r_train
        results_test[n_trees] = r_test
        
    return results_train, results_test
    




def print_results(results, file_name):
    cols = ['positive_weight',
            'mean_auc', 'sd_auc', 
            'mean_acc', 'sd_acc', 
            'mean_f1_score', 'sd_f1_score',
            'mean_precision', 'sd_precision',
            'mean_recall', 'sd_recall']
    indis = ['auc', 'acc', 'f1_score', 'precision', 'recall']
    
    L = []
    for k in results.keys():
        row = [k]
        print('\n\nRandom Forest performance at positive class weight =  : ', k)
        for i in range(0, len(indis)):
            mean_indicator = np.mean(results[k][i]).round(4)
            sd_indicator = np.std(results[k][i], axis = 0).round(2)
            row.append(mean_indicator)
            row.append(sd_indicator)
            print(f'the mean {indis[i]}  =  {mean_indicator}')
            print(f'the sd {indis[i]}  =  {sd_indicator}')
               
        L.append(row)
    L = pd.DataFrame(L, columns= cols)
    
    #print to csv
    L.to_csv(file_name, sep = ',', header=True)
   


    

     