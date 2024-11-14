import imblearn# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:31:30 2024

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

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, plot_tree

import imblearn 
from imblearn.over_sampling import RandomOverSampler


#my own function
import myFuncs as mf




def decision_tree(run_num, x_train, x_test, y_train, y_test, tree_depth, report_tree):
    model = DecisionTreeClassifier(random_state=run_num, max_depth=tree_depth, ccp_alpha=0)
    
    # Train the model using the training sets
    model.fit(x_train, y_train)
    
    #report on tree
    if report_tree:
        r = export_text(model)
        print(r)

    # Make predictions on training and testing sets
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    
    #performance indicators on training and testing sets
    #auc, acc, f1
    perf_train = mf.classification_performance(y_train, y_pred_train)
    perf_test = mf.classification_performance(y_test, y_pred_test)
    
    return perf_train, perf_test


    
def trials(df, x_vars, y_vars, tree_depth, max_expruns, oversample, report_tree = False): 
    auc_all_train = [0]*max_expruns
    acc_all_train = [0]*max_expruns
    f1_all_train = [0]*max_expruns
   
    auc_all_test = [0]*max_expruns
    acc_all_test = [0]*max_expruns
    f1_all_test = [0]*max_expruns
    

    for run_num in range(0, max_expruns): 
        
        x_train, x_test, y_train, y_test = mf.data_split_random(df, x_vars, y_vars, run_num)

        if oversample: 
            oversample = RandomOverSampler(sampling_strategy='minority', 
                                           random_state = 50 - run_num)
            x_train, y_train = oversample.fit_resample(x_train, y_train)

        result_train, result_test = decision_tree(run_num,
                                                  x_train, 
                                                  x_test, 
                                                  y_train.values.ravel(),
                                                  y_test.values.ravel(), 
                                                  tree_depth, 
                                                  report_tree)
        
        
        auc_all_train[run_num] = result_train[0]
        acc_all_train[run_num] = result_train[1]
        f1_all_train[run_num] = result_train[2]
        
        
        auc_all_test[run_num] = result_test[0]
        acc_all_test[run_num] = result_test[1]
        f1_all_test[run_num] = result_test[2]
        
        
    return (auc_all_train, acc_all_train, f1_all_train), (auc_all_test, acc_all_test, f1_all_test) 
  
    
  

def tree_depth_experiment(df, x_vars, y_vars, tree_depth_range, max_expruns, oversample = False):
    results_train = {}
    results_test = {}
    for d in tree_depth_range:
        r_train, r_test = trials(df, x_vars, y_vars, d, max_expruns, oversample)
        results_train[d] = r_train
        results_test[d] = r_test
        
    for d in tree_depth_range:
        print(f'at maximum tree depth = {d} : ')
        print('training set: mean auc is ', np.mean(results_train[d][0]).round(4))
        print('training set: std auc is ', '(', np.std(results_train[d][0], axis = 0).round(2), ')')
        print('test set: mean auc is ', np.mean(results_test[d][0]).round(4))
        print('test set: std auc is ', '(', np.std(results_test[d][0], axis = 0).round(2), ')')
        
        print('training set: mean acc score is ', np.mean(results_train[d][1]).round(4))
        print('training set: std acc score is ', '(', np.std(results_train[d][1], axis= 0).round(2), ')')
        print('test set: mean acc score is ', np.mean(results_test[d][1]).round(4))
        print('test set: std acc score is ', '(', np.std(results_test[d][1], axis= 0).round(2), ')')
        
        
        print('training set: mean f1 score is ', np.mean(results_train[d][2]).round(4))
        print('training set: std f1 score is ', '(', np.std(results_train[d][2], axis= 0).round(2), ')')
        print('test set: mean f1 score is ', np.mean(results_test[d][2]).round(4))
        print('test set: std f1 score is ', '(', np.std(results_test[d][2], axis= 0).round(2), ')')
        
        print('\n\n')
    
    return results_train, results_test
    


def print_results(results, file_name, no_indi = 3):
    cols = ['max_tree_depth',
            'mean_auc', 'sd_auc', 
            'mean_acc', 'sd_acc', 
            'mean_f1_score', 'sd_f1_score',
            'mean_precision', 'sd_precision',
            'mean_recall', 'sd_recall']
    L = []
    for k in results.keys():
        row = [k]
        for i in range(0, no_indi):
            mean_indicator = np.mean(results[k][i]).round(4)
            sd_indicator = np.std(results[k][i], axis = 0).round(2)
            row.append(mean_indicator)
            row.append(sd_indicator)
        L.append(row)
    L = pd.DataFrame(L, columns= cols[: (1+no_indi*2)])
    L.to_csv(file_name, sep = ',', header=True)
    
            

# def prunning(run_num, x_train, x_test, y_train, y_test, tree_depth, draw_roc):
#     #initially train an unpruned tree
#     model = DecisionTreeClassifier(random_state=run_num, ccp_alpha=0)
#     model.fit(X_train, y_train)
    
#     # Step 2: get prunning path
#     path = model.cost_complexity_pruning_path(X_train, y_train)
#     ccp_alphas = path.ccp_alphas  # List of effective alphas for pruning
#     impurities = path.impurities  # Total impurity of leaves for each alpha
    
#     # Step 3: Train a series of trees with different ccp_alpha values
#     Dtrees = []
#     for alpha in ccp_alphas:
#         pruned_tree = DecisionTreeClassifier(random_state=run_num, ccp_alpha=alpha)
#         pruned_tree.fit(X_train, y_train)
#         Dtrees.append(pruned_tree)
    
#     # Step 4: Plot accuracy vs. alpha to find the best level of pruning
#     train_scores = [tree.score(X_train, y_train) for tree in trees]
#     test_scores = [tree.score(X_test, y_test) for tree in trees]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(ccp_alphas, train_scores, marker='o', label='train accuracy', drawstyle="steps-post")
#     plt.plot(ccp_alphas, test_scores, marker='o', label='test accuracy', drawstyle="steps-post")
#     plt.xlabel("Alpha")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy vs. alpha for training and testing sets")
#     plt.legend()
#     plt.show()
    
#     # Step 5: Choose the best alpha (based on the plot or cross-validation)
#     best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    
#     # Train the pruned tree with the chosen alpha
#     pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
#     pruned_tree.fit(X_train, y_train)
    
#     # Plot the pruned decision tree
#     plt.figure(figsize=(12, 8))
#     plot_tree(pruned_tree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
#     plt.title("Pruned Decision Tree")
#     plt.show()