# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:53:00 2024

@author: dthtr
"""





#my own files
import myFuncs as mf
from myFuncs import continuous_vars, x_vars

import PCA_reduced
import logistic_regress
import logistic_regress_oversample
import decision_tree
import random_forest
import XGBoost




#_____DATA IMPORT AND PREPROCESSING__________________

data_path = 'data\\train.csv'

def data(data_path, log_transform, remove_outliers_var):
    df = mf.import_csv(data_path)
    
    if remove_outliers_var:
        mf.remove_outliers(df, remove_outliers_var , q = 0.99995)
    
    df = mf.data_format(df, log_transform)
    
    df = mf.rescaling(df, 'standard', continuous_vars)
    
    return df



df = data(data_path, log_transform = True, remove_outliers_var = 'pri_current_balance')


#df = PCA_reduced.PCA_reduced(df, continuous_vars)

all_vars = df.columns
all_vars = list(all_vars)
y_var = 'defaulted'




#__________LOGISTIC REGRESSION
decision_threshold_range = [0.4, 0.45, 0.5, 0.55, 0.6]

#with class_weight = 'balanced"
logistic_regress.decision_threshold_experiment(df, 
                                               x_vars, 
                                               y_var, 
                                               decision_threshold_range, 
                                               max_expruns = 5)


logistic_regress_oversample.decision_threshold_experiment(df,
                                                          x_vars,
                                                          y_var,
                                                          decision_threshold_range,
                                                          max_expruns=5)




#_________DECISON TREE___________
#no oversampling
tree_depth_range = [i for i in range(5, 61)]
dt_results_train, dt_results_test = decision_tree.tree_depth_experiment(df, 
                                                                        x_vars, 
                                                                        y_var, 
                                                                        tree_depth_range, 
                                                                        max_expruns = 5)
#print results
decision_tree.print_results(dt_results_train, 'decision_tree_perf_train_allvars.csv')
decision_tree.print_results(dt_results_test, 'decision_tree_perf_test_allvars.csv')
#seems like stuff plateau around depth = 40


#with oversampling
tree_depth_range = [i for i in range(5, 61)]
dt_results_train_over, dt_results_test_over = decision_tree.tree_depth_experiment(df, 
                                                                                  x_vars, 
                                                                                  y_var, 
                                                                                  tree_depth_range, 
                                                                                  max_expruns = 5,
                                                                                  oversample=True)
decision_tree.print_results(dt_results_train_over, 'decision_tree_oversample_perf_train_allvars.csv')
decision_tree.print_results(dt_results_test_over, 'decision_tree_oversample_perf_test_allvars.csv')





#________RANDOM FOREST_______________
class_weight_range = [ 
    {0: 1, 1: 1},
    {0: 1, 1: 3},
    {0: 1, 1: 4},
    {0: 1, 1: 5},
    {0: 1, 1: 10},
    {0: 1, 1: 20},
    ]

n_tree_range = [150, 200, 300]
max_depth_range = [5, 20, 35]



for n_tree in n_tree_range:
    for tree_depth in max_depth_range:
        rf_results_train, rf_results_test = random_forest.class_weight_eperiment(df,
                                                                                 x_vars,
                                                                                 y_var,
                                                                                 n_tree,
                                                                                 tree_depth,
                                                                                 class_weight_range,
                                                                                 max_expruns = 5)
        random_forest.print_results(rf_results_train, f"rf_cweight_tree{n_tree}_depth{tree_depth}_perf_train_allvars.csv")
        random_forest.print_results(rf_results_test, f"rf_cweight_tree{n_tree}_depth{tree_depth}_perf_test_allvars.csv")                                                                         

            



#________XGBOOST___________________________


#_________class weight no resample____________
weight_range = [1, 3, 4, 5, 10, 20, 25, 30]
n_tree_range = [150, 200, 350, 400, 450, 500]
max_depth_range = [4, 5, 7, 10, 15, 20]


weight_range = [1, 3, 4, 5, 10, 20, 25, 30]
n_tree_range = [250]
max_depth_range = [4, 5, 7, 10, 15, 20]



for n_tree in n_tree_range:
    for tree_depth in max_depth_range:
        xgb_results_train, xgb_results_test = XGBoost.class_weight_eperiment(df, 
                                                                             x_vars, 
                                                                             y_var, 
                                                                             n_tree, 
                                                                             tree_depth, 
                                                                             weight_range, 
                                                                             max_expruns = 5)
        XGBoost.print_results(xgb_results_train, f'xgb_pweight_trees{n_tree}_depth{tree_depth}_perf_train_allvars.csv', no_indi=5)
        XGBoost.print_results(xgb_results_test, f'xgb_pweight_trees{n_tree}_depth{tree_depth}_perf_test_allvars.csv', no_indi=5)


#______________class weight and resample
xgb_results_train, xgb_results_test = XGBoost.class_weight_eperiment(df, 
                                                                     x_vars, 
                                                                     y_var, 
                                                                     n_tree, 
                                                                     tree_depth, 
                                                                     weight_range, 
                                                                     max_expruns = 5)
XGBoost.print_results(xgb_results_train, f'xgb_pweight_trees{n_tree}_depth{tree_depth}_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test, f'xgb_pweight_trees{n_tree}_depth{tree_depth}_perf_test_allvars.csv', no_indi=5)





#_________STACKING_________________________________


stacking_train, stacking_test = stacking.running(df, 
                                                 x_vars, 
                                                 y_vars, 
                                                 max_exprun = 5, 
                                                 over_under_sampling = True)

