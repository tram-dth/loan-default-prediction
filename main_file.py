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
import stacking




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




# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

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




# =============================================================================
# DECISON TREE
# =============================================================================

def decision_tree_experiments(max_depth_range, resampling, max_expruns = 5):
    results_train, results_test = decision_tree.tree_depth_experiment(df, 
                                                                      x_vars, 
                                                                      y_var, 
                                                                      max_depth_range, 
                                                                      max_expruns,
                                                                      resampling)
    #print results
    if resampling:
        decision_tree.print_results(results_train, 'decision_tree_{resampling}_perf_train_allvars.csv')
        decision_tree.print_results(results_test, 'decision_tree_{resampling}_perf_test_allvars.csv')
    else:
        decision_tree.print_results(results_train, 'decision_tree_perf_train_allvars.csv')
        decision_tree.print_results(results_test, 'decision_tree_perf_test_allvars.csv')
    return 

#no oversampling
max_depth_range = [i for i in range(5, 61)]
decision_tree_experiments(max_depth_range, resampling = False)

#with oversampling
decision_tree_experiments(max_depth_range, resampling = 'smote')
decision_tree_experiments(max_depth_range, resampling = 'smoteenn')





# =============================================================================
# Random forest and XGBoost tunning experiments
# =============================================================================

def tree_ensemble_tunning(model, resampling, n_tree_range, max_depth_range, class_weight_range, max_expruns = 5):
    if model ==  'rf':
        experiment = random_forest.class_weight_eperiment
        print_results = random_forest.print_results
    
    elif model == 'xgb':
        experiment = XGBoost.class_weight_eperiment
        print_results = XGBoost.print_results
    
    for n_tree in n_tree_range:
        for tree_depth in max_depth_range:
            results_train, results_test = experiment(df,
                                                    x_vars,
                                                    y_var,
                                                    n_tree,
                                                    tree_depth,
                                                    class_weight_range,
                                                    resampling,
                                                    max_expruns)
            if resampling:
                print_results(results_train, f"{model}_{resampling}_pweight_trees{n_tree}_depth{tree_depth}_perf_train_allvars.csv")
                print_results(results_test, f"{model}_{resampling}_pweight_trees{n_tree}_depth{tree_depth}_perf_test_allvars.csv")                                                                             
            else:
                print_results(results_train, f"{model}_pweight_trees{n_tree}_depth{tree_depth}_perf_train_allvars.csv")
                print_results(results_test, f"{model}_pweight_trees{n_tree}_depth{tree_depth}_perf_test_allvars.csv")                                                                         
        
    return




#Random forest

#no resample
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

tree_ensemble_tunning('rf', False, n_tree_range, max_depth_range, class_weight_range)

#smote and smoteenn
class_weight_range = [{0: 1, 1: 1}, 
                      {0:1, 1:2},
                      {0:1, 1:3},
                      {0: 1, 1: 4} ]
n_tree_range = [200, 300]
max_depth_range = [5, 20, 35]



tree_ensemble_tunning('rf', 'smote', n_tree_range, max_depth_range, class_weight_range, max_expruns=3)
tree_ensemble_tunning('rf', 'smoteenn', n_tree_range, max_depth_range, class_weight_range, max_expruns=3)

         



#___________XGBoost

#no resample
weight_range = [1, 3, 4, 5, 10, 20, 25, 30]
n_tree_range = [150, 200, 350, 400, 450, 500]
max_depth_range = [4, 5, 7, 10, 15, 20]

tree_ensemble_tunning('xgb', False, n_tree_range, max_depth_range, weight_range)


#______resample_________________
weight_range = [1, 2, 3,  4]
n_tree_range = [200, 500]
max_depth_range = [4, 7]


tree_ensemble_tunning('xgb', 'smote', n_tree_range, max_depth_range, weight_range)
tree_ensemble_tunning('xgb', 'smoteenn', n_tree_range, max_depth_range, weight_range)



