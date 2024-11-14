# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:53:00 2024

@author: dthtr
"""





#my own files
import myFuncs as mf
from myFuncs import continuous_vars

import PCA_reduced
import logistic_regress
import logistic_regress_oversample
import decision_tree
import XGBoost




#_____SOME GLOBAL VARS AND FUNCTIONS__________________

data_path = 'data\\train.csv'

def data(data_path, log_transform, remove_outliers_var):
    df = mf.import_csv(data_path)
    
    if remove_outliers_var:
        mf.remove_outliers(df, remove_outliers_var , q = 0.99995)
    
    df = mf.data_format(df, log_transform)
    
    df = mf.rescaling(df, 'standard', continuous_vars)
    
    return df



#______________DATA PREPROCESSING__________________________________________

df = data(data_path, log_transform = True, remove_outliers_var = 'pri_current_balance')


#df = PCA_reduced.PCA_reduced(df, continuous_vars)



#________________var names__________________________
all_vars = df.columns
all_vars = list(all_vars)
y_var = 'defaulted'
x_vars = [v for v in all_vars if v != y_var]


#__logistic regression with different decision threshold

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




#_________decision trees with different depths___________
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

#________random forest____different method of sampling___________





#________xgboost__________different weight_________________

weight_range = [1, 3, 4, 5, 10, 20, 25, 30]

#at 100 trees, max_depth = 5
xgb_results_train_1, xgb_results_test_1 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 100, 
                                                                         tree_depth = 5, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)
#print results
#for train set 
XGBoost.print_results(xgb_results_train_1, 'xgb_pweight1_trees100_depth5_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_1, 'xgb_pweight1_trees100_depth5_perf_test_allvars.csv', no_indi=5)







#at 150 trees, max_depth = 4
xgb_results_train_2_0, xgb_results_test_2_0 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 150, 
                                                                         tree_depth = 4, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_2_0, 'xgb_pweight1_trees150_depth4_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_2_0, 'xgb_pweight1_trees150_depth4_perf_test_allvars.csv', no_indi=5)





#at 150 trees, max_depth = 5
xgb_results_train_2, xgb_results_test_2 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 150, 
                                                                         tree_depth = 5, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_2, 'xgb_pweight1_trees150_depth5_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_2, 'xgb_pweight1_trees150_depth5_perf_test_allvars.csv', no_indi=5)







#at 200 trees, max_depth = 4
xgb_results_train_3_0, xgb_results_test_3_0 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 200, 
                                                                         tree_depth = 4, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_3_0, 'xgb_pweight1_trees200_depth4_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_3_0, 'xgb_pweight1_trees200_depth4_perf_test_allvars.csv', no_indi=5)




#at 200 trees, max_depth = 5
xgb_results_train_3, xgb_results_test_3 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 200, 
                                                                         tree_depth = 5, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_3, 'xgb_pweight1_trees200_depth5_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_3, 'xgb_pweight1_trees200_depth5_perf_test_allvars.csv', no_indi=5)



#at 200 trees, max_depth = 10
xgb_results_train_3_2, xgb_results_test_3_2 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 200, 
                                                                         tree_depth = 10, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_3_2, 'xgb_pweight_trees200_depth10_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_3_2, 'xgb_pweight_trees200_depth10_perf_test_allvars.csv', no_indi=5)








#at 500 trees, max_depth = 4
xgb_results_train_5_0, xgb_results_test_5_0 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 500, 
                                                                         tree_depth = 4, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_5_0, 'xgb_pweight1_trees500_depth4_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_5_0, 'xgb_pweight1_trees500_depth4_perf_test_allvars.csv', no_indi=5)







#at 500 trees, max_depth = 5
xgb_results_train_5, xgb_results_test_5 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 500, 
                                                                         tree_depth = 5, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_5, 'xgb_pweight1_trees500_depth5_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_5, 'xgb_pweight1_trees500_depth5_perf_test_allvars.csv', no_indi=5)





#at 500 trees, max_depth = 10
xgb_results_train_6, xgb_results_test_6 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 500, 
                                                                         tree_depth = 10, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_6, 'xgb_pweight_trees500_depth10_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_6, 'xgb_pweight_trees500_depth10_perf_test_allvars.csv', no_indi=5)



#at 500 trees, max_depth = 15
xgb_results_train_6_5, xgb_results_test_6_5 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 500, 
                                                                         tree_depth = 15, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_6_5, 'xgb_pweight_trees500_depth15_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_6_5, 'xgb_pweight_trees500_depth15_perf_test_allvars.csv', no_indi=5)




#at 500 trees, max_depth = 20
xgb_results_train_7, xgb_results_test_7 = XGBoost.class_weight_eperiment(df, 
                                                                         x_vars, 
                                                                         y_var, 
                                                                         n_trees = 500, 
                                                                         tree_depth = 20, 
                                                                         pos_weight_range = weight_range, 
                                                                         max_expruns = 5)

XGBoost.print_results(xgb_results_train_7, 'xgb_pweight_trees500_depth20_perf_train_allvars.csv', no_indi=5)
XGBoost.print_results(xgb_results_test_7, 'xgb_pweight_trees500_depth20_perf_test_allvars.csv', no_indi=5)










