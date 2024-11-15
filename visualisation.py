# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:25:54 2024

@author: dthtr
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#my own module
import myFuncs as mf


# =============================================================================
# FUNCTIONS
# =============================================================================

indicators = ['mean_auc', 
              'mean_acc', 
              'mean_f1_score',  
              'mean_precision',
              'mean_recall']


def train_test_plot(path_train, path_test, x, y, title = False, print_fig = False):

    df_train = pd.read_csv(path_train, header=0, index_col=False)
    df_test = pd.read_csv(path_test, header=0, index_col=False)

    l1, = plt.plot(df_train[x], df_train[y], linestyle = '--', marker = 'o', color = 'coral')
    l2, = plt.plot(df_test[x], df_test[y], linestyle = '--', marker = 'o', color = 'dodgerblue')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.ylim(0, 1)
    plt.legend([l1, l2], ['training data', 'testing data'])
    if title:
        plt.title(title)
    if print_fig:
        plt.savefig(print_fig)
    plt.show()
    plt.clf()


def all_indicators_plot(path, x):
    df = pd.read_csv(path, header=0, index_col=False)
    colours = ['dodgerblue', 'lightseagreen', 'coral', 'darkviolet', 'gold']
    L = []
    for i, y in enumerate(indicators):
        line, = plt.plot(df[x], df[y], linestyle = '--', marker = 'o', color = colours[i])
        L.append(line)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.ylim(0, 1)
    plt.legend(L, indicators)
    plt.show()
    plt.clf()


#this is NOT PRcurve
def precision_recall_plot(path, x, title = False, print_fig = False):
    df = pd.read_csv(path, header=0, index_col=False)
    l_p, = plt.plot(df[x], df['mean_precision'], linestyle = '--', marker = 'o', color = 'dodgerblue')
    l_r, = plt.plot(df[x], df['mean_recall'], linestyle = '--', marker = 'o', color = 'coral')
    plt.ylim(0, 1)
    plt.xlabel(x)
    if title:
        plt.title(title)
    plt.legend([l_p, l_r], ['mean precision', 'mean recall'])
    if print_fig:
        plt.savefig(print_fig)
    plt.show()
    plt.clf()





def all_results(model, n_tree_range, depth_range, train_set = False):
    #list of all results
    L = []
    
    for n_tree in n_tree_range:
        for depth in depth_range:
            if model == 'xgb':
                folder = 'results_fullvars\\xgboost_no_resample'
                if train_set == True: 
                    fname = f'xgb_pweight_trees{n_tree}_depth{depth}_perf_train_allvars.csv'
                else:
                    fname = f'xgb_pweight_trees{n_tree}_depth{depth}_perf_test_allvars.csv'
            elif model == 'RF':
                folder = "results_fullvars\\random_forest_no_resample"
                if train_set == True:
                    fname = f'rf_cweight_tree{n_tree}_depth{depth}_perf_train_allvars.csv'
                else:
                    fname = f'rf_cweight_tree{n_tree}_depth{depth}_perf_test_allvars.csv'
            
            path = os.path.join(folder, fname) 
            df = pd.read_csv(path, header=0, index_col=False)
            for i, row in df.iterrows():
                try:
                    pw = row['positive_weight']
                    auc = row['mean_auc']
                    acc = row['mean_acc']
                    f1 = row['mean_f1_score']
                    precision = row['mean_precision']
                    recall = row['mean_recall']
                    result_tuple = (n_tree, depth, pw, auc, acc, f1, precision, recall)
                    L.append(result_tuple)
                except:
                    print('error at ntree = ', n_tree, ' depth = ', depth)

    
    #print the results into a single file
    results_df = pd.DataFrame(L, 
                              index = None, 
                              columns=['n_tree', 'max_depth', 'positive_weight', 'auc', 'acc', 'f1', 'precision', 'recall'])
    
    
    if model =='xgb':
        if train_set: 
            name = 'XGB_all_results_no_resample_train_set.csv'
        else:
            name = 'XGB_all_results_no_resample.csv'
    elif model == 'RF':
        if train_set:
            name = 'RF_all_results_no_resample_train_set.csv'
        else:
            name = 'RF_all_results_no_resample.csv'
    
    results_df.to_csv(name, sep = ',', header=True)
    
   
    return L


#plot performance of 1 hyperparameter 
#fix the other 2
#index: 0 = n_trees, 1 = max_depth, 2 = pos_weight
#Pi = index of varying parameter
#F1 = Fix parameter 1 value
#F2 = Fix parameter 2 value
def plot_1_hyperpara(results_train, results_test, Pi, F1, F1i, F2, F2i, model):
    names = {0: 'number of trees', 1: 'max tree depth', 2: 'positive weight', 
             3: 'AUC', 4:'ACC', 5:'F1_score', 6: 'precision', 7: 'recall' }
    
    abbre = {0:'tree', 1:'depth', 2: 'pweight'}
    L_train = [item for item in results_train if item[F1i] == F1  and item[F2i] == F2]
    
    L_test = [item for item in results_test if item[F1i] == F1  and item[F2i] == F2]
    
    x = [item[Pi] for item in L_test]
    
    for i in range(3, 8):
        y_train = [item[i] for item in L_train]
        y_test = [item[i] for item in L_test]
        
        l1, = plt.plot(x, y_train, linestyle = '--', marker = 'o', color = 'coral')
        l2, = plt.plot(x, y_test, linestyle = '--', marker = 'o', color = 'dodgerblue')
        plt.xlabel(names[Pi])
        plt.ylabel(names[i])
        plt.ylim(0, 1.1)
        plt.legend([l1, l2], ['training data', 'testing data'])
        
        title = f'{model} at {names[F1i]} = {F1} and {names[F2i]} = {F2}'       
        plt.title(title)
        plt.show()
        if model == 'XGBoost':
            figname = f'xgb_{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
        elif model == 'random forest':
            figname = f'rf_{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
        plt.savefig(figname)
        plt.clf()
        
   



# =============================================================================
# XGBoost - no resample 
# =============================================================================

def xgb_no_resample_plot(n_tree, depth, x = 'positive_weight'):    
    path_train = f'results_fullvars\\xgboost_no_resample\\xgb_pweight_trees{n_tree}_depth{depth}_perf_train_allvars.csv'
    path_test = f'results_fullvars\\xgboost_no_resample\\xgb_pweight_trees{n_tree}_depth{depth}_perf_test_allvars.csv'
    
    for y in indicators:
        train_test_plot(path_train, 
                        path_test, 
                        x, 
                        y, 
                        title = f'XGBoost with {n_tree} trees and max tree depth = {depth} ',
                        print_fig = f'xgb_trees_{y[5:]}_tree{n_tree}_depth{depth}.png')
    
    precision_recall_plot(path_test, 
                          x, 
                          title = f'XGBoost with {n_tree} trees and max tree depth = {depth}',
                          print_fig=f'xgb_trees_precision_recall_tree{n_tree}_depth{depth}.png')    




weight_range = [1, 3, 4, 5, 10, 20, 25, 30]
n_tree_range = [150, 200, 250, 300, 350, 400, 450, 500]
depth_range = [4, 5, 7, 10, 15, 20]



xgb_results = all_results('xgb', n_tree_range, depth_range)

#max AUC
L = sorted(xgb_results, key = lambda x: x[3], reverse=True)
print('max AUC at')
print(L[0])
print(L[0][3], L[0][0], L[0][1], L[0][2], sep=' & ')


#max ACC
L = sorted(xgb_results, key = lambda x: x[4], reverse=True)
print('max ACC at')
print(L[0])
print(L[0][4], L[0][0], L[0][1], L[0][2], sep=' & ')

#max F1
L = sorted(xgb_results, key = lambda x: x[5], reverse=True)
print('max F1 at')
print(L[0])
print(L[0][5], L[0][0], L[0][1], L[0][2], sep=' & ')

#max precision
L = sorted(xgb_results, key = lambda x: x[6], reverse=True)
print('max precision at')
print(L[0])
print(L[0][6], L[0][0], L[0][1], L[0][2], sep=' & ')


#max recall
L = sorted(xgb_results, key = lambda x: x[7], reverse=True)
print('max recall at')
print(L[0])
print(L[0][7], L[0][0], L[0][1], L[0][2], sep=' & ')



weight_range = [1, 3, 4, 5, 10, 20, 25, 30]
n_tree_range = [150, 200, 250, 300, 350, 400, 450, 500]
depth_range = [4, 5, 7, 10, 15, 20]
        
xgb_results_train = all_results('xgb', n_tree_range, depth_range, train_set=True)    
xgb_results_test = all_results('xgb', n_tree_range, depth_range)


#plot number of trees vs perf at max depth = 4, pos weight = 4
plot_1_hyperpara(xgb_results_train, xgb_results_test, 0, 4, 1, 4, 2, 'XGBoost')

#plot max depth vs per at number of trees = 500 and pos_weight = 5
plot_1_hyperpara(xgb_results_train, xgb_results_test, 1, 500, 0, 4, 2, 'XGBoost')






# =============================================================================
# Random forest no resample
# =============================================================================

n_tree_range = [150, 200, 300]
max_depth_range = [5, 20, 35]
rf_results_train = all_results('RF', n_tree_range, max_depth_range, train_set=True)
rf_results_test = all_results('RF', n_tree_range, max_depth_range)


#max AUC
L = sorted(rf_results_test, key = lambda x: x[3], reverse=True)
print('max AUC at')
print(L[0])
print(L[0][3], L[0][0], L[0][1], L[0][2], sep=' & ')


#max ACC
L = sorted(rf_results_test, key = lambda x: x[4], reverse=True)
print('max ACC at')
print(L[0])
print(L[0][4], L[0][0], L[0][1], L[0][2], sep=' & ')

#max F1
L = sorted(rf_results_test, key = lambda x: x[5], reverse=True)
print('max F1 at')
print(L[0])
print(L[0][5], L[0][0], L[0][1], L[0][2], sep=' & ')

#max precision
L = sorted(rf_results_test, key = lambda x: x[6], reverse=True)
print('max precision at')
print(L[0])
print(L[0][6], L[0][0], L[0][1], L[0][2], sep=' & ')


#max recall
L = sorted(rf_results_test, key = lambda x: x[7], reverse=True)
print('max recall at')
print(L[0])
print(L[0][7], L[0][0], L[0][1], L[0][2], sep=' & ')


# vary positive weight, fix n_tree = 300, depth = 5
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 5, 1, 'random forest')

# vary positive weight, fix n_tree = 200, depth = 5
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 200, 0, 5, 1, 'random forest')

# vary positive weight, fix n_tree = 150, depth = 5
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 200, 0, 5, 1, 'random forest')


# vary positive weight, fix n_tree = 300, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 20, 1, 'random forest')

# vary positive weight, fix n_tree = 200, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 200, 0, 20, 1, 'random forest')


# vary positive weight, fix n_tree = 150, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 150, 0, 20, 1, 'random forest')


# vary positive weight, fix n_tree = 300, depth = 35
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 35, 1, 'random forest')


# vary positive weight, fix n_tree = 200, depth = 35
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 200, 0, 35, 1, 'random forest')


# vary positive weight, fix n_tree = 150, depth = 35
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 150, 0, 35, 1, 'random forest')




# vary positive weight, fix n_tree = 300, depth = 35
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 35, 1, 'random forest')


# vary positive weight, fix n_tree = 300, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 20, 1, 'random forest')


# vary positive weight, fix n_tree = 300, depth = 5
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 5, 1, 'random forest')






# vary number of trees, fix positive weight = 4, depth = 5
plot_1_hyperpara(rf_results_train, rf_results_test, 0, 5, 1, 4, 2, 'random forest')

# vary number of trees, fix positive weight = 4, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 0, 20, 1, 4, 2, 'random forest')

# vary number of trees, fix positive weight = 4, depth = 35
plot_1_hyperpara(rf_results_train, rf_results_test, 0, 35, 1, 4, 2, 'random forest')




# vary depth, fix ntree = 300, positive weight = 4
plot_1_hyperpara(rf_results_train, rf_results_test, 1, 300, 0, 4, 2, 'random forest')

# vary depth, fix ntree = 200, positive weight = 4
plot_1_hyperpara(rf_results_train, rf_results_test, 1, 200, 0, 4, 2, 'random forest')


# vary depth, fix ntree = 150, positive weight = 4
plot_1_hyperpara(rf_results_train, rf_results_test, 1, 150, 0, 4, 2, 'random forest')
