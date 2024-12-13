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
    colours = ['dodgerblue', 'lightseagreen', 'coral', 'darkviolet']
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



#assemble evaluation results from train or test set prediction
def assemble_results(model, resampling, n_tree_range, depth_range, train_set = False):
    #list of all results
    L = []
    
    
    if resampling:
        folder = f'results_fullvars\\{model.upper()}_resampling'
        file_prefix = f'{model}_{resampling}_pweight_'
    else:
        folder = f'results_fullvars\\{model.upper()}_no_resampling'
        file_prefix = f'{model}_pweight_'
    
    if train_set:
        file_suffix = '_perf_train_allvars.csv'
    else:
        file_suffix = '_perf_test_allvars.csv'
    
            
    for n_tree in n_tree_range:
        for depth in depth_range:
            fname = file_prefix + f'trees{n_tree}_depth{depth}' + file_suffix
            path = os.path.join(folder, fname) 
            df = pd.read_csv(path, header=0, index_col=False)
            for i, row in df.iterrows():
                try:
                    pw = row['positive_weight']
                    auc = row['mean_auc']
                    #acc = row['mean_acc']
                    f1 = row['mean_f1_score']
                    precision = row['mean_precision']
                    recall = row['mean_recall']
                    result_tuple = (n_tree, depth, pw, auc, f1, precision, recall)
                    L.append(result_tuple)
                except:
                    print('error at ntree = ', n_tree, ' depth = ', depth)

    
    #print the results into a single file
    results_df = pd.DataFrame(L, 
                              index = None, 
                              columns=['n_tree', 'max_depth', 'positive_weight', 'auc', 'f1', 'precision', 'recall'])
    
    
    file_out_name = file_prefix[:-8] + 'all_results' + file_suffix[5:]
    
    results_df.to_csv(file_out_name, sep = ',', header=True)
    
   
    return L


#print best result out for latex
def best_perf(results):
    names = {0: 'number of trees', 1: 'max tree depth', 2: 'positive weight', 
             3: 'AUC', 4: 'F1 score', 5: 'precision', 6: 'recall' }
    
    print('results order: ntree, depth, pos_weight, AUC, F1, precision, recall\n')
    
    for i in range(3, 7):
        L = sorted(results, key = lambda x: x[i], reverse=True)
        print(f'max {names[i]} = {L[0][i]} at: ')
        s = [str(x) for x in L[0]]
        s = names[i] + ' & ' +  ' & '.join(s)
        print(s + '\\\\ \n\n')
        
    return



def RESULTS(model, resampling, n_tree_range, depth_range):
    print(f'\n model = {model}, resampling = {resampling} \n')
    
    results_train = assemble_results(model, resampling, n_tree_range, depth_range, train_set=True)
    results_test = assemble_results(model, resampling, n_tree_range, depth_range)
    best_perf(results_test)
    return results_train, results_test
    


#plot performance when 1 hyperparameter varies
#fix the other 2
#index: 0 = n_trees, 1 = max_depth, 2 = pos_weight
#Pi = index of varying parameter in the result tuple
#F1, F1i = Fix parameter 1 value and index tuple
#F2, F2i = Fix parameter 2 value and index in result tuple
def plot_1_hyperpara(results_train, results_test, Pi, F1, F1i, F2, F2i, model, resampling = False):
    names = {0: 'number of trees', 1: 'max tree depth', 2: 'positive weight', 
             3: 'AUC', 4:'F1_score', 5: 'precision', 6: 'recall' }
    
    abbre = {0:'tree', 1:'depth', 2: 'pweight'}
    L_train = [item for item in results_train if item[F1i] == F1  and item[F2i] == F2]
    
    L_test = [item for item in results_test if item[F1i] == F1  and item[F2i] == F2]
    
    x = [item[Pi] for item in L_test]
    
    for i in range(3, 7):
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
            if resampling: 
                figname = f'xgb_{resampling}_{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
            else:
                figname = f'xgb_{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
        elif model == 'random forest':
            if resampling:
                figname = f'rf_{resampling}{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
            else: 
                figname = f'rf_{abbre[F1i]}{F1}_{abbre[F2i]}{F2}_{names[i]}.png'
        plt.savefig(figname)
        plt.clf()
        
   





# =============================================================================
# Random forest no resample
# =============================================================================

n_tree_range = [150, 200, 300]
max_depth_range = [5, 20, 35]
rf_results_train, rf_results_test = RESULTS('rf', False, n_tree_range, max_depth_range)



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


# vary positive weight, fix n_tree = 300, depth = 20
plot_1_hyperpara(rf_results_train, rf_results_test, 2, 300, 0, 20, 1, 'random forest')


# =============================================================================
# Random forest - SMOTE
# =============================================================================

n_tree_range = [200, 300]
max_depth_range = [5, 20, 35]

rf_smote_results_train, rf_smote_results_test  = RESULTS('rf', 'smote', n_tree_range, max_depth_range)

rf_smoteenn_results_train, rf_smoteenn_results_test  = RESULTS('rf', 'smoteenn', n_tree_range, max_depth_range)


# =============================================================================
# XGBoost - no resample 
# =============================================================================

n_tree_range = [150, 200, 250, 300, 350, 400, 450, 500]
depth_range = [4, 5, 7, 10, 15, 20]


#no resampling        
xgb_results_train, xgb_results_test = RESULTS('xgb', False, n_tree_range, depth_range)


#plot number of trees vs perf at max depth = 4, pos weight = 4
plot_1_hyperpara(xgb_results_train, xgb_results_test, 0, 4, 1, 4, 2, 'XGBoost')

#plot max depth vs per at number of trees = 500 and pos_weight = 5
plot_1_hyperpara(xgb_results_train, xgb_results_test, 1, 500, 0, 4, 2, 'XGBoost')


# vary number of trees, fix positive weight = 4, depth = 5
plot_1_hyperpara(xgb_results_train, xgb_results_test, 0, 10, 1, 4, 2, 'XGBoost')

# vary positive weight, fix n_tree = 300, depth = 20
plot_1_hyperpara(xgb_results_train, xgb_results_test, 2, 500, 0, 10, 1, 'XGBoost')


# =============================================================================
# XGBoost Resample smote  
# =============================================================================
n_tree_range = [200, 500]
depth_range = [4, 7]

xgb_results_smote_train, xgb_results_smote_test = RESULTS('xgb', 'smote', n_tree_range, depth_range)



# =============================================================================
# XGBoost Resample smoteenn
# =============================================================================

n_tree_range = [200, 500]
depth_range = [4, 7]
xgb_results_smoteenn_train, xgb_results_smoteenn_test = RESULTS('xgb', 'smoteenn', n_tree_range, depth_range)







