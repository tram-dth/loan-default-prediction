# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:02:31 2024

@author: dthtr
"""

import os

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from math import log, log10, log1p

import sklearn
sklearn.set_config(transform_output = "pandas")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve


import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# =============================================================================
# VARIABLE NAMES
# =============================================================================


irrelevant_vars =  [ 'current_pincode_id', 
                     'disbursal_date', 
                     'uniqueid', 
                     'perform_cns_score_description']



id_vars = ['branch_id', 
           'supplier_id', 
           'manufacturer_id', 
           'state_id', 
           'employee_code_id']

binary = ['employed',
          'mobileno_avl_flag', 
          'aadhar_flag', 
          'pan_flag', 
          'voterid_flag',
          'driving_flag', 
          'passport_flag',
          'defaulted']


continuous_vars = [
     'age',
     'asset_cost',
     'average_acct_age',
     'cns',
     'credit_his_len',
     'delinquent_accts_in_last_six_months',
     'disbursed_amount',
     'ltv',
     'new_accts_in_last_six_months',
     'no_of_inquiries',
     'pri_active_accts',
     'pri_current_balance',
     'pri_disbursed_amount',
     'pri_no_of_accts',
     'pri_overdue_accts',
     'pri_sanctioned_amount',
     'primary_instal_amt',
     'sec_active_accts',
     'sec_current_balance',
     'sec_disbursed_amount',
     'sec_instal_amt',
     'sec_no_of_accts',
     'sec_overdue_accts',
     'sec_sanctioned_amount'
    ]



log_vars = [
     'asset_cost',
     'average_acct_age',
     'cns',
     'credit_his_len',
     'delinquent_accts_in_last_six_months',
     'disbursed_amount',
     'ltv',
     'new_accts_in_last_six_months',
     'pri_active_accts',
     'pri_disbursed_amount',
     'pri_no_of_accts',
     'pri_overdue_accts',
     'pri_sanctioned_amount',
     'primary_instal_amt',
     'sec_active_accts',
     'sec_disbursed_amount',
     'sec_instal_amt',
     'sec_no_of_accts',
     'sec_overdue_accts',
     'sec_sanctioned_amount'
    ]



large_num_vars = [
    'asset_cost', 
    'disbursed_amount',
    'pri_current_balance', 
    'pri_sanctioned_amount', 
    'pri_disbursed_amount',
    'sec_current_balance',
    'sec_sanctioned_amount',
    'sec_disbursed_amount',
    'primary_instal_amt'
    ]

x_vars = [
     'aadhar_flag',
     'age',
     'asset_cost',
     'average_acct_age',
     'branch_id',
     'cns',
     'credit_his_len',
     'delinquent_accts_in_last_six_months',
     'disbursed_amount',
     'driving_flag',
     'employed',
     'employee_code_id',
     'ltv',
     'manufacturer_id',
     'mobileno_avl_flag',
     'new_accts_in_last_six_months',
     'no_of_inquiries',
     'pan_flag',
     'passport_flag',
     'pri_active_accts',
     'pri_current_balance',
     'pri_disbursed_amount',
     'pri_no_of_accts',
     'pri_overdue_accts',
     'pri_sanctioned_amount',
     'primary_instal_amt',
     'sec_active_accts',
     'sec_current_balance',
     'sec_disbursed_amount',
     'sec_instal_amt',
     'sec_no_of_accts',
     'sec_overdue_accts',
     'sec_sanctioned_amount',
     'state_id',
     'supplier_id',
     'voterid_flag'
    ]



y_vars = ['defaulted']

# =============================================================================
# DATA IMPORT AND PROCESSING
# =============================================================================

def import_csv(filename):
    df = pd.read_csv(filename, header=0) 
    df = df.rename(str.lower, axis='columns')
    return df



def data_format(df, log_transform):  
    #drop unnecessary variables
    df.drop(irrelevant_vars, axis = 'columns', inplace = True)
    
    
    #rename some var
    df.rename({'perform_cns_score': 'cns', 
               'credit_history_length': 'credit_his_len',
               'loan_default': 'defaulted'}, 
              axis = 1, 
              inplace = True)

    for var in large_num_vars:
         df[var] = df[var].transform(lambda x: x/(10**5))
       
    
    
    #transform date of birth to year of birth
    def dob_to_age(s):
        year = s.split(sep = '-')[-1]
        return 2018 - int(year)
    
    
    df['age'] = df['date_of_birth'].apply(dob_to_age)
    df.drop(['date_of_birth'], axis = 'columns', inplace = True)
    
    
    #encode employment into 0 and 1
    d_employed = {'Salaried': 1, 'Self employed': 0}
    df['employed'] = np.where(df['employment_type'] == 'Salaried', 1, 0)
    df.drop(['employment_type'], axis = 'columns', inplace = True)
    
    

    #credit history length
    #average account age
    #in months
    def time_in_month(s):
        L = s.split()
        y = int(L[0][:-3])
        m = int(L[1][:-3])
        return y*12 + m
    
    
    df['credit_his_len'] = df['credit_his_len'].apply(time_in_month)
    df['average_acct_age'] = df['average_acct_age'].apply(time_in_month)
    
    
    if log_transform:
        for var in log_vars:
            df[var] = df[var].transform(lambda x: log(x+1))
            
        
    
    
    return df



#to remove outliers based on a variable
def remove_outliers(df, var, q):
    threshold = df[var].quantile(q)
    outliers = df[(df[var] >= threshold)].index
    df.drop(outliers, inplace = True)

 


def rescaling(df, rescale, variables):
    #rescale if needed
    if rescale == 'standard': 
        transformer = StandardScaler() 
    elif rescale == 'minmax':
        transformer  = MinMaxScaler()
    elif rescale == 'robust':
        transformer = RobustScaler()
    elif rescale == 'quantile':
        transformer = QuantileTransformer()
             
    rescaled = transformer.fit_transform(df[variables])
    not_rescaled = df.drop(columns = variables)
    
    df = pd.concat([rescaled, not_rescaled], axis = 'columns')
    
    return df



def data_split_random(df, x_vars, y_vars, run_num):
    y = df[y_vars]
    x = df[x_vars]    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=0.4, 
                                                        random_state=run_num)
    return x_train, x_test, y_train, y_test





# =============================================================================
# STATS
# =============================================================================

def corr_mat_heatmap(df, continuous_vars):
    df_continuous = df[continuous_vars]
    
    #correlation matrix
    corr_mat = df_continuous.corr(method='pearson')
    corr_mat = corr_mat.round(2)
    corr_mat.to_csv('correlation_matrix.csv' ,sep=',', encoding='utf-8', index=True, header=True)

    #heat map
    my_cmap = sns.diverging_palette(240,240, as_cmap=True)
    
    plt.figure(figsize=(30, 20))
    ax = sns.heatmap(corr_mat, annot=True, cmap = my_cmap, vmin = -1, vmax = 1)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.xticks(rotation=70)
    plt.show()
    
    return corr_mat








# =============================================================================
# FOR VISUALISATION
# =============================================================================


def histogram(df, feature_name, bins, colour, print_folder):
    feature_data = df[feature_name]
    title = feature_name
    fig = plt.figure()
    plt.hist(feature_data, bins, alpha = 0.9, color = colour)
    plt.title(title)
    plt.show()
    
    if print_folder:
        fig_name = f"hist_{feature_name}.png"
        path = os.path.join(print_folder, fig_name)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    return


#assume grouped by a binary variable
def histogram_grouped(df, var, bins, group_by, print_folder):
    title = var
    df_temp = df[[var, group_by]]
    x0 = df_temp.loc[df[group_by] == 0][var]
    x1 = df_temp.loc[df[group_by] == 1][var]
    
    fig = plt.figure()
    plt.title(title)
    plt.hist(x0, bins, alpha=0.7, color = 'mediumturquoise', ec='darkgray', label = f'not {group_by}')
    plt.hist(x1, bins, alpha=0.8, color = 'red', ec='darkgray', label = f'{group_by}')
    plt.legend(loc='upper right')
    plt.show()
    
    if print_folder:
        fig_name = f"hist_{var}.png"
        path = os.path.join(print_folder, fig_name)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    
    return



def scatter(df, x_name, y_name):
    title = f'{x_name} and {y_name}'
    x = df[x_name]
    y = df[y_name]
    plt.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.show()
    
    
def histogram_vars(df, variables, bins, colour, group_by = False, print_folder = False):
    if group_by:
        for var in variables: histogram_grouped(df, var, bins, group_by, print_folder)
    
    else: 
        for var in variables: histogram(df, var, bins, colour, print_folder)    
    return 
    

def hist_cumulative(df, var, bins, colour):
    fig = plt.Figure()
    plt.title(f'cumulative histogram of {var}')
    plt.hist(df[var], bins, cumulative=True, color= colour, edgecolor='lightgray')
    plt.show()
    plt.close(fig)
    






def boxplot(df, categorical, continuous, print_folder):
    bp = sns.boxplot(data=df, x=categorical, y=continuous, hue= categorical, fill =  False, width=.4)
    bp_fig = bp.get_figure()  
    plt.show()
    
    if print_folder:
        path = os.path.join(print_folder, f'boxplot_{continuous}')
        bp_fig.savefig(path)
   
    plt.close(bp_fig)
    


def boxplot_vars(df, categorical, continuous_vars, print_folder = False):
    for var in continuous_vars:
        boxplot(df, categorical, var, print_folder)
    
 
  
# =============================================================================
# FOR CLASSIFICATION MODELS
# =============================================================================

#manual combination of over and under sampling
def over_under_sampler():
    over = RandomOverSampler(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    # define pipeline
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    
    return pipeline
    


    
def classification_performance(y_true, y_pred, all_indicators = False):
    
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred, average=None)   
    acc = roc_auc_score(y_true, y_pred)
    results = [auc, acc, f1]
    
    if all_indicators:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        results.extend([precision, recall])
    
    return tuple(results)


def draw_roc(y, y_pred_p):   
    lr_fpr, lr_tpr, _ = roc_curve(y, y_pred_p)
    #plot the roc curve for the model 
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic-model')
    #axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #show the legend
    plt.legend()
    #show the plot
    plt.show()
    plt.clf()

       