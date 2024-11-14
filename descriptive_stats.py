# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:55:54 2024

@author: dthtr
"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from math import log, log10, log1p



#my own module (for functions)
import myFuncs as mf

from myFuncs import continuous_vars


data_path = 'data\\train.csv'


#_____IMPORT AND FORMAT DATA______________________
df = mf.import_csv(data_path)

df = mf.data_format(df, log_transform= False)

#_____DESCRIPTIVE STATS__________________________

#for the whole dataset

df.describe().round(3).transpose().to_csv('descriptive_stats.csv', sep=',', encoding='utf-8', index=True, header=True)

#for defaulted 
descriptive_defaulted = df.loc[(df['defaulted'] == 1)].describe().round(3)
descriptive_defaulted.transpose().to_csv('descriptive_defaulted.csv', sep=',', encoding='utf-8', index=True, header=True)

#for not defaulted
descriptive_not_defaulted = df.loc[(df['defaulted'] == 0)].describe().round(3)
descriptive_not_defaulted.transpose().to_csv('descriptive_not_defaulted.csv', sep=',', encoding='utf-8', index=True, header=True)





#_________CORRELATION MATRIX_____________
corr_mat = mf.corr_mat_heatmap(df, continuous_vars)
L = [(corr_mat[continuous_vars[i]][continuous_vars[j]], continuous_vars[i], continuous_vars[j]) for i in range(0, len(continuous_vars)) for j in range(0, i) ]
L.sort(key = lambda x: abs(x[0]), reverse = True)
L_corr_df = pd.DataFrame(L, columns = ['corr', 'var1', 'var2'])
L_corr_df.to_csv('corr_sorted.csv', sep= ',', encoding='utf-8',index = False, header= True)



L_high = [i for i in L if abs(i[0]) >= 0.4]
L_high_corr_vars = []
for item in L_high:
    L_high_corr_vars.append(item[1])
    L_high_corr_vars.append(item[2])

L_high_corr_vars = sorted(set(L_high_corr_vars))



#_____VISUALISATION_________________

mf.histogram_vars(df, continuous_vars, bins = 100, colour= 'blue', group_by= False, print_folder = 'figures\\overall_no_log')

mf.histogram_vars(df, continuous_vars, bins = 100, colour = False, group_by= 'defaulted', print_folder = 'figures\\grouped_no_log')


mf.histogram_vars(df.loc[(df['defaulted'] == 1)], continuous_vars, bins = 100, colour = 'orangered', group_by= False, print_folder = 'figures\\grouped_no_log\\defaulted')
mf.histogram_vars(df.loc[(df['defaulted'] == 0)], continuous_vars, bins = 100, colour = 'mediumturquoise', group_by= False, print_folder = 'figures\\grouped_no_log\\not_defaulted')


mf.histogram_grouped(df, 'disbursed_amount' , bins = 100, group_by = 'defaulted', print_folder=False)


L_boxplot = [
     'disbursed_amount',
     'ltv',
     'pri_current_balance',
     'pri_disbursed_amount',
     'pri_sanctioned_amount',
     'primary_instal_amt',
     'sec_current_balance',
     'sec_disbursed_amount',
     'sec_instal_amt',
     'sec_sanctioned_amount'
     ]


mf.boxplot_vars(df, 'defaulted', L_boxplot)



#________CHECK OUTLIERS_______________

#out of all variables
#pri_current_balance is one of the most skewed
#also, individuals at the extreme end of pri_current_balance
#do not default 
#so may be remove them from dataset

#first, check the cumulative histogram and some quantile

# Plot the cumulative histogram

df_defaulted = df.loc[(df['defaulted'] == 1)]
df_notdefaulted = df.loc[(df['defaulted'] == 0)]

mf.hist_cumulative(df, 'pri_current_balance', bins = 200, colour = 'skyblue')
mf.hist_cumulative(df_defaulted, 'pri_current_balance', bins = 200, colour = 'skyblue')
mf.hist_cumulative(df_notdefaulted, 'pri_current_balance', bins = 200, colour = 'skyblue')

#may be remove the top 0.001 (0.1 percent)
df['pri_current_balance'].quantile(0.99995)
df_defaulted['pri_current_balance'].quantile(0.999)
df_defaulted['pri_current_balance'].quantile(0.9999)


df.loc[df['pri_current_balance'] > 350]['pri_current_balance']
df_defaulted.loc[df['pri_current_balance'] > 350]['pri_current_balance']
df_notdefaulted.loc[df['pri_current_balance'] > 650]['pri_current_balance']




#__________NOW SOME OUTLIERS REMOVED, AND WITH LOG TRANSFORM_____
dfnew = mf.import_csv(train_data_path)
mf.remove_outliers(dfnew, 'pri_current_balance', q = 0.99)

dfnew = mf.data_format(dfnew, log_transform = True)


mf.histogram(dfnew, 'pri_current_balance', 100, colour = 'skyblue', print_folder = False)





