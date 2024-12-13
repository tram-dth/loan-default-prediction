import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


#my own module
import myFuncs as mf
from myFuncs import continuous_vars

data_path = 'data\\train.csv'


def data(data_path, log_transform, remove_outliers_var):
    df = mf.import_csv(data_path)
    
    if remove_outliers_var:
        mf.remove_outliers(df, remove_outliers_var , q = 0.999)
    
    df = mf.data_format(df, log_transform)
    
    df = mf.rescaling(df, 'standard', continuous_vars)
    
    return df


 

def trying_out(data_path, continuous_vars, k, log_transform, remove_outliers_var):
    df = data(data_path, log_transform, remove_outliers_var)
    df_continuous = df[continuous_vars]
    pca = PCA(n_components = len(continuous_vars))  # Set n_components to the number of variables (40 in this case)
    pca.fit(df_continuous)
    
    print(f'try log_transform = {log_transform}, and remove_outliers_var = {remove_outliers_var} ')
    print(f'total explained variance ratio of first {k} components : ', pca.explained_variance_ratio_[:15].sum())
    
    return pca




def plotting_cumsum_explained_variance():
    #trying out
    pca1 = trying_out(data_path, continuous_vars, 15, log_transform = False, remove_outliers_var = False)
    pca2 = trying_out(data_path, continuous_vars, 15, log_transform = True, remove_outliers_var = False)
    pca3 = trying_out(data_path, continuous_vars, 15, log_transform = True, remove_outliers_var = 'pri_current_balance')

    p1, = plt.plot(np.cumsum(pca1.explained_variance_ratio_), color = 'blue')
    p2, = plt.plot(np.cumsum(pca2.explained_variance_ratio_), color = 'green')
    p3, = plt.plot(np.cumsum(pca3.explained_variance_ratio_), color = 'red')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance ratio')
    plt.legend([p1, p2, p3], ['no change', 'log transformed', 'log transformed and outliers removed'])



#______FOR OTHER FILES_______________________
def PCA_reduced(df, continuous_vars):
    pca_reduced = PCA(n_components= 15)
    pca_reduced.fit(df[continuous_vars])

    df_continuous_reduced = pca_reduced.fit_transform(df[continuous_vars])
    df_categorical = df.drop(columns = continuous_vars)
    
    dfnew = pd.concat([df_continuous_reduced, df_categorical], axis = 'columns')
    
    return dfnew





#______FOR VISUALISATION_______

