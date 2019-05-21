# -*- coding: utf-8 -*-
"""
Analysis performed for the bee medulla neuron model

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

##### 1. load the data
wave_x = np.linspace(300, 700, num=5)

## fitted model
raw_data = pd.read_excel('analytical weights.xlsx')
# fitted model responses
fit_data = raw_data.values[:,5::].astype(np.float64)
# fitted model weights (UV, blue, green)
fit_weights = raw_data.values[:, 3:6].astype(np.float64)

## real data
empirical_data = pd.read_excel('SI empirical data.xlsx', sheet_name='0')
## simulated 5500
simulate_data = pd.read_csv('SI simulated 5500 neurons.csv').values.astype(np.float64)

# data wrapper
class Database(object):
    pass

##### 2. functions
from sklearn.decomposition import PCA
def PCAwrapper(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    # dimension reduction
    pcaX = pca.transform(X)
    return pca.explained_variance_ratio_, pcaX

def scatter3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    array_len = np.shape(data)[0]
    for i in range(array_len):
        ax.scatter(data[i,0], data[i,1], data[i,2], c='tab:gray')
    ax.tick_params(labelsize=10)

# time-series Kmeans
import tslearn, sklearn
from tslearn.clustering import TimeSeriesKMeans
def tsKMeans_num_cluster(X, n_trials, max_n_cluster):
    min_n_cluster = 2
    v_clusters = np.arange(min_n_cluster, max_n_cluster)
    n_seeds = n_trials
        # recorder
    sc_recorder = np.zeros((len(v_clusters),n_seeds))
    for i_seed in range(n_seeds):
        for num_cluster in v_clusters:
            model=TimeSeriesKMeans(n_clusters = num_cluster, tol=1e-05, metric='euclidean',    random_state=i_seed)
            fitted_model = model.fit(X)
            y_pred= fitted_model.predict(X)
            s_sc = sklearn.metrics.silhouette_score(X, y_pred, metric='euclidean')
            sc_recorder[num_cluster-min_n_cluster, i_seed]=s_sc
    return sc_recorder

def tsKMeans_simple(X, n_cluster, random_state):
    model=TimeSeriesKMeans(n_clusters = n_cluster, tol=1e-05, metric='euclidean', random_state=random_state)
    fitted_model = model.fit(X)
    y_pred = fitted_model.predict(X)
    return y_pred

# dirichlet process gaussian mixture model
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import DPGMM
import math

def dpgmm_simple(X, init_numC, random_state):
    model = DPGMM(n_components = init_numC, n_iter=100, tol=0.000001, random_state=random_state)
    model.fit(X)
    y = model.predict(X)
    cluster_num = len(np.unique(y))
    return cluster_num, y

##### 3. plots
def multi_plot(row, col, fs_tuple, sy_bool, sx_bool, X, num_cluster, lineW, ts=False, labels = None):

    if ts==True:
        # model generation
        model=TimeSeriesKMeans(n_clusters = num_cluster, tol=1e-05, metric='euclidean')
        fitted_model = model.fit(X)
        labels = fitted_model.predict(X)

    f, axes = plt.subplots(row, col, figsize=fs_tuple, sharey=sy_bool, sharex=sx_bool)


    labelsize=10
    fontsize=10

    cluster_pool = np.unique(labels)
    for index, i_cluster in enumerate(cluster_pool):
        sub_mat = X[labels==i_cluster, :]
        # unravel
        figrow, figcol = np.unravel_index(index, dims=[row, col])
        # plot
        if row > 1 and col > 1:

            for iCurve in range(np.shape(sub_mat)[0]):
                axes[figrow,figcol].plot(sub_mat[iCurve,:], 'r', linewidth=lineW)
            # after plot, modify the axes
            for i_col in range(col):
                axes[-1,i_col].set_xticks([0,16,32,48,64,80])
                axes[-1,i_col].set_xticklabels(str(300+80*(i)) for i in np.arange(6))
                axes[-1,i_col].set_xlabel('Wavelength [nm]', fontsize=fontsize)
                axes[-1,i_col].tick_params(axis='x', labelsize=labelsize)
            for i_row in range(row):
                axes[i_row,0].set_yticks([-1,0,1])
                axes[i_row,0].tick_params(axis='y', labelsize=labelsize)
        elif row > 1 and col == 1:
            for i_curve in range(np.shape(sub_mat)[0]):
                axes[figrow].plot(sub_mat[i_curve,:], 'r', linewidth=lineW)
            axes[-1,0].set_xticks([0,16,32,48,64,80])
            axes[-1,0].set_xticklabels(str(300+80*(i)) for i in np.arange(6))
            axes[-1,0].set_xlabel('Wavelength [nm]', fontsize=fontsize)
            axes[-1,0].tick_params(axis='x', labelsize=labelsize)
            for i_row in range(row):
                axes[i_row,0].set_yticks([-1,0,1])
                axes[i_row,0].tick_params(axis='y', labelsize=labelsize)
        elif row == 1 and col > 1:
            for i_curve in range(np.shape(sub_mat)[0]):
                axes[figcol].plot(sub_mat[i_curve,:], 'r', linewidth=lineW)
            for i_col in range(col):
                axes[-1,i_col].set_xticks([0,16,32,48,64,80])
                axes[-1,i_col].set_xticklabels(str(300+80*(i)) for i in np.arange(6))
                axes[-1,i_col].set_xlabel('Wavelength [nm]', fontsize=fontsize)
                axes[-1,i_col].tick_params(axis='x', labelsize=labelsize)
            axes[0,0].set_yticks([-1,0,1])
            axes[0,0].tick_params(axis='y', labelsize=labelsize)
    return (f, axes)


if __name__=='__main__':
    database = Database()
    database.raw_data=raw_data; database.fit_data=fit_data; database.fit_weights=fit_weights
    database.empirical_data=empirical_data; database.simulate_data=simulate_data
    # PCA -> then cluster the PCA results
    database.variance_ratio, database.pcaWeights = PCAwrapper(database.fit_weights)
    # dpgmm for weights
    (wClusters, wlabels)=dpgmm_simple(database.pcaWeights, 22, 2)
    print('optimal number of clusters for PCA 2D weights = {}'.format(wClusters))
    #print('optimal number of clusters for response curves = {}'.format(cClusters))

    # 100 times on the 5500 neurons; dpgmm
    n_trials = 2
    n_clusters_sim = np.zeros((n_trials,))
    for i_trial in range(n_trials):
        (sClusters, slabels)=dpgmm_simple(database.simulate_data, 30, i_trial)
#        num_clusters_sim[i]=sClusters

    sc_recorder_sim = tsKMeans_num_cluster(database.simulate_data, n_trials, 30)
#    sns.tsplot(np.transpose(wSilh), ci='sd',color='k')
