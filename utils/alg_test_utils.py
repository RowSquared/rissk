import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import metrics
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt

METRIC_SCORES = [metrics.rand_score, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score]
CLUSTER_CLASSES = [cluster.MiniBatchKMeans, cluster.SpectralClustering, cluster.AgglomerativeClustering, cluster.DBSCAN]
RANDOM_STATE = 321
N_CLUSTERS = 2


def get_cluster_args(func_name) -> dict:
    if func_name == 'cluster.MiniBatchKMeans':
        return {'n_clusters': N_CLUSTERS, 'random_state': RANDOM_STATE, 'n_init': 1}
    elif func_name == 'cluster.SpectralClustering':
        return {'n_clusters': N_CLUSTERS, 'random_state': RANDOM_STATE, 'n_init': 1, 'n_jobs': 6}
    elif func_name == 'cluster.AgglomerativeClustering':
        return {'n_clusters': N_CLUSTERS, 'random_state': RANDOM_STATE}
    elif func_name == 'cluster.DBSCAN':
        return {'eps': 0.5, 'min_samples': 5, 'random_state': RANDOM_STATE, 'n_jobs': 6}
    else:
        raise ValueError(f'Unknown cluster method: {func_name}')


def calc_pair_metrics(labels_a: list, labels_b: list) -> tuple:
    metric_scores = [metrics.rand_score, metrics.mutual_info_score, metrics.homogeneity_score,
                     metrics.completeness_score]

    res = [f(labels_a, labels_b) for f in metric_scores]
    return tuple(res)


def calc_cluster(data: Union[pd.DataFrame, np.ndarray], cluster_class):
    assert cluster_class in CLUSTER_CLASSES, f'Unknown cluster method: {cluster_class}'
    kwargs = get_cluster_args(cluster_class.__name__)
    cls = cluster_class(**kwargs)
    cls.fit(data)
    return cls


def get_param_path() -> str:
    return '../data'


def save_fitted_cluster(fitted_cluster_class, path: str, suffix: str) -> None:
    parameters = fitted_cluster_class.get_params()
    with open(os.path.join(path, f'{fitted_cluster_class.__name__}_{suffix}.pickle'), 'wb') as f:
        pickle.dump(parameters, f)


def plot_data_pca(data: np.ndarray, n_components: int, unit_scale: bool = True, pc_start: int = 0, pc_end: int = 1):
    pca = decomposition.PCA(n_components=n_components)
    if unit_scale:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
        data = scaler.fit_transform(data)
    pca.fit(data)
    data_pca = pca.transform(data)
    assert pc_start < pc_end, 'pc_start > pc_end'
    for i in range(pc_start, pc_end):
        plt.scatter(data_pca[:, i], data_pca[:, i + 1])
        plt.show()
