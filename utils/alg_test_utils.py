import os
import pickle
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import decomposition
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

METRIC_SCORES = [metrics.rand_score, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score]
CLUSTER_CLASSES = [cluster.MiniBatchKMeans, cluster.SpectralClustering, cluster.AgglomerativeClustering, cluster.DBSCAN]
RANDOM_STATE = 321
N_CLUSTERS = 2


def drop_all_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all').dropna(axis=0, how='any')


def _dbscan_find_optimal_minpts(data: np.ndarray) -> int:
    return data.shape[1] + 1


def _dbscan_find_optimal_eps(data: np.ndarray, minpts: int):
    arr = neighbors.kneighbors_graph(data, n_neighbors=minpts - 1, mode='distance').toarray()
    arr = -1 * np.sort(-1 * arr, axis=1)
    arr = arr[:, :minpts + 1]
    # plt.plot(np.median(arr, axis=0))

    return 0.3  # todo value only valid for tunisia data scaled to 0, 1


def get_cluster_args(func_name, data: Union[pd.DataFrame, np.ndarray]) -> dict:
    if func_name == 'MiniBatchKMeans':
        return {'n_clusters': N_CLUSTERS, 'random_state': RANDOM_STATE, 'n_init': 1}
    elif func_name == 'SpectralClustering':
        return {'n_clusters': N_CLUSTERS, 'random_state': RANDOM_STATE, 'n_init': 1, 'n_jobs': 6}
    elif func_name == 'AgglomerativeClustering':
        return {'n_clusters': N_CLUSTERS}
    elif func_name == 'DBSCAN':
        min_samples = _dbscan_find_optimal_minpts(data)
        eps = _dbscan_find_optimal_eps(data, min_samples)
        return {'eps': eps, 'min_samples': min_samples, 'n_jobs': 6}
    else:
        raise ValueError(f'Unknown cluster method: {func_name}')


def calc_pair_metrics(labels_a: list, labels_b: list) -> tuple:
    metric_scores = METRIC_SCORES

    res = [f(labels_a, labels_b) for f in metric_scores]
    return tuple(res)


def fit_cluster(data: Union[pd.DataFrame, np.ndarray], cluster_class):
    assert cluster_class.__name__ in [x.__name__ for x in CLUSTER_CLASSES], f'Unknown cluster method: {cluster_class}'
    kwargs = get_cluster_args(cluster_class.__name__, data)
    cls = cluster_class(**kwargs)
    cls.fit(data)
    return cls




def save_fitted_cluster(fitted_cluster_class, path: str, suffix: str) -> None:
    parameters = fitted_cluster_class.get_params()
    with open(os.path.join(path, f'{fitted_cluster_class.__name__}_{suffix}.pickle'), 'wb') as f:
        pickle.dump(parameters, f)


def scale_data(data: np.ndarray):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)


def pca(data: np.ndarray, n_components: int, unit_scale: bool = False):
    pca = decomposition.PCA(n_components=n_components)

    if unit_scale:
        data = scale_data(data)

    pca.fit(data)

    return pca.transform(data)


def plot_data(data: np.ndarray, labels: Union[list, np.ndarray] = None, dim_start: int = 0, dim_end: int = 1):
    assert dim_start < dim_end, 'dim_start > dim_end'
    for i in range(dim_start, dim_end):
        plt.scatter(data[:, i], data[:, i + 1], c=labels)
        #plt.legend()
        plt.show()


def plot_data_pca(data: np.ndarray, n_components: int, labels: Union[list, np.ndarray] = None,  unit_scale: bool = False, pc_start: int = 0, pc_end: int = 1):
    data = pca(data, n_components, unit_scale)

    plot_data(data, labels=labels, dim_start=pc_start, dim_end=pc_end)


def transform_data(data: np.ndarray, random_state: np.random.RandomState = None, column_subset: list = None, perturbation: bool = False) -> np.ndarray:
    res_data = np.copy(data)

    # select column subset if necessary
    if column_subset:
        res_data = res_data[:, column_subset]

    # add random noise to columns of data
    if perturbation:
        res_data = res_data + np.random.normal(0, 0.1, size=res_data.shape)

    return res_data


