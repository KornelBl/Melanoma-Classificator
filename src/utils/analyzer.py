import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def get_logs_pickle(dir_path, filename='logs.pyc') -> pd.DataFrame:
    file_name = os.path.join(dir_path, 'logs.pyc')
    logs = pd.read_pickle(file_name)

    for x in logs:
        logs[x] = logs[x].apply(lambda y: np.array(y))
    return logs


def get_means_logs(logs:pd.DataFrame) -> dict:
    means = dict()
    for x, y in logs.sum().items():
        means[x] = y / len(logs)
    return means

def get_means_limited(logs:pd.DataFrame,limit=None) -> dict:
    if limit is None:
        limit = logs["val_auc"].apply(len).min()
    means = dict()
    for x in logs:
        means[x] = np.zeros(limit)
        for y in logs[x]:
            means[x] += y[:limit]
        means[x] = means[x]/len(logs)
    return means


def get_mean_std_from_logs(logs:pd.DataFrame, metric="val_auc", get_best_func = np.max):
    mean = logs[metric].apply(get_best_func).mean()
    std = logs[metric].apply(get_best_func).std()
    return mean, std
