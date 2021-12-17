import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid


def generate_dataset(path):
    """Generate a single dataset starting from different files in a directory"""
    dim_vector = 30
    dataset = np.empty((1,dim_vector))

    for file in os.listdir(path):
        name = os.path.join(path, file)
        data = np.loadtxt(name, delimiter=',', skiprows=1)
        dataset = np.concatenate((dataset, data))

    # remove first row
    dataset = dataset[1:,:]

    return dataset


def find_best_param(X):

    # DBSCAN is used as clustering tecnique
    param_grid = {'eps': list(np.arange(0.1, 0.4, 0.01)), 'min_samples': list(range(1, 10, 1))}
    params = list(ParameterGrid(param_grid))
    sil_thr = 0  # visualize results only for combinations with silhouette above the threshold
    unc_thr = 10  # visualize results only for combinations with unclustered% below the threshold

    print("{:11}\t{:11}\t{:11}\t{:11}\t{:11}". \
          format('        eps', 'min_samples', ' n_clusters', ' silhouette', '    unclust%'))
    for i in range(len(params)):
        db = DBSCAN(**(params[i]))
        y_db = db.fit_predict(X)
        cluster_labels_all = np.unique(y_db)
        cluster_labels = cluster_labels_all[cluster_labels_all != -1]
        n_clusters = len(cluster_labels)
        if n_clusters > 1:
            X_cl = X[y_db != -1, :]
            y_db_cl = y_db[y_db != -1]
            silhouette = silhouette_score(X_cl, y_db_cl)
            uncl_p = (1 - y_db_cl.shape[0] / y_db.shape[0]) * 100
            if silhouette > sil_thr and uncl_p < unc_thr:
                print("{:11.2f}\t{:11}\t{:11}\t{:11.2f}\t{:11.2f}%" \
                      .format(db.eps, db.min_samples, n_clusters, silhouette, uncl_p))


def main():

    path = "C:/Users/boezi/PycharmProjects/Pommerman/causal/patches"
    X = generate_dataset(path)
    print(f"Number of data points: {X.shape[0]}")
    find_best_param(X)


if __name__ == '__main__':
    main()