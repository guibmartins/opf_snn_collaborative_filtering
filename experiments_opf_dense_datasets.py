import csv
import datetime
import os
import sys
import numpy as np
import pandas as pd

import utils.constants as c
import utils.datatools as d

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

from opfython.models.unsupervised_snn import UnsupervisedAnnOPF, UnsupervisedSnnOPF as OPFSNN
from opfython.models.unsupervised import UnsupervisedOPF
from opfython.models.ann_unsupervised import ANNUnsupervisedOPF
from opfython.models.unsupervised import ClusteringOPF

# Global variable 'dataset_name'
dataset_name = None


def evaluate_model(X, y_true, y_pred, clusters=None, seed=1):

    db, ss = np.nan, np.nan

    if dataset_name in c.NO_LABELS_DATASET:

        if clusters is None:
            if len(np.unique(y_pred)) > 1:
                db = davies_bouldin_score(X, y_pred)
                ss = silhouette_score(X, y_pred)

        else:
            if len(np.unique(clusters)) > 1:
                db = davies_bouldin_score(X, clusters)
                ss = silhouette_score(X, clusters)

        hg = cp = vm = np.nan
        ari = ami = np.nan

        print(f"DB-index: {db:.4f}")
        print(f"Silhouette score: {ss:.4f}")

    else:
        hg, cp, vm = homogeneity_completeness_v_measure(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)

        print(f"Homogeneity: {hg:.4f}")
        print(f"Completeness: {cp:.4f}")
        print(f"V-measure: {vm:.4f}")

        print(f"Adjusted Rand-Index: {ari:.4f}")
        print(f"Adjusted Mutual Information: {ami:.4f}\n\n")

    return {'db_index': db,
            'silhouette_score': ss,
            'homogeneity': hg,
            'completeness': cp,
            'v_measure': vm,
            'adjusted_rand_index': ari,
            'adjusted_mutual_info': ami}


def run_opf_snn_with_ann(k, seed=1, test_perc=.2):

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    # scaler = StandardScaler()
    # X_norm = scaler.fit_transform(X)
    X_norm = X[indices, :]
    y_norm = y[indices]

    prm = {'name': 'kdtree', 'leaf_size': 30, 'seed': seed}
    # prm = {'name': 'hnsw', 'M': 16, 'C': 20, 'seed': seed, 'n_jobs': 1}

    opf = UnsupervisedAnnOPF(max_k=k, ann_params=prm)
    clusters = opf.fit(X_norm)

    print(np.bincount(clusters))

    evaluate_model(X_norm, y_norm, clusters, seed)

    return opf.subgraph.n_clusters


def run_opf_snn(k, dens='pdf', dist='euclidean', seed=1, test_perc=.2):

    df_row = {'timestamp': datetime.datetime.now().timestamp(),
              'random_state': seed,
              'algorithm': 'opf_snn',
              'k_max': k}

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    print("Dataset size: ", X.shape)

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    train_idx = indices[:h]
    val_idx = indices[h:]

    if dataset_name in ['google_reviews', 'ml100k', 'ml1m']:
        # Standardize input data
        scaler = StandardScaler()

        # Selecting training and validation subsets
        X_train = scaler.fit_transform(X[train_idx, :])
        X_val = scaler.transform(X[val_idx, :])

    else:
        X_train, X_val = X[train_idx, :], X[val_idx, :]

    # Selecting training and validation labels
    y_train, y_val = y[train_idx], y[val_idx]

    opf = OPFSNN(max_k=k, distance=dist, density_computation=dens)

    if dataset_name in c.NO_LABELS_DATASET:
        opf.fit(X_train, None)

    else:
        opf.fit(X_train, y_train)
        opf.propagate_labels()

    # print("Max distance (possible):", opf.subgraph.density)

    df_row['k_best'] = opf.subgraph.best_k
    df_row['n_clusters'] = opf.subgraph.n_clusters

    # Creating the list of clusters
    clusters = [node.cluster_label for node in opf.subgraph.nodes]
    print("Samples / cluster:", np.bincount(clusters))

    # Creating the list of clusters
    preds_train = [node.predicted_label for node in opf.subgraph.nodes]
    print("Samples / class:", np.bincount(preds_train))

    # Predicting clusters for test samples
    preds, pred_clusters, _ = opf.predict(X_val)

    if dataset_name in c.NO_LABELS_DATASET:

        db = silhouette_score(X_train, clusters)
        print(f"Silhouette score (training): {db:.4f}")

        print("Predictions / cluster:", np.bincount(pred_clusters))
        metrics = evaluate_model(X_val, y_val, preds, pred_clusters, seed)
    else:

        db = silhouette_score(X_train, preds_train)
        print(f"Silhouette score (training): {db:.4f}")

        print("Predictions / class:", np.bincount(preds))
        metrics = evaluate_model(X_val, y_val, preds, None, seed)

    # Adding result metrics to the dictionary
    df_row.update(metrics)

    return df_row


# Traditional unsupervised OPF
def run_opf_baseline(k, seed=1, dist='euclidean', test_perc=.2):

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    train_idx = indices[:h]
    val_idx = indices[h:]

    X_train, X_val = X[train_idx, :], X[val_idx, :]
    y_train, y_val = y[train_idx], y[val_idx]

    opf = UnsupervisedOPF(max_k=k, distance=dist)
    opf.fit(X_train, y_train)

    opf.propagate_labels()

    clusters = [node.cluster_label for node in opf.subgraph.nodes]
    print("Samples / cluster:", np.bincount(clusters))

    # Creating the list of clusters
    preds = [node.predicted_label for node in opf.subgraph.nodes]
    print("Samples / class:", np.bincount(preds))

    # evaluate_model(X_train, y_train, preds, None, seed)
    preds, _, _ = opf.predict(X_val)
    print("Predictions / cluster:", np.bincount(preds))
    evaluate_model(X_val, y_val, preds, None, seed)


# Unsupervised OPF with ANN graph
def run_opf_ann_baseline(k, seed=1, test_perc=.2):

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    # scaler = StandardScaler()
    # X_norm = scaler.fit_transform(X)
    X_norm = X[indices, :]
    y_norm = y[indices]

    opf = ANNUnsupervisedOPF(max_k=k, ann_params={'name': 'kdtree'})
    opf.fit(X_norm)

    # Creating the list of clusters
    clusters = [node.cluster_label for node in opf.subgraph.nodes]

    print(clusters)
    print("Samples per cluster:", np.bincount(clusters))

    evaluate_model(X_norm, y_norm, clusters, seed)


# Conventional k-means clustering
def run_kmeans_baseline(k, seed=1, test_perc=.2):

    print('\nRunning k-means...')

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    train_idx = indices[:h]
    val_idx = indices[h:]

    if dataset_name in ['google_reviews', 'ml100k', 'ml1m']:
        # Standardize input data
        scaler = StandardScaler()

        # Selecting training and validation subsets
        X_train = scaler.fit_transform(X[train_idx, :])
        X_val = scaler.transform(X[val_idx, :])

    else:
        X_train, X_val = X[train_idx, :], X[val_idx, :]

    y_train, y_val = y[train_idx], y[val_idx]

    kmeans = KMeans(k, init='random', n_init=1, max_iter=100, random_state=seed)
    preds = kmeans.fit_predict(X_train)
    print("Samples / cluster:", np.bincount(preds))

    # evaluate_model(X_train, y_train, preds, None, seed)
    preds = kmeans.predict(X_val)
    print("Predictions / cluster:", np.bincount(preds))
    evaluate_model(X_val, y_val, preds, None, seed)


# Alternative implementation of unsupervised OPF (more efficient)
def run_opf_fast_baseline(k, seed=1, dist='euclidean', test_perc=.2):

    df_row = {'timestamp': datetime.datetime.now().timestamp(),
              'random_state': seed,
              'algorithm': 'opf',
              'k_max': k}

    # Chooses the dataset by key and assigns data and labels X and y, respectively
    X, y = d.switch(dataset_name)()

    print("Dataset size: ", X.shape)

    # np.random.seed(seed)
    randg = np.random.RandomState(seed)

    indices = list(range(X.shape[0]))
    randg.shuffle(indices)
    h = int(len(indices) * (1 - test_perc))

    train_idx = indices[:h]
    val_idx = indices[h:]

    if dataset_name in ['google_reviews', 'ml100k', 'ml1m']:

        # Standardize input data
        scaler = StandardScaler()

        # Selecting training and validation subsets
        X_train = scaler.fit_transform(X[train_idx, :])
        X_val = scaler.transform(X[val_idx, :])

    else:

        X_train, X_val = X[train_idx, :], X[val_idx, :]

    y_train, y_val = y[train_idx], y[val_idx]

    opf = ClusteringOPF(max_k=k, distance=dist)

    if dataset_name in c.NO_LABELS_DATASET:
        opf.fit(X_train, None)

    else:
        opf.fit(X_train, y_train)
        opf.propagate_labels()

    df_row['k_best'] = opf.subgraph.best_k
    df_row['n_clusters'] = opf.subgraph.n_clusters

    clusters = [node.cluster_label for node in opf.subgraph.nodes]
    print("Samples / cluster:", np.bincount(clusters))

    # Creating the list of clusters
    preds_train = [node.predicted_label for node in opf.subgraph.nodes]
    print("Samples / class:", np.bincount(preds_train))

    # Predicting clusters for test samples
    preds, pred_clusters, _ = opf.predict(X_val)

    db = np.nan
    n_pred_labels = np.bincount(preds_train)
    if dataset_name in c.NO_LABELS_DATASET:

        if len(n_pred_labels) > 1:
            db = silhouette_score(X_train, clusters)

        print(f"Silhouette score (training): {db:.4f}")

        print("Predictions / cluster:", np.bincount(pred_clusters))
        metrics = evaluate_model(X_val, y_val, preds, pred_clusters, seed)
    else:

        if len(n_pred_labels) > 1:
            db = silhouette_score(X_train, preds_train)
            
        print(f"Silhouette score (training): {db:.4f}")

        print("Predictions / class:", np.bincount(preds))
        metrics = evaluate_model(X_val, y_val, preds, None, seed)

    # Adding result metrics to the dictionary
    df_row.update(metrics)

    return df_row


if __name__ == '__main__':

    # Output results directory
    output_dir = "out_dense"
    distance = 'squared_euclidean'

    arg_dataset = str(sys.argv[1])
    arg_first_run = int(sys.argv[2])
    arg_n_runs = int(sys.argv[3])
    arg_kmax = int(sys.argv[4])
    arg_seed_state = int(sys.argv[5])
    arg_to_file = int(sys.argv[6])

    dataset_name = arg_dataset

    print("Dataset: ", dataset_name)

    current_path = ""

    if arg_to_file != -1:

        # Set the project's directory
        data_dir = os.getcwd()

        current_path = "/".join([data_dir, output_dir, f"{dataset_name}_results.csv"])

        if not os.path.exists(current_path):
            # Initializes pandas dataframe with pre-defined column header
            df = pd.DataFrame(columns=c.HEADER_DENSE_DATA_RESULTS)

            # Saves dataframe to file
            df.to_csv(current_path)

    # df = pd.read_csv(current_path, header=0)

    # Sets an instance of random state generator (rg)
    rg = np.random.RandomState(arg_seed_state)
    random_states = rg.randint(1, 1e5, arg_n_runs + arg_first_run)

    if arg_kmax != 0:
        k_range = [arg_kmax]
    else:
        k_range = [i for i in range(5, 55, 5)]

    for it in range(arg_first_run, arg_n_runs + arg_first_run):

        print("Running experiment: ", it, " with random state = ", random_states[it], "...")

        for _k in k_range:

            # Initializes temporary dataframe
            df = pd.DataFrame(columns=c.HEADER_DENSE_DATA_RESULTS)

            # Runs the traditional unsupervised OPF numpy-optimized
            df.loc[len(df)] = run_opf_fast_baseline(_k, dist=distance, seed=random_states[it])

            # Runs the proposed clustering approach named OPF-SNN
            df.loc[len(df)] = run_opf_snn(_k, dist=distance, seed=random_states[it], dens='pdf')

            if arg_to_file != -1:

                print('Saving results to file...')
                # Saving current experiments to the end of the results .csv file
                df.to_csv(current_path, na_rep='NA', mode='a', header=False, quoting=csv.QUOTE_NONE)
