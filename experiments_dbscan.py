import csv
import os
import sys
import time
from datetime import datetime

import tqdm
import numpy as np
import pandas as pd

import utils.constants as c
import cf_algorithms.similarity as s
from utils import rectools as r
from utils import datatools as d
from opfython.models.unsupervised_snn import UnsupervisedSnnOPF as OPFSNN
from cf_algorithms.cf_dbscan import DBSCAN
from sklearn.preprocessing import LabelEncoder

test_perc = 0.1


def similarity_function(x, y, alpha=1.):

    _dist = abs(x.cost - y.cost) / 1000.0
    return (1. - _dist) ** alpha


def get_datetime(str_format=None):

    if str_format is None:
        str_format = "%b-%d-%Y_%H-%M-%S-%f"

    current_date = datetime.now()
    return current_date.strftime(str_format).lower()


def main(n_neighbors: int, density='pdf', distance='jaccard', similarity='pearson',
         custom_state=None, dataset=None):

    print("RUNNING DBSCAN...")
    print("  --> Distance measure:", distance)
    print("  --> Similarity measure:", similarity)
    print('  --> Dataset:', dataset)

    output = [custom_state]

    _df, dim = d.get_dataset(dataset)

    # Creates CF-based dataset
    R, X, user_mapping, item_mapping = r.create_cf_matrix(_df, size=dim)

    # Split data for CF-based evaluation
    idx_train, idx_val, X_test = r.train_test_split_sg(
        X, user_size=test_perc, item_size=test_perc, custom_state=custom_state)

    # Set training and evaluation sets
    X_train, X_val = R.toarray()[idx_train], R.toarray()[idx_val]

    del R

    # for i, u in enumerate(idx_val):
    #
    #     items = X_test.get(u)
    #     X_val[i, items] = 0.

    # Clustering with OPF-SNN to optimize DBSCAN later
    opf = OPFSNN(max_k=n_neighbors, distance=distance, density_computation=density)

    # Computing clusters by OPF-SNN
    _, clusters = opf.fit_predict(X_train, I_train=idx_train)

    print("Optimizing DBSCAN...")

    radius = [node.radius for node in opf.subgraph.nodes]

    n_clusters = len(np.unique(clusters))
    # eps = best_eps = opf.subgraph.density
    best_eps = np.max(radius)
    # min_size = 20
    best_size = 5
    min_dist = 10000.0
    for ms in range(10, int(np.max(np.bincount(clusters)) + 1), 10):

        eps = np.max(radius)
        print(f"Eps: {eps:.2f}\t Min samples: {ms}")

        # Finding a suitable eps value
        while eps > np.min(radius):

            tmp_dbscan = DBSCAN(eps, min_samples=ms, distance_function=distance,
                                custom_state=custom_state, verbose=False)
            tmp_dbscan.fit(X_train)

            le = LabelEncoder()
            encoded_pred = le.fit_transform(tmp_dbscan.labels)

            labels_dist = np.abs(n_clusters - len(np.unique(encoded_pred)))

            if labels_dist < min_dist:
                min_dist = labels_dist
                best_eps = eps
                best_size = ms

            if labels_dist == 0:
                break

            eps -= 0.2

        print("..", end="")

    print("\nOptimal eps:", best_eps)
    print("Optimal min_samples:", best_size)

    # Computing clusters by DBSCAN
    dbscan = DBSCAN(best_eps, best_size, distance_function=dist, custom_state=custom_state)
    raw_clusters = dbscan.fit_predict(X_train)

    le = LabelEncoder()
    encoded_clusters = le.fit_transform(raw_clusters)

    # Predicting clusters for test users
    preds = dbscan.predict(X_val)
    encoded_preds = le.transform(preds)

    output.extend([best_eps, best_size, len(np.unique(encoded_clusters)), distance, similarity])

    print("DBSCAN clusters: \n", np.bincount(encoded_clusters))

    # Mapping (dictionary) users by their corresponding assigned cluster
    clusters = []
    for i in np.unique(encoded_clusters):
        tmp = [j for j in range(X_train.shape[0]) if encoded_clusters[j] == i]
        clusters.append(tmp)

    # Recommendation stage
    # Choose the similarity function
    sim_fn = s.SIMILARITIES.get(similarity)

    # Choose the aggregation function to recommendation with
    aggr_fn = r.aggregation_functions.get('mean_centered')

    print("DBSCAN - Predicting ratings for test users...")

    start = time.time()

    X_pred = {}

    # For each test user predict his/her list of test items
    for z, value in tqdm.tqdm(enumerate(X_test.items()), total=len(X_test.keys())):
        # Unpack user inner id and list of items (inner ids)
        user, items = value

        # Cluster of test sample z
        idx_c = int(encoded_preds[z])

        neighbors = [idx_train[j] for j in clusters[idx_c]]

        # Compute similarities from user cluster (neighborhood)
        W = [sim_fn(X_val[z], X_train[j]) for j in clusters[idx_c]]
        W = np.array(W)

        # Assign predicted ratings for the active test user
        X_pred[user] = [aggr_fn(X, user, i, neighbors, W) for i in items]

    end = time.time()

    print(f"Recommendation time: {end - start:.2f} seconds")

    # top_k = X_train.shape[1]
    # for _, items in X_test.items():
    #     top_k = np.minimum(top_k, len(items))

    # For each test user, get his/her list of true ratings 
    X_true = r.get_true_ratings(X, X_test)

    # For each test user set a relevance threshold
    X_rel = r.get_average_ratings(X, X_test, baseline=4.0)

    # Prediction quality
    mae, rmse = r.evaluate_prediction_task(X_true, X_pred, verbose=True)

    output.extend([mae, rmse])

    # Ranking quality
    for k in [1, 5, 10]:
        ndcg_at_k = r.evaluate_ranking_task(X_true, X_pred, X_rel, k=k, verbose=True)
        output.append(ndcg_at_k)

    # Recommendation quality
    precision = []
    recall = []
    for k in [1, 5, 10]:
        precision_at_k, recall_at_k = r.evaluate_recommendation_quality(X_true, X_pred, X_rel, k=k, verbose=True)
        precision.append(precision_at_k)
        recall.append(recall_at_k)

    output.extend(precision)
    output.extend(recall)

    return output


if __name__ == '__main__':

    save = True
    # dist = ['cosine']
    dist = []
    sim_func = 'pearson'
    dens = 'pdf'

    arg_dataset = str(sys.argv[1])
    arg_first_run = int(sys.argv[2])
    arg_n_runs = int(sys.argv[3])
    arg_kmax = int(sys.argv[4])
    arg_seed = int(sys.argv[5])

    # Set an instance of random state generator (rg)
    rg = np.random.RandomState(arg_seed)
    random_states = rg.randint(1, 1e5, arg_n_runs + arg_first_run)

    data_dir = os.getcwd()
    kmax_interval = [arg_kmax] if arg_kmax != 0 else c.KMAX_INTERVAL
    distances = dist if len(dist) > 0 else c.DISTANCE_FUNCTIONS

    for it in range(arg_first_run, arg_n_runs + arg_first_run):

        print("Running experiment ", it, "...")

        cur_path = "/".join([data_dir, "out", arg_dataset, "dbscan",
                             f"{it}_{random_states[it]}_{arg_dataset}.csv"])

        print("Current output file path:", cur_path)

        if save:
            if not os.path.exists(cur_path):
                # Initializes pandas dataframe with pre-defined column header
                df = pd.DataFrame(columns=c.HEADER_DBSCAN_CSV)

                # Saves dataframe to file
                df.to_csv(cur_path)

        for k_max in kmax_interval:

            df = pd.DataFrame(columns=c.HEADER_DBSCAN_CSV)

            for dist in distances:

                # Runs the experiment itself
                cur_output = main(k_max, dens, distance=dist, similarity=sim_func,
                                  custom_state=random_states[it], dataset=arg_dataset)

                if save:
                    # Append a new row of data to the end of dataframe
                    df.loc[len(df)] = cur_output

                    # Update the existing csv file
                    df.to_csv(cur_path, mode='a', header=False, na_rep='NA', quoting=csv.QUOTE_NONE)

                    print("Current experiment saved to: '", cur_path, "'...")
