import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import tqdm

import cf_algorithms.similarity as s
from utils import constants as c
from utils import datatools as d
from utils import rectools as r
from cf_algorithms.cf_kmeans import KMeans

test_perc = 0.1


def main(n_clusters: int, distance='euclidean', similarity='pearson', custom_state=None, dataset=None):

    print("RUNNING K-MEANS...")
    print("  --> N_clusters:", n_clusters)
    print("  --> Distance measure:", distance)
    print("  --> Similarity measure:", similarity)
    print('  --> Dataset:', dataset)

    output = [custom_state, n_clusters, distance, similarity]

    df, dim = d.get_dataset(dataset)

    R, X, user_mapping, item_mapping = r.create_cf_matrix(df, size=dim)

    # Split data for CF-based evaluation
    idx_train, idx_val, X_test = r.train_test_split_sg(
        X, user_size=test_perc, item_size=test_perc, custom_state=custom_state)

    # Set training and evaluation sets
    X_train, X_val = R.toarray()[idx_train], R.toarray()[idx_val]

    # Clustering stage
    kmeans = KMeans(n_clusters, distance_function=distance, custom_state=custom_state)

    # Computing clusters of training users by k-Means
    _clusters = kmeans.fit(X_train)

    # Predicting the clusters of test users
    preds = kmeans.predict(X_val)

    print("k-Means clusters: ", np.bincount(_clusters))

    # Mapping (dictionary) users by
    # corresponding assigned cluster
    clusters = []
    for i in range(kmeans.n_centers):
        tmp = [j for j in range(X_train.shape[0]) if _clusters[j] == i]
        clusters.append(tmp)

    # Recommendation stage
    # Choose the similarity function
    sim_fn = s.SIMILARITIES.get(similarity)

    # Choose the aggregation function to recommendation with
    aggr_fn = r.aggregation_functions.get('mean_centered')

    print("k-Means - Predicting ratings from test users...")

    start = time.time()

    X_pred = {}

    # For each test user predict his/her list of test items
    for z, value in tqdm.tqdm(enumerate(X_test.items()), total=len(X_test.keys())):
        # Unpack user inner id and list of items (inner ids)
        user, items = value

        # Cluster of test sample z
        c = int(preds[z])

        neighbors = [idx_train[j] for j in clusters[c]]

        # Compute similarities from user cluster (neighborhood)
        W = [sim_fn(X_val[z], X_train[j]) for j in clusters[c]]
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
    X_rel = r.get_average_ratings(X, X_test, baseline=4.)

    # Prediction quality
    mae, rmse = r.evaluate_prediction_task(X_true, X_pred, verbose=True)
    output.extend([mae, rmse])

    precision, recall = [], []
    for k in [1, 5, 10]:
        # Ranking quality
        ndcg_at_k = r.evaluate_ranking_task(X_true, X_pred, X_rel, k=k, verbose=True)
        output.append(ndcg_at_k)

        # Recommendation quality
        precision_at_k, recall_at_k = r.evaluate_recommendation_quality(X_true, X_pred, X_rel, k=k, verbose=True)
        precision.append(precision_at_k)
        recall.append(recall_at_k)

    output.extend(precision)
    output.extend(recall)

    return output


if __name__ == '__main__':

    save = True

    _s = 'pearson'

    arg_dataset = str(sys.argv[1])
    arg_first_run = int(sys.argv[2])
    arg_n_runs = int(sys.argv[3])
    arg_nclusters = int(sys.argv[4])
    arg_seed = int(sys.argv[5])
    arg_dist = str(sys.argv[6])
    arg_kmax = int(sys.argv[7])

    if arg_dist in c.DISTANCE_FUNCTIONS:
        distances = [arg_dist]
    else:
        distances = c.DISTANCE_FUNCTIONS

    # Set an instance of random state generator (rg)
    rg = np.random.RandomState(arg_seed)
    random_states = rg.randint(1, 1e5, arg_n_runs + arg_first_run)

    # Set the project's directory
    data_dir = os.getcwd()

    for it in range(arg_first_run, arg_n_runs + arg_first_run):

        print("Running experiment ", it, " with random state = ", random_states[it], "...")

        # # Set the project's directory
        # data_dir = os.getcwd()

        cur_path = "/".join([data_dir, "out", arg_dataset,
                             "kmeans", f"{it}_{random_states[it]}_{arg_dataset}.csv"])

        print("Current output file path:", cur_path)

        if save:
            if not os.path.exists(cur_path):
                # Initializes pandas dataframe with pre-defined column header
                df = pd.DataFrame(columns=c.HEADER_KMEANS_CSV)

                # Saves dataframe to file
                df.to_csv(cur_path)

        if arg_nclusters != 0:
            n_clusters_interval = [arg_nclusters]
        else:
            read_path = "/".join([data_dir, "out", arg_dataset,
                                  "opf_snn", f"{it}_{random_states[it]}_{arg_dataset}.csv"])

            tmp_df = pd.read_csv(read_path)
            n_clusters_interval = tmp_df.query(f"distance == '{distances[0]}' and kmax == {arg_kmax}")["nclusters"].tolist()

        for nc in n_clusters_interval:

            df = pd.DataFrame(columns=c.HEADER_KMEANS_CSV)

            for dist in distances:

                # Runs the experiment itself
                cur_output = main(nc, distance=dist, similarity=_s, custom_state=random_states[it], dataset=arg_dataset)

                if save:
                    # Append a new row of data to the end of dataframe
                    df.loc[len(df)] = cur_output

                    # Update the existing csv file
                    df.to_csv(cur_path, mode='a', header=False, na_rep='NA', quoting=csv.QUOTE_NONE)
                    # df = pd.DataFrame(columns=c.HEADER_KMEANS_CSV)

                    print("Current experiment saved to: '", cur_path, "'...")
