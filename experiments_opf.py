import os
import sys
import time
from datetime import datetime

import tqdm
import numpy as np
import pandas as pd

import utils.constants as c
import utils.datatools as d
import cf_algorithms.similarity as s
from utils import rectools as r
from opfython.opfython.models.unsupervised import ClusteringOPF as OPFClustering


test_perc = 0.1


def similarity_function(x, y, alpha=1.):

    _dist = abs(x.cost - y.cost) / 1000.0
    return (1. - _dist) ** alpha


def get_datetime(str_format=None):

    if str_format is None:
        str_format = "%b-%d-%Y_%H-%M-%S-%f"

    current_date = datetime.now()
    return current_date.strftime(str_format).lower()


def main(n_neighbors: int, density='pdf',
         distance='jaccard', similarity='pearson',
         custom_state=None, dataset=None):

    print("RUNNING CF-OPF...")
    print("  --> N_neighbors:", n_neighbors)
    print("  --> Density computation:", density)
    print("  --> Distance measure:", distance)
    print("  --> Similarity measure:", similarity)
    print('  --> Dataset:', dataset)

    output = [custom_state, n_neighbors]

    df, dim = d.get_dataset(dataset)

    R, X, user_mapping, item_mapping = r.create_cf_matrix(df, size=dim)

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

    # Clustering stage
    opf = OPFClustering(max_k=n_neighbors, distance=distance)

    # Computing clusters by CF-OPF
    clusters = opf.fit_predict(X_train, I_train=idx_train)

    # Predicting clusters for test users
    _, _, predsg = opf.predict(X_val, I_val=idx_val)

    output.extend([opf.subgraph.best_k, opf.subgraph.n_clusters, density, distance, similarity])

    print("OPF - samples per clusters: ", np.bincount(clusters))

    # Mapping (dictionary) users by their corresponding assigned cluster
    clusters = []
    for i in range(opf.subgraph.n_clusters):
        tmp = [j for j in range(opf.subgraph.n_nodes)
               if opf.subgraph.nodes[j].cluster_label == i]
        clusters.append(tmp)

    # Recommendation stage
    # Choose the similarity function
    sim_fn = s.SIMILARITIES.get(similarity)

    # Choose the aggregation function to recommendation with
    aggr_fn = r.aggregation_functions.get('mean_centered')

    print("OPF - Predicting ratings for test users...")

    start = time.time()

    X_pred = {}

    # For each test user predict his/her list of test items
    for z, value in tqdm.tqdm(enumerate(X_test.items()), total=len(X_test.keys())):

        # Unpack user inner id and list of items (inner ids)
        user, items = value

        c = int(predsg.nodes[z].cluster_label)

        adjacents = [idx_train[j] for j in clusters[c]]

        # Compute similarities from user cluster (neighborhood)
        # W = [sim_fn(predsg.nodes[z].shared_adjacency, opf.subgraph.nodes[j].shared_adjacency) for j in clusters[c]]
        W = [sim_fn(predsg.nodes[z].features, opf.subgraph.nodes[j].features) for j in clusters[c]]
        W = np.array(W)

        # Assign predicted ratings for the active test user
        X_pred[user] = [aggr_fn(X, user, i, adjacents, W) for i in items]

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
        print("\n")
        ndcg_at_k = r.evaluate_ranking_task(X_true, X_pred, X_rel, k=k, verbose=True)
        output.append(ndcg_at_k)

    # Recommendation quality
    precision = []
    recall = []
    for k in [1, 5, 10]:
        print("\n")
        precision_at_k, recall_at_k = r.evaluate_recommendation_quality(X_true, X_pred, X_rel, k=k, verbose=True)
        precision.append(precision_at_k)
        recall.append(recall_at_k)

    output.extend(precision)
    output.extend(recall)

    return output


if __name__ == '__main__':

    save = False
    # dist = ['jaccard']
    dist = []

    arg_dataset = str(sys.argv[1])
    arg_first_run = int(sys.argv[2])
    arg_n_runs = int(sys.argv[3])
    arg_kmax = int(sys.argv[4])
    # arg_density = str(sys.argv[5])
    arg_seed = int(sys.argv[5])

    # Set an instance of random state generator (rg)
    rg = np.random.RandomState(arg_seed)
    random_states = rg.randint(1, 1e5, arg_n_runs + arg_first_run)

    sim_func = 'pearson'
    # dens = arg_density  # density function = 'pdf' (always)
    dens = 'pdf'

    data_dir = os.getcwd()

    if arg_kmax != 0:
        kmax_interval = [arg_kmax]
    else:
        kmax_interval = c.KMAX_INTERVAL

    if len(dist) > 0:
        distances = dist
    else:
        distances = c.DISTANCE_FUNCTIONS

    for it in range(arg_first_run, arg_n_runs + arg_first_run):

        print("Running experiment ", it, "...")

        cur_path = "/".join([data_dir, "out", arg_dataset, "opf", f"{it}_{random_states[it]}_{arg_dataset}.csv"])

        print("Current output file path:", cur_path)

        if save:
            df = pd.DataFrame(columns=c.HEADER_CSV)
            df.to_csv(cur_path)

        for k_max in kmax_interval:

            for dist in distances:

                # Runs the experiment itself
                cur_output = main(k_max, dens, dist, sim_func, random_states[it], arg_dataset)

                if save:
                    # Append a new row of data to the end of dataframe
                    df.loc[len(df)] = cur_output

                    # Update the existing csv file
                    df.to_csv(cur_path, mode='a', header=False)
                    df = pd.DataFrame(columns=c.HEADER_CSV)

                    print("Current experiment saved to: '", cur_path, "'...")
