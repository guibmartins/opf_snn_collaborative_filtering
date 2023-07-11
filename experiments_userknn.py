import csv
import os
import sys
import utils.constants as c
import utils.datatools as d

import numpy as np
import pandas as pd

from utils import rectools as r
from cf_algorithms.neighborhood_based import UserKnn


test_perc = 0.1


def main(n_neighbors: int, similarity='pearson', custom_state=None, dataset=None):

    print("RUNNING USER-KNN...")
    print("  --> N_neighbors:", n_neighbors)
    print("  --> Similarity measure:", similarity)
    print('  --> Dataset:', dataset)

    output = [custom_state, n_neighbors, similarity]

    df_train, _ = d.get_dataset(dataset)

    userknn = UserKnn(n_neighbors=n_neighbors, similarity=similarity)
    userknn.fit(df_train)

    # Split data for CF-based evaluation
    idx_train, idx_val, X_test = r.train_test_split_sg(
        userknn.X, user_size=test_perc, item_size=test_perc, custom_state=custom_state)

    userknn.train_indices = idx_train

    # Recommendation stage
    # Predicting rating on the test set
    X_pred = userknn.predict(X_test, k=n_neighbors)

    # top_k = len(userknn.item_ids.keys())
    # for _, items in X_test.items():
    #     top_k = np.minimum(top_k, len(items))

    # For each test user, get his/her list of true ratings 
    X_true = r.get_true_ratings(userknn.X, X_test)

    # For each test user set a relevance threshold
    X_rel = r.get_average_ratings(userknn.X, X_test, baseline=4.0)

    # Prediction quality
    mae, rmse = r.evaluate_prediction_task(X_true, X_pred, verbose=True)

    output.extend([mae, rmse])

    # Ranking quality
    for k in [1, 5, 10]:
        # print("\n")
        ndcg_at_k = r.evaluate_ranking_task(X_true, X_pred, X_rel, k=k, verbose=True)
        output.append(ndcg_at_k)

    # Recommendation quality
    precision = []
    recall = []

    for k in [1, 5, 10]:
        # print("\n")
        precision_at_k, recall_at_k = r.evaluate_recommendation_quality(X_true, X_pred, X_rel, k=k, verbose=True)
        precision.append(precision_at_k)
        recall.append(recall_at_k)

    output.extend(precision)
    output.extend(recall)

    return output


if __name__ == '__main__':

    save = True

    arg_dataset = str(sys.argv[1])
    arg_first_run = int(sys.argv[2])
    arg_n_runs = int(sys.argv[3])
    arg_neighbors = int(sys.argv[4])
    arg_seed = int(sys.argv[5])

    # Set an instance of random state generator (rg)
    rg = np.random.RandomState(arg_seed)
    random_states = rg.randint(1, 1e5, arg_n_runs + arg_first_run)

    # Set the default similarity function (person, cosine, jaccard, euclidean, or squared_euclidean)
    sim_func = 'pearson'
    # sim_func = 'jaccard'

    data_dir = os.getcwd()

    if arg_neighbors != 0:
        k_interval = [arg_neighbors]
    else:
        k_interval = c.KMAX_INTERVAL

    for it in range(arg_first_run, arg_n_runs + arg_first_run):

        print("Running experiment ", it, " with random state = ", random_states[it], "...")

        cur_path = "/".join([data_dir, "out", arg_dataset, "user_knn", f"{it}_{random_states[it]}_{arg_dataset}.csv"])

        print("Current output file path:", cur_path)

        if save:
            if not os.path.exists(cur_path):
                # Initializes pandas dataframe with pre-defined column header
                df = pd.DataFrame(columns=c.HEADER_USERKNN_CSV)

                # Saves dataframe to file
                df.to_csv(cur_path)

        for n_neigh in k_interval:

            df = pd.DataFrame(columns=c.HEADER_USERKNN_CSV)

            # Runs the experiment itself
            cur_output = main(n_neigh, similarity=sim_func, custom_state=random_states[it], dataset=arg_dataset)

            if save:
                # Append a new row of data to the end of dataframe
                df.loc[len(df)] = cur_output

                # Update the existing csv file
                df.to_csv(cur_path, mode='a', header=False, na_rep='NA', quoting=csv.QUOTE_NONE)

                print("Current experiment saved to: '", cur_path, "'...")
