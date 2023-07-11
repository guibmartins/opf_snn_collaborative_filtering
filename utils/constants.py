EPSILON = 1e-20

MIN_COSINE = -1.
MAX_COSINE = 1.
MIN_PEARSON = -1.
MAX_PEARSON = 1.
MIN_JACCARD = 0
MAX_JACCARD = 1.

DENSITY_FUNCTIONS = ['eigen_centrality', 'pdf']
DISTANCE_FUNCTIONS = ['cosine', 'euclidean', 'jaccard', 'pearson', 'squared_euclidean']
SIMILARITY_FUNCTIONS = ['cosine', 'jaccard', 'pearson']
KMAX_INTERVAL = [10, 20, 30, 40, 50, 60]

NO_LABELS_DATASET = ['google_reviews', 'frogs', 'ml100k', 'ml1m', 'facebook', 'spam']

HEADER_CSV = ['run',
              'kmax', 'kbest', 'nclusters',
              'density', 'distance', 'similarity',
              'mae', 'rmse',
              'ndcg@1', 'ndcg@5', 'ndcg@10',
              'precision@1', 'precision@5', 'precision@10',
              'recall@1', 'recall@5', 'recall@10']

HEADER_USERKNN_CSV = ['run',
                      'n_neighbors',
                      'similarity',
                      'mae', 'rmse',
                      'ndcg@1', 'ndcg@5', 'ndcg@10',
                      'precision@1', 'precision@5', 'precision@10',
                      'recall@1', 'recall@5', 'recall@10']

HEADER_KMEANS_CSV = ['run',
                     'n_clusters',
                     'distance',
                     'similarity',
                     'mae', 'rmse',
                     'ndcg@1', 'ndcg@5', 'ndcg@10',
                     'precision@1', 'precision@5', 'precision@10',
                     'recall@1', 'recall@5', 'recall@10']

HEADER_DBSCAN_CSV = ['run',
                     'eps',
                     'min_samples',
                     'n_clusters',
                     'distance',
                     'similarity',
                     'mae', 'rmse',
                     'ndcg@1', 'ndcg@5', 'ndcg@10',
                     'precision@1', 'precision@5', 'precision@10',
                     'recall@1', 'recall@5', 'recall@10']

HEADER_HDBSCAN_CSV = ['run',
                      'min_cluster_size',
                      'n_clusters',
                      'distance',
                      'similarity',
                      'mae', 'rmse',
                      'ndcg@1', 'ndcg@5', 'ndcg@10',
                      'precision@1', 'precision@5', 'precision@10',
                      'recall@1', 'recall@5', 'recall@10']

HEADER_DENSE_DATA_RESULTS = [
    'timestamp',
    'random_state',
    'algorithm',
    'k_max',
    'k_best',
    'n_clusters',
    'db_index',
    'silhouette_score',
    'homogeneity',
    'completeness',
    'v_measure',
    'adjusted_rand_index',
    'adjusted_mutual_info'
]


