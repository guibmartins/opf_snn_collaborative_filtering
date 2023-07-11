import os.path
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.datasets import make_moons

dense_data_dir = "data_dense"

data_path = {
    'bcw': '/'.join([dense_data_dir, 'breast_cancer.csv']),
    'blood': '/'.join([dense_data_dir, 'blood_transfusion.csv']),
    'ccrf': '/'.join([dense_data_dir, 'cervical_cancer.csv']),
    'cmc': '/'.join([dense_data_dir, 'cmc.csv']),
    'diabetic': '/'.join([dense_data_dir, 'diabetic_debrecen.csv']),
    'facebook': '/'.join([dense_data_dir, 'facebook_live_sellers.csv']),
    'frogs': '/'.join([dense_data_dir, 'frogs_anuran_calls.csv']),
    'google_reviews': '/'.join([dense_data_dir, 'google_review_ratings.csv']),
    'ml100k': '/'.join([dense_data_dir, 'ml100k_movie_ratings_genres.csv']),
    'ml1m': '/'.join([dense_data_dir, 'ml1m_adapted_genre_ratings.csv']),
    'mm': '/'.join([dense_data_dir, 'mammographic_mass.csv']),
    'spam': '/'.join([dense_data_dir, 'spambase.csv'])
}


# A python implementation of c-based switch (conditional structure)
def switch(key: str):
    if key == 'blood':
        return load_blood_transfusion
    elif key == 'bcw':
        return load_breast_cancer_wisconsin
    elif key == 'ccrf':
        return load_cervical_cancer
    elif key == 'cmc':
        return load_contraceptive_method
    elif key == 'diabetic':
        return load_diabetic_retinopathy
    elif key == 'digits':
        return load_skl_digits
    elif key == 'facebook':
        pass
    elif key == 'frogs':
        return load_frogs_anuran_calls
    elif key == 'google_reviews':
        return load_google_reviews
    elif key == 'mm':
        return load_mammographic_mass
    elif key == 'spam':
        return load_spambase
    elif key == 'ml100k':
        return load_movielens_100k
    elif key == 'ml1m':
        return load_movielens_1m
    elif key == 'olivetti_faces':
        return load_olivetti_faces
    elif key == 'synthetic_moons':
        return load_synthetic_moons
    else:
        raise ValueError('A valid dataset name should be used.')


def get_dataset(name: str):

    data_dir = os.getcwd()

    if name == 'ml100k':

        ratings_path = "/".join([data_dir, "data", "movielens_100k", "ratings.csv"])
        raw = pd.read_csv(ratings_path, sep='\t',
                          names=['UserId', 'MovieId', 'Rating', 'Timestamp'], usecols=[0, 1, 2])

        n = raw['UserId'].unique().shape[0]
        m = raw['MovieId'].unique().shape[0]

        print("Data dimensions:", n, m)

    elif name == 'ml1m':

        rating_path = "/".join([data_dir, "data", "movielens_1m", "ratings.csv"])
        raw = pd.read_csv(rating_path, sep="::",
                          names=['UserId', 'MovieId', 'Rating', 'Timestamp'],
                          usecols=[0, 1, 2], engine="python")

        n = raw['UserId'].unique().shape[0]
        m = raw['MovieId'].unique().shape[0]

        print("Data dimensions:", n, m)

    elif name == 'mlsmall':

        rating_path = "/".join([data_dir, "data", "movielens_latest_small", "ratings.csv"])
        raw = pd.read_csv(rating_path, sep=",", header=0, usecols=[0, 1, 2])

        n = raw['userId'].unique().shape[0]
        m = raw['movieId'].unique().shape[0]

        print("Data dimensions:", n, m)

    elif name == 'amzgiftcards':

        rating_path = "/".join([data_dir, "data", "amazon_review", "amazon_gift_cards_filtered.csv"])
        # raw = pd.read_csv(rating_path, sep=",",
        #                   names=['userid', 'itemid', 'rating', 'timestamp'], usecols=[0, 1, 2])
        raw = pd.read_csv(rating_path, sep=",",
                          names=['userid', 'itemid', 'rating'], usecols=[0, 1, 2])

        n = raw['userid'].unique().shape[0]
        m = raw['itemid'].unique().shape[0]

        print("Data dimensions:", n, m)

    elif name == 'amzmagazinesubs':

        rating_path = "/".join([data_dir, "data", "amazon_review", "amazon_magazine_subs_filtered2.csv"])
        # raw = pd.read_csv(rating_path, sep=",",
        #                   names=['userid', 'itemid', 'rating', 'timestamp'], usecols=[0, 1, 2])
        raw = pd.read_csv(rating_path, sep=",",
                          names=['userid', 'itemid', 'rating'], usecols=[0, 1, 2])

        n = raw['userid'].unique().shape[0]
        m = raw['itemid'].unique().shape[0]

        print("Data dimensions:", n, m)

    else:
        raw = None
        n, m = 0, 0

    return raw, (n, m)


def load_synthetic_moons(n_samples=2000, noise=.2, adj_labels=False):

    X, y = make_moons(n_samples=n_samples, noise=noise)

    if adj_labels:
        y += 1

    return X, y


def load_skl_digits(adj_labels=False):

    X, y = load_digits(return_X_y=True)

    if adj_labels:
        y += 1

    return X, y


def load_olivetti_faces(adj_labels=False):

    X, y = fetch_olivetti_faces(return_X_y=True)

    if adj_labels:
        y += 1

    return X, y


# https://archive-beta.ics.uci.edu/ml/datasets/facebook+live+sellers+in+thailand
# samples: 7051
# no target
def load_facebook_live_sellers(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('facebook')

    df = pd.read_csv(path, sep=',', index_col=0, header=0, engine='c')

    # Convert string datetime to timestamp
    df[['status_published']] = df[['status_published']].applymap(
        lambda x: datetime.timestamp(datetime.strptime(x, "%m/%d/%Y %H:%M")))

    X = df.values[:, 1:-4].astype(np.float32)
    y = np.zeros(df.values.shape[0], dtype=int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
# samples: 7195
# target: 10 (0-9)
def load_frogs_anuran_calls(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('frogs')

    df = pd.read_csv(path, sep=',', header=0, engine='c')

    labels = df.values[:, -2].astype(str)
    X = df.values[:, :22].astype(np.float32)
    y = LabelEncoder().fit_transform(labels)

    if adj_labels:
        y += 1

    return X, y


def load_movielens_100k(path=None, adj_labels=False):

    if path is None:
        path = data_path.get('ml100k')

    df = pd.read_csv(path, sep=',', names=list(range(20)))

    X = df.values.astype(np.float32)
    y = np.zeros(df.shape[0], dtype=int)

    if adj_labels:
        y += 1

    return X, y


def load_movielens_1m(path=None, adj_labels=False, missing='knn'):

    if not os.path.exists(str(path)):
        path = data_path.get('ml1m')

    df = pd.read_csv(path, sep=',', index_col=False, header=None)

    if missing == 'knn':
        X = KNNImputer(n_neighbors=5).fit_transform(df.values)
    else:
        X = SimpleImputer().fit_transform(df.values)

    y = np.zeros(df.shape[0], dtype=int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
def load_blood_transfusion(path, adj_labels=True):
    if not os.path.exists(path):
        path = data_path.get('blood')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')
    df.dropna(how='all', inplace=True)

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
def load_breast_cancer_wisconsin(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('bcw')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, 2:].astype(np.float32)
    y = df.values[:, 1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29
def load_cervical_cancer(path=None, adj_labels=False):

    if not os.path.exists(str(path)):
        path = data_path.get('ccrf')

    df = pd.read_csv(path, sep=',', index_col=False, engine='c')

    X = df.values[:, :-4].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
def load_contraceptive_method(path=None, adj_labels=False):
    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('cmc')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if not adj_labels:
        y -= 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
def load_diabetic_retinopathy(path=None, adj_labels=False):
    if not os.path.exists(str(path)):
        path = data_path.get('diabetic')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float64)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/Tarvel+Review+Ratings
def load_google_reviews(path=None, adj_labels=False):

    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('google_reviews')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = KNNImputer(missing_values=0.).fit_transform(df.values.astype(np.float32))
    y = np.zeros(shape=(X.shape[0],))

    if adj_labels:
        y += 1

    return X, y


# http://archive.ics.uci.edu/ml/datasets/mammographic+mass
def load_mammographic_mass(path=None, adj_labels=False):
    if not os.path.exists(str(path)):
        path = data_path.get('mm')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = df.values[:, :-1].astype(np.float32)
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y


# https://archive.ics.uci.edu/ml/datasets/spambase
def load_spambase(path=None, adj_labels=False):
    if (path is None) or (not os.path.exists(path)):
        path = data_path.get('spam')

    df = pd.read_csv(path, sep=',', index_col=False, header=None, engine='c')

    X = SimpleImputer().fit_transform(df.values[:, :-1].astype(np.float64))
    y = df.values[:, -1].astype(int)

    if adj_labels:
        y += 1

    return X, y