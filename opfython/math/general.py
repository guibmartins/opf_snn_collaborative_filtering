"""General-based mathematical methods.
"""

import numpy as np
from scipy import linalg

import opfython.math.distance as d
import opfython.utils.logging as l
import opfython.utils.exception as e

logger = l.get_logger(__name__)


def confusion_matrix(labels, preds):
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The confusion matrix.

    """

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors matrix
    c_matrix = np.zeros((n_class, n_class))

    for label, pred in zip(labels, preds):
        c_matrix[label][pred] += 1

    return c_matrix


def normalize(array):
    """Normalizes an input array.

    Args:
        array (np.array): Array to be normalized.

    Returns:
        The normalized version (between 0 and 1) of the input array.

    """

    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    norm_array = (array - mean) / std

    return norm_array


def opf_accuracy(labels, preds):
    """Calculates the accuracy between true and predicted labels using OPF-style measure.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The OPF accuracy measure between 0 and 1.

    """

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors matrix
    errors = np.zeros((n_class, 2))

    # Gathering the amount of labels per class
    counts = np.bincount(labels)

    for label, pred in zip(labels, preds):
        if label != pred:
            errors[pred][0] += 1
            errors[label][1] += 1

    # Calculating the float value of the true label errors
    errors[:, 1] /= counts

    # Calculating the float value of the predicted label errors
    errors[:, 0] /= (np.nansum(counts) - counts)

    # Calculates the sum of errors per class
    errors = np.nansum(errors, axis=1)

    # Calculates the OPF accuracy
    accuracy = 1 - (np.sum(errors) / (2 * n_class))

    return accuracy


def opf_accuracy_per_label(labels, preds):
    """Calculates the accuracy per label between true and predicted labels using OPF-style measure.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The OPF accuracy measure per label between 0 and 1.

    """

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors array
    errors = np.zeros(n_class)

    # Gathering the amount of labels per class
    _, counts = np.unique(labels, return_counts=True)
    
    for label, pred in zip(labels, preds):
        if label != pred:
            errors[label] += 1

    # Calculating the float value of the true label errors
    errors /= counts

    # Calculates the OPF accuracy
    accuracy = 1 - errors

    return accuracy


def pre_compute_distance(data, output, distance='log_squared_euclidean'):
    """Pre-computes a matrix of distances based on an input data.

    Args:
        data (np.array): Array of samples.
        output (str): File to be saved.
        distance (str): Distance metric to be used.

    """

    logger.info('Pre-computing distances ...')

    # Gathering the size of pre-computed matrix
    size = data.shape[0]

    # Creating an matrix of pre-computed distances
    distances = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            distances[i][j] = d.DISTANCES[distance](data[i], data[j])

    # Saves the distance matrix to an output
    np.savetxt(output, distances)

    logger.info('Distances saved to: %s.', output)


def purity(labels, preds):
    """Calculates the purity measure of an unsupervised technique.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the assigned labels by the clusters.

    Returns:
        The purity measure.

    """

    # Calculating the confusion matrix
    c_matrix = confusion_matrix(labels, preds)

    # Calculating the purity measure
    _purity = np.sum(np.max(c_matrix, axis=0)) / len(labels)

    return _purity


def mean_absolute_error(labels, preds):
    """Calculates the Mean Absolute Error (MAE) between true and predicted labels.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The MAE measure between 0 and 1.

    """

    # Making sure that labels is a numpy array
    labels = np.asarray(labels)

    # Making sure that predictions is a numpy array
    preds = np.asarray(preds)

    # Number of testing samples to be evaluated
    n = float(len(labels))

    if n <= 0:
        raise e.ValueError('`n` should be a positive real number.')

    return np.sum(np.abs(labels - preds)) / n


def mean_squared_error(labels, preds, square_root=False):
    """Calculates the Mean Squared Error (MSE) between true and predicted labels.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.
        square_root (bool): Boolean that indicated whether to apply squared root or not to MSE.

    Returns:
        The MSE or RMSE measure between 0 and 1.

    """

    # Making sure that labels is a numpy array
    labels = np.asarray(labels)

    # Making sure that predictions is a numpy array
    preds = np.asarray(preds)

    # Number of testing samples to be evaluated
    n = float(len(labels))

    if n <= 0:
        raise e.ValueError('`n` should be a positive real number.')

    mse = np.sum((labels - preds) ** 2) / n

    if square_root:
        # Calculate the root-mean squared error (RMSE)
        return mse ** 0.5

    return mse


def eigen_centrality(A: np.ndarray) -> tuple:

    eig_val, eig_vec = linalg.eigh(A)
    idx_max = np.argmax(eig_val.real)

    return eig_vec.T[idx_max].real, eig_val[idx_max].real


def norm_eigen_centrality(A: np.ndarray, tol: np.float64 = 1e-10) -> tuple:

    ones = np.ones((A.shape[0]))
    dA = np.diagflat(A.dot(ones) + tol)
    N = linalg.inv(dA).dot(A)
    eigval, eigvec = linalg.eigh(N.T)

    # Select the index of the principal (max) eigen value
    imax = np.argmax(eigval)

    return eigvec.T[imax].real, eigval[imax].real
