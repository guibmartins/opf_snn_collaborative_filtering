"""Data loading utilities.
"""

import json as j

import numpy as np

import opfython.utils.logging as l

logger = l.get_logger(__name__)


def load_csv(csv_path):
    """Loads a CSV file into a numpy array.

    Please make sure the .csv is uniform along all rows and columns.

    Args:
        csv_path (str): String holding the .csv's path.

    Returns:
        A numpy array holding the loaded data.

    """

    logger.info('Loading file: %s ...', csv_path)

    try:
        csv = np.loadtxt(csv_path, delimiter=',')

    except OSError as e:
        logger.error(e)

        return None

    logger.info('File loaded.')

    return csv


def load_txt(txt_path):
    """Loads a .txt file into a numpy array.

    Please make sure the .txt is uniform along all rows and columns.

    Args:
        txt_path (str): A path to the .txt file.

    Returns:
        A numpy array holding the loaded data.

    """

    logger.info('Loading file: %s...', txt_path)

    try:
        txt = np.loadtxt(txt_path, delimiter=' ')

    except OSError as e:
        logger.error(e)

        return None

    logger.info('File loaded.')

    return txt


def load_json(json_path):
    """Loads a .json file into a numpy array.

    Please make sure the .json is uniform along all keys and items.

    Args:
        json_path (str): Path to the .json file.

    Returns:
        A numpy array holding the loaded data.

    """

    logger.info('Loading file: %s ...', json_path)

    try:
        with open(json_path) as f:
            json_file = j.load(f)

    except Exception as e:
        logger.error(e)

        return None

    logger.info('File loaded.')

    # Creating a list to hold the parsed JSON
    json = []

    # For every record in the JSON
    for d in json_file['data']:
        # Gathering meta informations
        meta = np.asarray([d['id'], d['label']])

        # Gathering features
        features = np.asarray(d['features'])

        # Stacking and appending the whole record
        json.append(np.hstack((meta, features)))

    # Transforming the list into a numpy array
    json = np.asarray(json)

    return json
