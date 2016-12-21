# Contains common utility functions

import os

import numpy as np
import pickle as pkl
from numpy import linalg as LA

from constants import PREDICTIONS_DIR, TRAIN_FILE_PATH


def get_text_predictions(db_indexes, match_indexes):
    """Write rnn results to file.

    Args:
        db_indexes(np.array): train matching indexes.
        match_indexes(list): the predicted nn_num [0,2,3, ...]
    """
    train_captions = np.array(get_lines_in_file(TRAIN_FILE_PATH))
    train_predicted_indexes = db_indexes[range(len(match_indexes)), match_indexes]
    train_predicted_indexes = [int(el) for el in train_predicted_indexes]
    return train_captions[train_predicted_indexes].tolist()


def save_predictions(epoch_num, matching_inds, predicted_match_indexs, config, output_dir,
                     split, time_stamp,
                     debug=False):
    """Save predictions to file.

    Args:
        epoch_num(int): the epoch number.
        matching_inds(np.array): sized [num_samples, nn_num] - the nn indexes in the train file.
        predicted_match_indexs(list): length(num_samples) each in the range [0, nn_num].
        config(MpiiConfig): contains configuration.
        output_dir(str): the path to save the results.
        split(str): 'val', or 'test'.
        debug(bool): if in debug mode.
        time_stamp(str): the timestamp.
    """
    file_name = get_file_name_from_path(split) + '_' + time_stamp + '.pkl'
    file_name = file_name.replace(' ', '_')
    file_path = os.path.join(output_dir, file_name)
    file_path = file_path.replace(' ', '_')
    if debug:
        file_path = add_debug_to_name(file_path)

    params = {}
    params['params'] = config.__dict__
    params['params']['file_path'] = file_path
    params['params']['epoch'] = epoch_num
    predicted_captions = get_text_predictions(matching_inds, predicted_match_indexs)
    params['predicted_captions'] = predicted_captions

    file_path = os.path.join(PREDICTIONS_DIR, split, file_name)
    if debug:
        file_path = os.path.join(PREDICTIONS_DIR, 'debug', file_name)

    write_pkl_file(params, file_path)


def get_file_paths_in_dir(dir_path, extension=None):
    """Get all the file

    Args:
        dir_path(str): the directory path.
        extension(str): file extension.

    Returns:
        list. contains all the paths of the files in the directory.
    """
    if extension != None:
        return [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
                if os.path.splitext(file_name)[1] == extension]

    return [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]


def read_pkl_file(file_path):
    """Read pkl file.

    Args:
        file_path(str): the pickle file path.

    Returns:
        np.array. the context of the pickle file.
    """
    with open(file_path, 'rb') as handle:
        data = pkl.load(handle)

    return data


def get_lines_in_file(input_file_path):
    """Get text lines in a file.

    Args
        input_file_path(str): The path of the text file

    Returns:
        list. contains all the lines in the file.
    """
    with open(input_file_path, 'r') as reader:
        lines = reader.readlines()

    return [line.rstrip('\n') for line in lines]


def add_debug_to_name(file_path):
    """Add 'debug' in the end of the file name

    Args:
        file_path(str): the path of the file.

    Returns:
        str. the path added by '_debug' string before the extension.
    """
    return os.path.splitext(file_path)[0] + '_debug' + os.path.splitext(file_path)[1]


def write_pkl_file(data, file_path):
    """Write data to pickle file.

    Args:
        data(dict): the data to save.
        file_path(st): the path of the file
    """
    with open(file_path, 'wb') as writer:
        pkl.dump(data, writer)


def get_file_name_from_path(file_path):
    """Get the file name without extension from the file path.

    Args:
        file_path(str): the path of the file.

    Returns:
        str. the name of the file without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_index_of_nn(train_canidates_per_caption, val_caption):
    """Get the index that minimizes the distance among train canidates per example.

    Args:
        train_canidates_per_caption(np.array): shape [nn_num, feature_num].
        val_caption(np.array): shape [1, feature_num].

    Returns:
        int. the index of the nn among all the given nn.
    """
    diff = train_canidates_per_caption - val_caption
    diff_norm = LA.norm(diff, axis=1)
    return np.argmin(diff_norm)
