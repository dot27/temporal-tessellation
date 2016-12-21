"""Utilities for parsing the mpii data and feeding the rnn batches.

This code assumes that you have training, validation and test data that will be processed here.
Training data should contain clips, and their matching captions in a shared space. (In the paper i
used cca for that). Then i created the following (train.pkl, val.pkl, test.pkl):

train.pkl: This was created as following:
  ** split the train data to two parts, one consists of 40% of the data and the other contains the rest.
  ** For each sample out of the 40% **clips** i found the 5 nearest neighboring captions out of the remaining data.
  ** Out of these 5 neighbours i used the ones that their captions is the most similar to the
    current clip(t) - this gives gt_caption(t).

train.pkl:
{
 INPUT_TRAIN_X(numpy.array): size (sample_num,  feature_size * 2), row i consists of clip(t), gt_caption(t-1),
  caption(-1) == 0.
 INPUT_TRAIN_Y: size (sample_num,  feature_size), row i consists of caption(t).
 TRAIN_MATCHING_INDS: The caption matching indexes in the text file.
 TRAIN_DB: Not in use.
}

val.pkl:
{
 INPUT_VAL_X(numpy.array): size (sample_num,  feature_size), row i consists of clip(t).
 INPUT_VAL_Y: size (sample_num,  feature_size), row i consists of caption(t).
 VAL_MATCHING_INDS: The caption matching indexes in the text file.
 VAL_DB: (sample_num, 5 (number of neighbours),feature_size): VAL_DB[i, j, k] is
  [i] - sample number, [j] - the  j'th neighbouring caption from the training set, [k] - feature.

}

test.pkl: Same has val.pkl
"""

import os
import time

import numpy as np

from utils import read_pkl_file
from constants import VAL_DB, TEST_DB, INPUT_TRAIN_X, INPUT_TRAIN_Y, INPUT_VAL_X, \
                      INPUT_VAL_Y, INPUT_TEST_X, INPUT_TEST_Y, VAL_MATCHING_INDS, TEST_MATCHING_INDS, \
                      TRAIN_MATCHING_INDS, TRAIN_DB


def read_mpii_raw_data(data_to_train_dir_path, debug=False):
  """Load mpii raw data

  Returns:
      tuple. consists of train_x, train_y,
        val, val_db
        test, test_db
  """
  print 'Loading data'
  start_time = time.time()
  if debug:
    train_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'train_debug.pkl'))
    val_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'val_debug.pkl'))
    test_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'test_debug.pkl'))

  else:
    train_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'train.pkl'))
    val_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'val.pkl'))
    test_data = read_pkl_file(os.path.join(data_to_train_dir_path, 'test.pkl'))

  input_train_x, input_train_y, train_matching_inds, train_db = \
    train_data[INPUT_TRAIN_X], train_data[INPUT_TRAIN_Y], train_data[TRAIN_MATCHING_INDS], train_data[TRAIN_DB]

  input_val_x, input_val_y, val_matching_inds, val_db = \
    val_data[INPUT_VAL_X], val_data[INPUT_VAL_Y], val_data[VAL_MATCHING_INDS], val_data[VAL_DB]

  input_test_x, input_test_y, test_matching_inds, test_db = \
    test_data[INPUT_TEST_X], test_data[INPUT_TEST_Y], test_data[TEST_MATCHING_INDS], test_data[TEST_DB]

  for key, value in [['input_train_x', input_train_x.shape], ['input_train_y', input_train_y.shape],
                     ['train_matching_inds', train_matching_inds.shape], ['train_db', train_db.shape],

                     ['input_val_x', input_val_x.shape], ['input_val_y', input_val_y.shape],
                     ['val_matching_inds', val_matching_inds.shape], ['val_db', val_db.shape],

                     ['input_test_x', input_test_x.shape], ['input_test_y', input_test_y.shape],
                     ['test_matching_inds', test_matching_inds.shape], ['test_db', test_db.shape]]:

    print '{} shape is {}'.format(key, value)

  print 'Finished loading data in {} minutes.'.format((time.time() - start_time)/60)

  return input_train_x, input_train_y, train_matching_inds, train_db, \
         input_val_x, input_val_y, val_matching_inds, val_db, \
         input_test_x, input_test_y, test_matching_inds, test_db


def mpii_iterator(raw_x, raw_y, batch_size, num_steps, db=None, is_schedule_sampling=False):
  """Iterate on the mpii data.

  This generates batch_size pointers into the raw mpii data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_x: train_input sized [num_samples, 2*feature_num].
    raw_x: train_input sized [num_samples, feature_num].
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    db(np.array): sized [sample_num, nn_num, feature_num].

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps,
                                                      feature_num* {2, 1}].

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data_x = raw_x
  raw_data_y = raw_y

  data_len, feature_num_x = raw_data_x.shape
  _ , feature_num_y = raw_data_y.shape
  batch_len = data_len // batch_size
  data_x = np.zeros([batch_size, batch_len, feature_num_x])
  data_y = np.zeros([batch_size, batch_len, feature_num_y])

  if not (db is None):
    nn_num = db.shape[1]
    db_batched = np.zeros([batch_size, batch_len, nn_num, feature_num_x])
    if is_schedule_sampling:
      db_batched = np.zeros([batch_size, batch_len, nn_num, feature_num_x/2])

  for i in range(batch_size):
    data_x[i] = raw_data_x[batch_len * i:batch_len * (i + 1)]
    data_y[i] = raw_data_y[batch_len * i:batch_len * (i + 1)]
    if not (db is None):
      db_batched[i] = db[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps
  if not (db is None):
    epoch_size = (batch_len) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data_x[:, i*num_steps:(i+1)*num_steps]
    y = data_y[:, i*num_steps:(i+1)*num_steps]
    if db is None:
      yield (x, y)

    else:
      possible_y = db_batched[:, i*num_steps:(i+1)*num_steps]
      yield (x,y, possible_y)
