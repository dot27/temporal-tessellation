# Contains utilities for training the rnn and making predictions.

import time

import numpy as np
import tensorflow as tf

from rnn import MPIImodel
from time import gmtime, strftime
from utils import get_index_of_nn
from utils import save_predictions
from my_reader import mpii_iterator
from constants import RNN_PREDICTIONS_TEST_DIR_PATH, \
                      RNN_PREDICTIONS_VAL_DIR_PATH, \
                      RNN_DEBUG_PREDICTIONS_TEST_DIR_PATH, NO_PREDICITONS


def run_epoch(session, model, input_train_x, input_train_y, eval_op, verbose=False):
  """Runs the model on the given data."""
  # The input x consists of a concatenation.
  assert (input_train_x.shape[0] == input_train_y.shape[0])
  assert (input_train_x.shape[1] == input_train_y.shape[1] * 2)
  samples_num = input_train_x.shape[0]
  epoch_size = ((samples_num // model.batch_size) - 1) // model.num_steps
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  for step, (x, y) in enumerate(mpii_iterator(input_train_x, input_train_y, model.batch_size,
                                              model.num_steps)):
    fetches = [model.cost, model.final_state, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps

    print "Processed {:.2f}% of epoch.".format(100 * float(step) / epoch_size)
    print("Current Train l2 loss: %.3f" % (costs / iters))

  return costs / iters


def get_predictions(session, model, input_x, input_y, db, split, verbose=False, is_write_rnn_vectors=False):
  """Runs predictions."""
  rnn_vectors_output = NO_PREDICITONS
  if is_write_rnn_vectors:
    rnn_vectors_output = np.zeros_like(input_y)

  assert (input_x.shape == input_y.shape)
  samples_num = input_x.shape[0]
  epoch_size = ((samples_num // model.batch_size) - 1) // model.num_steps
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  matching_indexes = []

  for step, (x, y, possible_y) in enumerate(mpii_iterator(input_x, input_y, model.batch_size,
                                                          model.num_steps, db=db)):
    if step == 0:
      prev_selection = np.zeros_like(x)

    fetches = [model.cost, model.final_state, model.output]
    feed_dict = {}
    feed_dict[model.input_data] = np.concatenate([x, prev_selection], axis=2)

    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    cost, state, y_hat = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps
    possible_y = np.squeeze(possible_y)

    assert (possible_y == db[step]).all()
    match_index = get_index_of_nn(possible_y, y_hat)
    prev_selection = possible_y[match_index]
    prev_selection = prev_selection.reshape(x.shape)

    matching_indexes.append(match_index)
    if is_write_rnn_vectors:
      rnn_vectors_output[step] = y_hat

    print "Processed {:.2f}% of epoch.".format(100*float(step)/epoch_size)
    print split
    print("Prediction Current l2 loss: %.3f" % (costs / iters))

  if not is_write_rnn_vectors:
    return costs / iters, matching_indexes, NO_PREDICITONS

  return costs / iters, matching_indexes, rnn_vectors_output


def main(train_config, eval_config,
         input_train_x, input_train_y, train_matching_inds, train_db,
         input_val_x, input_val_y, val_matching_inds, val_db,
         input_test_x, input_test_y, test_matching_inds, test_db,
         trained_model_file_path=None,
         is_write_rnn_vectors=False,
         is_load_trained_model=False,
         debug=False):

  # Set as true once you find the best parameters, and choose the proper epoch.
  is_save_test_predictions = False

  config = train_config
  eval_config = eval_config
  test_config = eval_config

  time_stamp = str(int(round(time.time() * 1000))) + strftime("%Y-%m-%d %H:%M:%S", gmtime())
  time_stamp = time_stamp.replace(' ', '_')

  val_output_dir = RNN_PREDICTIONS_VAL_DIR_PATH
  test_output_dir = RNN_PREDICTIONS_TEST_DIR_PATH

  if debug:
    val_output_dir = RNN_DEBUG_PREDICTIONS_TEST_DIR_PATH
    test_output_dir = RNN_DEBUG_PREDICTIONS_TEST_DIR_PATH

  model_time_stamp = str(int(round(time.time() * 1000))) + strftime("%Y-%m-%d %H:%M:%S", gmtime())
  model_time_stamp = model_time_stamp.replace(' ', '_')

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = MPIImodel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = MPIImodel(is_training=False, config=eval_config)
      mtest = MPIImodel(is_training=False, config=test_config)

    if not is_load_trained_model:
      tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):
      time_stamp = str(int(round(time.time() * 1000))) + strftime("%Y-%m-%d %H:%M:%S", gmtime())
      time_stamp = time_stamp.replace(' ', '_')

      start_time = time.time()
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      # Train
      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_l2_loss = run_epoch(session, m, input_train_x, input_train_y, m.train_op,
                                verbose=True)

      print("Epoch: %d Train l2 loss: %.3f" % (i + 1, train_l2_loss))
      print 'Finished epcoh data in {} minutes.'.format((time.time() -
                                                         start_time) / 60)

      # Val
      start_time = time.time()
      val_l2_loss, val_predicted_match_indexs, val_rnn_output_vectors = \
        get_predictions(session, mvalid, input_val_x, input_val_y, val_db, 'val', verbose=True,
                        is_write_rnn_vectors=is_write_rnn_vectors)

      print("Epoch: %d Val l2 loss: %.3f" % (i + 1, val_l2_loss))
      print 'Finished epcoh preciction in {} minutes.'.format((time.time() -
                                                               start_time) / 60)

      save_predictions(i, val_matching_inds, val_predicted_match_indexs, train_config, val_output_dir,
                       'val', time_stamp,
                       debug=debug)

      # Test
      if is_save_test_predictions:
        start_time = time.time()
        test_l2_loss, test_predicted_match_indexs, test_rnn_output_vectors = \
          get_predictions(session, mtest, input_test_x, input_test_y, test_db, 'test', verbose=True,
                          is_write_rnn_vectors=is_write_rnn_vectors)

        print("Epoch: %d test l2 loss: %.3f" % (i + 1, test_l2_loss))
        print 'Finished epcoh preciction in {} minutes.'.format((time.time() -
                                                                 start_time) / 60)

        save_predictions(i, test_matching_inds, test_predicted_match_indexs, train_config, test_output_dir,
                         'val', time_stamp,
                         debug=debug)
