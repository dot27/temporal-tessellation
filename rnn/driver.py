# This file contains utilities for running rnn model.

from rnn import get_config
from constants import DIVIED_DATA_DIR
from my_reader import read_mpii_raw_data
from train_and_predictions_options import main


def drive_net(debug=True):
  """Iterate over the network parameters."""
  data_to_train_dir_path = DIVIED_DATA_DIR
  train_config = get_config('train', debug=debug)
  eval_config = get_config('prediction', debug=debug)

  input_train_x, input_train_y, train_matching_inds, train_db, \
  input_val_x, input_val_y, val_matching_inds, val_db, \
  input_test_x, input_test_y, test_matching_inds, test_db = read_mpii_raw_data(data_to_train_dir_path,
                                                                               debug=debug)

  main(train_config, eval_config,
       input_train_x, input_train_y, train_matching_inds, train_db,
       input_val_x, input_val_y, val_matching_inds, val_db,
       input_test_x, input_test_y, test_matching_inds, test_db,
       debug=debug)


if __name__ == "__main__":
    drive_net(debug=True)
