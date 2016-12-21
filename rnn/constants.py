# Contains common constants.

import os

# Set REL_PATH to your'e directory path.
REL_PATH = ''

# Set TRAIN_FILE_PATH to the training ground truth file path.
TRAIN_FILE_PATH = os.path.join(REL_PATH, 'training_ground_truth file path.txt')


QUERY = 'query'
QUERY_GT = 'QUERY_GT'
VAL = 'val'
TEST = 'test'
VAL_DB = 'VAL_DB'
TEST_DB = 'TEST_DB'
QUERY_VIDEOS = 'QUERY_VIDEOS'
QUERY_CAPTIONS = 'QUERY_CAPTIONS'
DB_VIDEOS = 'DB_VIDEOS'
DB_CAPTIONS = 'DB_CAPTIONS'
QUERY_MATCHES = 'QUERY_MATCHES'
QUERY_DISTANCES = 'QUERY_DISTANCES'
TRAIN_MATCHES = 'TRAIN_MATCHES'
TRAIN_DB = TRAIN_MATCHES

VAL_MATCHES = 'VAL_MATCHES'
VAL_DISTANCES = 'VAL_DISTANCES'
TEST_MATCHES = 'TEST_MATCHES'
TEST_DISTANCES = 'TEST_DISTANCES'
QUERY_MATCHING_INDS = 'QUERY_MATCHING_INDS'
TRAIN_MATCHING_INDS = 'TRAIN_MATCHING_INDS'
VAL_MATCHING_INDS = 'VAL_MATCHING_INDS'
TEST_MATCHING_INDS = 'TEST_MATCHING_INDS'
GT_QUERY_MATCHING_INDS = 'GT_QUERY_MATCHING_INDS'
GT_VAL_MATCHING_INDS = 'GT_VAL_MATCHING_INDS'
GT_TEST_MATCHING_INDS = 'GT_TEST_MATCHING_INDS'

VAL_CAPTIONS = 'setnences_val_mapped'
VAL_VIDEOS = 'video_val_mapped'
TEST_CAPTIONS = 'setnences_tst_mapped'
TEST_VIDEOS = 'video_tst_mapped'

TRAIN_DIVIDED_DIR_PATH = os.path.join(REL_PATH, 'TESSELLATION/DATA_FOR_RNN_CONTEXT/train_splitted')
TRAIN_DIVIDED_DIR_NN_PATH = os.path.join(REL_PATH, 'TESSELLATION/DATA_FOR_RNN_CONTEXT/train_splitted_nn_gt')
TRAIN_SAMPLES = 101072

BEST_POSSIBLE_MATCHES_DIR = \
                    os.path.join(REL_PATH, 'VIDEO_CAPTION_RESULTS/rnn_tesselation/best_possible_mathces_by_hglmm')

BEST_POSSIBLE_MATCHES_QUERY = os.path.join(BEST_POSSIBLE_MATCHES_DIR, 'query_best_possible_captions.txt')
BEST_POSSIBLE_MATCHES_VAL = os.path.join(BEST_POSSIBLE_MATCHES_DIR, 'val_best_possible_captions.txt')
BEST_POSSIBLE_MATCHES_TEST = os.path.join(BEST_POSSIBLE_MATCHES_DIR, 'test_best_possible_captions.txt')


NN_MATCHES_DIR = os.path.join(REL_PATH, 'VIDEO_CAPTION_RESULTS/rnn_tesselation/nn_only_mathces_cca')
NN_MATCHES_QUERY_TXT = os.path.join(NN_MATCHES_DIR, 'nn_matches_query_captions.txt')
NN_MATCHES_VAL_TXT = os.path.join(NN_MATCHES_DIR, 'nn_matches_val_captions.txt')
NN_MATCHES_TEST_TXT = os.path.join(NN_MATCHES_DIR, 'nn_matches_test_captions.txt')

DIVIED_DATA_DIR = os.path.join(REL_PATH, 'TESSELLATION/DATA_FOR_RNN_CONTEXT/divided_data/')
QUERY_FILE_TEXT_GT_PATH = os.path.join(REL_PATH, 'query_gt.txt')
DB_FILE_TEXT_GT_PATH = os.path.join(REL_PATH, 'db_query_gt.txt')

INPUT_TRAIN_X = 'input_train_x'
INPUT_TRAIN_Y = 'input_train_y'

INPUT_VAL_X = VAL
INPUT_VAL_Y = 'input_val_y'

INPUT_TEST_X = TEST
INPUT_TEST_Y = 'input_test_y'
PREDICTIONS_DIR = os.path.join(REL_PATH, 'VIDEO_CAPTION_RESULTS/rnn_tesselation/rnn_predictions/')
RNN_PREDICTIONS_VAL_DIR_PATH = os.path.join(PREDICTIONS_DIR, 'val')
RNN_PREDICTIONS_TEST_DIR_PATH = os.path.join(PREDICTIONS_DIR, 'test')
RNN_DEBUG_PREDICTIONS_TEST_DIR_PATH = os.path.join(PREDICTIONS_DIR, 'debug')
RESULTS_DIR = os.path.join(REL_PATH, 'VIDEO_CAPTION_RESULTS/rnn_tesselation/rnn_predictions_resutls')
VAL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'val')
TEST_RESULTS_DIR = os.path.join(RESULTS_DIR, 'test')

RESULTS_DIR_DEBUG = os.path.join(PREDICTIONS_DIR, 'debug')

RNN_OUTPUTS_VECTORS_DIR = \
    os.path.join(REL_PATH, 'VIDEO_CAPTION_RESULTS/rnn_tesselation/rnn_output_vectors')
VAL_RNN_OUTPUT_VECTORS_DIR = os.path.join(RNN_OUTPUTS_VECTORS_DIR, 'val')
TEST_RNN_OUTPUT_VECTORS_DIR = os.path.join(RNN_OUTPUTS_VECTORS_DIR, 'test')
DEBUG_RNN_OUTPUT_VECTORS_DIR = os.path.join(RNN_OUTPUTS_VECTORS_DIR, 'debug')

NO_PREDICITONS = -1
RNN_PREDICTED_VECTORS = 'rnn_predicted_vectors'
ALL_TRAINED_MODELS_DIR_PATH = os.path.join(REL_PATH, 'TESSELLATION/DATA_FOR_RNN_CONTEXT/trained_models')
TRAINED_MODELS_DIR_PATH = os.path.join(ALL_TRAINED_MODELS_DIR_PATH, 'models')
TRAINED_MODELS_DEBUG_DIR_PATH = os.path.join(ALL_TRAINED_MODELS_DIR_PATH, 'debug')
TRAINED_MODELS_CONFIGURAION_DIR_PATH = \
    os.path.join(REL_PATH, 'TESSELLATION/DATA_FOR_RNN_CONTEXT/trained_models/configutaion_models')

TRAINED_MODELS_CONFIGURAION_DIR_PATH_DEBUG = os.path.join(TRAINED_MODELS_CONFIGURAION_DIR_PATH, 'debug')
TRAINED_MODELS_CONFIGURAION_DIR_PATH_VAL = os.path.join(TRAINED_MODELS_CONFIGURAION_DIR_PATH, 'val')
TRAINED_MODELS_CONFIGURAION_DIR_PATH_TRAIN = os.path.join(TRAINED_MODELS_CONFIGURAION_DIR_PATH, 'train')
