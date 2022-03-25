import os
import helper
import util
import pickle

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_INPUT_KEY = '/in/'
EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))


def load_nn_statistics(out_dir):
  return helper.load_pickle(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY + '_loss_values.pkl')


def save_statistics(out_dir, statistics):
  util.ensure_dir_exists(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY)
  pickle.dump(statistics, open(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY + '_loss_values.pkl', 'wb'))
  return
