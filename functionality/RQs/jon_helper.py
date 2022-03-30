import os
import helper
import util
import pickle
from prediction import predict_data_outcomes
from sequence_generation import load_sequences_from_cutsites

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


def calculate_predictions(data, models, new_targets=False, sample_size=1003524):
  gene_data = load_sequences_from_cutsites(data, new_targets, sample_size)
  predictions = predict_data_outcomes(gene_data, models, False)
  return predictions


def plot_rsqs(tr1_rsq, tr2_rsq, te1_rsq, te2_rsq):
  print('test?')
  return