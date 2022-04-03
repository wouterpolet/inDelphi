import os
import functionality.helper as helper
import functionality.author_helper as util
import pickle
import functionality.prediction as pred
from functionality.sequence_generation import load_sequences_from_cutsites


def load_nn_statistics(out_dir):
  return helper.load_pickle(out_dir + helper.FOLDER_PARAM_KEY + helper.FOLDER_STAT_KEY + '_loss_values.pkl')


def save_statistics(out_dir, statistics):
  util.ensure_dir_exists(out_dir + helper.FOLDER_PARAM_KEY + helper.FOLDER_STAT_KEY)
  pickle.dump(statistics, open(out_dir + helper.FOLDER_PARAM_KEY + helper.FOLDER_STAT_KEY + '_loss_values.pkl', 'wb'))
  return


def calculate_predictions(data, models, new_targets=False, sample_size=helper.SAMPLE_SIZE):
  gene_data = load_sequences_from_cutsites(data, new_targets, sample_size)
  preds = pred.Prediction(30, 28, models)
  predictions = preds.predict_all_sequence_outcomes(gene_data)
  return predictions

