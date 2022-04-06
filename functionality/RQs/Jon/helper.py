import pickle
import pandas as pd

import functionality.helper as helper
import functionality.prediction as pred
import functionality.author_helper as util
from functionality.sequence_generation import load_sequences_from_cutsites

LOSS_VALUE_FILE = '_loss_values.pkl'


def load_nn_statistics(out_dir):
  try:
    return helper.load_pickle(out_dir + LOSS_VALUE_FILE)
  except:
    nn1_loss = helper.load_pickle(out_dir + 'loss_nn1/' + LOSS_VALUE_FILE)
    nn1_loss = nn1_loss.rename(columns={"train_rsq": "train_rsq1", "test_rsq": "test_rsq1"})
    nn2_loss = helper.load_pickle(out_dir + 'loss_nn2/' + LOSS_VALUE_FILE)
    nn2_loss = nn2_loss.rename(columns={"train_rsq": "train_rsq2", "test_rsq": "test_rsq2"})
    return pd.concat([nn1_loss, nn2_loss]).groupby(['iteration', 'train_sample_size', 'test_sample_size']).sum().reset_index()


def save_statistics(out_dir, statistics):
  util.ensure_dir_exists(out_dir)
  pickle.dump(statistics, open(out_dir + LOSS_VALUE_FILE, 'wb'))
  return


def calculate_predictions(data, models, new_targets=False, sample_size=helper.SAMPLE_SIZE):
  gene_data = load_sequences_from_cutsites(data, new_targets, sample_size)
  preds = pred.Prediction(30, 28, models)
  predictions = preds.predict_all_sequence_outcomes(gene_data)
  return predictions
