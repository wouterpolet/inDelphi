import os
import sys


root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_folder)

import argparse
import pandas as pd
import autograd.numpy as np
import autograd.numpy.random as npr

import warnings
from pandas.core.common import SettingWithCopyWarning

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve

from functionality.author_helper import print_and_log
from functionality.RQs.Jon.helper import load_nn_statistics
import functionality.ins_network

import functionality.RQs.Jon.plots as plt
import functionality.helper as helper
import functionality.RQs.Jon.nn as nn
from functionality.ins_network import featurize

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)


def learning_curves(data_nm, total_values, save_dir, plot_type):
  knn = functionality.ins_network.InsertionModel(model_folder, model_folder + helper.FOLDER_STAT_KEY)
  rate_stats, bp_stats = knn.get_statistics(data_nm, total_values)

  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  X, y, normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
  knn = KNeighborsRegressor()
  train_sizes, train_scores, test_scores = learning_curve(knn, X, y, cv=100,
                                                          scoring='neg_mean_squared_error',
                                                          n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))

  knn.fit(X, y)
  # y_pred = knn.predict(X)
  # print('R-Squared: ' + str(r2_score(y, y_pred)))
  # print('KNN Score: ' + str())
  train_mean = -1 * np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  test_mean = -1 * np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  plt.plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, knn.score(X, y), plot_type, save_dir)
  # train_scores_mean = -train_scores.mean(axis=1)
  # validation_scores_mean = -test_scores.mean(axis=1)
  # plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean)



def load_and_plot_model_loss(model_folder, all_data, figure_dir, plot_type, new_model=False):
  loss_values = load_nn_statistics(model_folder)
  plt.plot_nn_loss_epoch(loss_values, save_file=figure_dir + plot_type)

  print_and_log("Original Learning Curve for Insertion Model...", log_fn)
  if new_model:
    total_values = helper.load_pickle(model_folder + 'total_phi_delfreq.pkl')
  else:
    total_values = helper.load_pickle(model_folder + helper.FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  learning_curves(all_data, total_values, figure_dir, plot_type)

  if 'seed' in loss_values.columns:
    return loss_values['seed'][0]
  else:
    return npr.RandomState(1)


def model_creation(data, model_type):
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  out_folder = out_dir + model_type
  try:
    nn_max = helper.load_neural_networks(out_folder + 'max/')
    nn_split = helper.load_neural_networks(out_folder + 'split/')
    nn_fix = helper.load_neural_networks(out_folder + 'fix/')
  except:
    print_and_log("Training Neural Networks...", log_fn)
    nn_split, nn_max, nn_fix = nn.create_neural_networks(data, log_fn, out_folder, exec_id, seed_value)

  return nn_split, nn_max, nn_fix


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execution Details')

  parser.add_argument('--model_folder', dest='model_folder', type=str, help='Variable indicating the execution id of the trained neural network and knn')
  args = parser.parse_args()
  if args.model_folder:
    user_exec_id = args.model_folder
  else:
    raise Exception("Please specify --model_folder")

  out_dir, log_fn, exec_id = helper.initialize_files_and_folders(user_exec_id)
  if user_exec_id != exec_id:
    raise Exception("Please specify a valid pre-trained model")

  print_and_log("Loading pre-trained networks...", log_fn)
  model_folder = out_dir + 'fig_3/'
  nn_path = model_folder + helper.FOLDER_PARAM_KEY
  models = helper.load_models(model_folder)
  nn_params = models['nn']
  nn2_params = models['nn_2']
  rate_model = models['rate']
  bp_model = models['bp']
  normalizer = models['norm']
  all_data = pd.concat(helper.read_data(helper.INPUT_DIRECTORY + 'dataset.pkl'), axis=1).reset_index()

  # Loading and plotting the current model loss values
  print_and_log("Learning Curve for Current Neural Networks...", log_fn)
  seed_value = load_and_plot_model_loss(model_folder, all_data, out_dir + helper.FOLDER_GRAPH_KEY, 'Original')

  # Training a new model with alterations to the NN
  print_and_log("Learning new Neural Network - Split...", log_fn)

  subfolder = 'fig_3_opt/'
  split_nns, max_nns, fix_nns = model_creation(all_data, subfolder)
  # Loading and plotting the current model loss values
  print_and_log("Learning Curve for Current Neural Networks...", log_fn)
  model_folder = out_dir + subfolder + 'split/'
  load_and_plot_model_loss(model_folder, all_data, out_dir + helper.FOLDER_GRAPH_KEY, 'Split', True)
  model_folder = out_dir + subfolder + 'max/'
  load_and_plot_model_loss(model_folder, all_data, out_dir + helper.FOLDER_GRAPH_KEY, 'Max', True)
  model_folder = out_dir + subfolder + 'fix/'
  load_and_plot_model_loss(model_folder, all_data, out_dir + helper.FOLDER_GRAPH_KEY, 'Fix', True)


