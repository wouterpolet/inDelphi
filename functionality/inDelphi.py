import argparse
import os
import pickle
import warnings
from collections import defaultdict

import autograd.numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import helper
from author_helper import print_and_log
import ins_network as knn
import neural_networks as nn
import figure_generation
import prediction as pred
from sequence_generation import load_sequences_from_cutsites

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)


EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))
DELETION_LEN_LIMIT = 30


def get_args(args):
  """
  Get program arguements and set local variables
  @param args:
  @return:
  """
  exec_id = ''
  train_models = True
  load_prediction = False
  new_targets = False

  if args.load_pred:
    load_prediction = args.load_pred == 'True'

  if args.new_targets:
    new_targets = args.new_targets == 'True'

  if args.model_folder:
    exec_id = args.model_folder
    train_models = False

  if args.exec_type:
    execution_flow = args.exec_type
  else:
    execution_flow = 'both'

  return train_models, exec_id, load_prediction, execution_flow, new_targets


def model_creation(data, model_type):
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  out_folder = out_dir + model_type
  print_and_log("Training Neural Networks...", log_fn)
  nn_params, nn2_params = nn.create_neural_networks(data, log_fn, out_folder, exec_id)
  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  print_and_log("Training KNN...", log_fn)
  total_values = helper.load_total_phi_delfreq(out_folder)
  rate_model, bp_model, normalizer = knn.train_knn(data, total_values, out_folder, out_folder + FOLDER_STAT_KEY)
  model = {'nn': nn_params, 'nn_2': nn2_params, 'rate': rate_model, 'bp': bp_model, 'norm': normalizer}
  return model


def calculate_predictions(data, models, in_del, new_targets=False):
  """
  Calculate all predictions for the data provided using the models
  @param data: dataframe with the samples
  @param models: all models (NN1, nn2, bp, rates and normalizer)
  @param in_del: if predicting in_dels or frequencies
  @param new_targets: if to predict new targets or previous cached ones
  @return: dataframe with prediction per sample
  """
  # Getting the cutsites for human gene (approx 226,000,000)

  if in_del:
    print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions_file = 'in_del_distribution_mesc.pkl'
    if os.path.exists(predictions_file):
      predictions_file = 'in_del_distribution_u2os.pkl'
  else:
    print_and_log("Loading Gene Cutsites...", log_fn)
    data = load_sequences_from_cutsites(data, new_targets, sample_size=1003524)
    print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions_file = 'freq_distribution.pkl'

  predictions = pred.Prediction(30, 28, models, data)

  print_and_log("Storing Predictions...", log_fn)
  helper.store_predictions(out_dir, predictions_file, predictions)

  return predictions


def get_observed_values(data):
  """
  Calculate observed values from data provided
  @param data: dataframe of samples
  @return: the observed value per sample
  """
  unique_samples = data['Sample_Name'].unique()
  grouped_res = {}
  for sample_name in unique_samples:
    res = {}
    sample_del = data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'DELETION')]

    res[1] = sum(data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'INSERTION') & (data['Indel'].str.startswith('1+'))]['countEvents'])
    total = res[1]
    for i in range(1, DELETION_LEN_LIMIT + 1):
      res[-i] = sum(sample_del[sample_del['Size'] == i]['countEvents'].tolist())
      total += res[-i]

    # Normalize
    for length, count in res.items():
      res[length] = count / total
    grouped_res[sample_name] = res

  return grouped_res


def calculate_figure_3(train_model, load_prediction, new_targets):
  """
  Supprocess to train/load models, data, predict and plot figure 3f
  @rtype: object
  """
  fig3_predictions = None
  # Loading predictions if specified & file exists
  if load_prediction:
    fig3_predictions = helper.load_predictions(out_dir)

  if fig3_predictions is None:
    if train_model:  # Training model
      print_and_log("Loading data...", log_fn)
      all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
      models_3 = model_creation(all_data_mesc, 'fig_3/')
    else:  # Loading model
      model_folder = out_dir + 'fig_3/'
      print_and_log("Loading models...", log_fn)
      models_3 = helper.load_models(model_folder)
    # Making predictions from model
    fig3_predictions = calculate_predictions(input_dir + 'genes/mart_export.txt', models_3, in_del=False, new_targets=new_targets)

  print_and_log("Plotting Figure...", log_fn)
  figure_generation.figure_3(fig3_predictions)
  return


# TODO fix entire function
def calculate_figure_4(train_model, load_prediction):
  """
  Supprocess to train/load models, data, predict and plot figure 4b
  @rtype: object
  """
  fig4a_predictions = None
  fig4b_predictions = None

  train_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}train_mesc.pkl'
  test_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}test_mesc.pkl'
  train_u2os_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}train_u2os.pkl'
  test_u2os_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}test_u2os.pkl'

  libA = load_lib_data(input_dir + 'libX/', 'libA')

  # Loading predictions if specified & file exists
  if load_prediction:
    # TODO fix load predictions
    fig4a_predictions = helper.load_predictions(out_dir)
    fig4b_predictions = helper.load_predictions(out_dir)
    test_mesc = helper.load_pickle(test_mesc_file)
    test_u2os = helper.load_pickle(test_u2os_file)

    prediction_files = os.listdir(out_dir + FOLDER_PRED_KEY)
    if len(prediction_files) == 3:
      mesc_file = ''
      u2os_file = ''
      for prediction_file in prediction_files:
        if "mesc" in prediction_file:
          mesc_file = prediction_file
        elif "u2os" in prediction_file:
          u2os_file = prediction_file
      if mesc_file != '' and u2os_file != '':
        fig4a_predictions = helper.load_pickle(out_dir + FOLDER_PRED_KEY + mesc_file)
        fig4b_predictions = helper.load_pickle(out_dir + FOLDER_PRED_KEY + u2os_file)

  print_and_log("Loading data...", log_fn)
  if fig4a_predictions is None or fig4b_predictions is None:
    # Loading Data
    if train_model:
      all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1)
      all_data_mesc = all_data_mesc.reset_index()
      print_and_log(f"mESC Loaded - Count(Items): {len(all_data_mesc)}", log_fn)
      all_data_u2os = pd.concat(helper.read_data(input_dir + 'U2OS.pkl'), axis=1)
      all_data_u2os = all_data_u2os.reset_index()
      all_data_u2os = all_data_u2os.rename(columns={'deletionLength': 'Size'})

      print_and_log(f"u2OS Loaded - Count(Items): {len(all_data_u2os)}", log_fn)
      # Reshuffling the data
      unique_mesc = np.random.choice(all_data_mesc['Sample_Name'].unique(), size=189, replace=False)
      test_mesc = all_data_mesc[all_data_mesc['Sample_Name'].isin(unique_mesc)]
      train_mesc = pd.merge(all_data_mesc, test_mesc, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

      unique_mesc = np.random.choice(train_mesc['Sample_Name'].unique(), size=1095, replace=False)
      train_mesc = all_data_mesc[all_data_mesc['Sample_Name'].isin(unique_mesc)]

      # removing exception cases - aka deletions, with Homology length not 0 and no counter events
      wrong_grna = all_data_u2os[(all_data_u2os['Type'] == 'DELETION') & (all_data_u2os['homologyLength'] != 0)].groupby('Sample_Name').sum()
      wrong_grna = wrong_grna.reset_index()
      wrong_grna = wrong_grna[wrong_grna['countEvents'] == 0]['Sample_Name']
      all_data_u2os = all_data_u2os[all_data_u2os["Sample_Name"].isin(wrong_grna) == False]
      unique_u2os = np.random.choice(all_data_u2os['Sample_Name'].unique(), size=185, replace=False)
      test_u2os = all_data_u2os[all_data_u2os['Sample_Name'].isin(unique_u2os)]
      train_u2os = pd.merge(all_data_u2os, test_mesc, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

      # Store the data used:
      print_and_log("Storing Predictions...", log_fn)

      with open(train_mesc_file, 'wb') as out_file:
        pickle.dump(train_mesc, out_file)
      with open(test_mesc_file, 'wb') as out_file:
        pickle.dump(test_mesc, out_file)

      with open(train_u2os_file, 'wb') as out_file:
        pickle.dump(train_u2os, out_file)
      with open(test_u2os_file, 'wb') as out_file:
        pickle.dump(test_u2os, out_file)

      # Models for Figure 4
      models_4a = model_creation(train_mesc, 'fig_4mesc/')
      models_4b = model_creation(train_u2os, 'fig_4u20s/')
    else:
      test_mesc = helper.load_pickle(test_mesc_file)
      test_u2os = helper.load_pickle(test_u2os_file)
      print_and_log("Loading models...", log_fn)
      models_4a = load_models(out_dir + 'fig_4mesc/')
      models_4b = load_models(out_dir + 'fig_4u20s/')

    test_mesc_targets = get_targets(libA, test_mesc)
    test_u2os_targets = get_targets(libA, test_u2os)
    fig4a_predictions = calculate_predictions(test_mesc_targets, models_4a, in_del=True)
    fig4b_predictions = calculate_predictions(test_u2os_targets, models_4b, in_del=True)

  # Get Observed Values
  print_and_log("Calculating the Observed Values...", log_fn)
  fig4a_observations = get_observed_values(test_mesc)
  fig4b_observations = get_observed_values(test_u2os)

  print_and_log("Calculating Pearson Correlation...", log_fn)
  pearson_mESC = pred.get_pearson_pred_obs(fig4a_predictions, fig4a_observations)
  pearson_u2OS = pred.get_pearson_pred_obs(fig4b_predictions, fig4b_observations)

  print_and_log("Plotting Figure...", log_fn)
  figure_generation.figure_4(pearson_mESC, pearson_u2OS, out_dir + FOLDER_GRAPH_KEY + 'plot_4b_' + exec_id)

  return


if __name__ == '__main__':
  # Execution Parameters
  parser = argparse.ArgumentParser(description='Execution Details')
  parser.add_argument('--process', dest='exec_type', choices=['3f', '4b', 'both'], type=str, help='Which model / figure to reproduce')
  parser.add_argument('--model_folder', dest='model_folder', type=str, help='Variable indicating the execution id of the trained neural network and knn')
  parser.add_argument('--load_pred', dest='load_pred', type=str, help='File name used to predict outcomes')
  parser.add_argument('--new_targets', dest='new_targets', type=str, help='Boolean indicating if new targets should be calculated')
  args = parser.parse_args()
  train_models, user_exec_id, load_prediction, execution_flow, new_targets = get_args(args)

  # Program Local Directories
  out_directory, log_file, execution_id = helper.initialize_files_and_folders(user_exec_id)
  global log_fn
  log_fn = log_file
  global out_dir
  out_dir = out_directory
  global exec_id
  exec_id = execution_id
  global input_dir
  input_dir = EXECUTION_PATH + FOLDER_INPUT_KEY

  out_nn_param_dir = out_dir + FOLDER_PARAM_KEY
  out_stat_dir = out_dir + FOLDER_STAT_KEY
  out_plot_dir = out_dir + FOLDER_GRAPH_KEY

  if execution_flow == '3f':
    calculate_figure_3(train_models, load_prediction, new_targets)
  elif execution_flow == '4b':
    calculate_figure_4(train_models, load_prediction)
  else:
    calculate_figure_3(train_models, load_prediction, new_targets)
    calculate_figure_4(train_models, load_prediction)
  print_and_log('Execution complete - model folder: ' + exec_id, log_fn)

