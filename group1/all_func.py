import argparse
import datetime
import glob
import os
import pickle
import warnings
from collections import defaultdict

import autograd.numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import helper
import ins_network as knn
import neural_networks as nn
import plot_3f as plt_3
import plot_4b as plt_4
import prediction as pred
import util as util
from sequence_generation import load_sequences_from_cutsites

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_GRAPH_KEY = 'plots/'
FOLDER_LOG_KEY = 'logs/'
FOLDER_PRED_KEY = 'predictions/'
FOLDER_INPUT_KEY = '/in/'
EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))


def initialize_files_and_folders(user_exec_id):
  # Set output location of model & params
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
  util.ensure_dir_exists(out_place)
  exec_id = ''
  # num_folds = helper.count_num_folders(out_place)
  if user_exec_id == '' or helper.count_num_folders(out_place) < 1:
    exec_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
  else:
    latest = datetime.datetime.strptime('1990/01/01', '%Y/%m/%d')
    for name in os.listdir(out_place):
      try:
        datetime.datetime.strptime(name, "%Y%m%d_%H%M")
      except ValueError:
        if name == user_exec_id:
          exec_id = name
          break
        else:
          continue
      date_time_obj = datetime.datetime.strptime(name, "%Y%m%d_%H%M")
      if name == user_exec_id:
        latest = date_time_obj
        break
      if latest < date_time_obj:
        latest = date_time_obj
    if exec_id == '':
      exec_id = latest.strftime("%Y%m%d_%H%M")


  # if use_prev and num_folds >= 1:
  #   out_letters = helper.alphabetize(num_folds - 1)
  # else:
  #   out_letters = helper.alphabetize(num_folds)

  out_dir = out_place + exec_id + '/'
  util.ensure_dir_exists(out_dir + FOLDER_PRED_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_GRAPH_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_LOG_KEY)

  log_fn = out_dir + FOLDER_LOG_KEY + '_log_%s.out' % datetime.datetime.now().strftime("%Y%m%d_%H%M")
  with open(log_fn, 'w') as f:
    pass
  helper.print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, exec_id


# TODO fix / optimize
# Issue we do not have the same type of data they have
def parse_header(header):
  w = header.split(' ')
  gene_kgid = w[0].replace('>', '')
  chrom = w[1]
  start = int(w[2]) - 30
  end = int(w[3]) + 30
  data_type = w[4]
  return gene_kgid, chrom, start, end


# TODO fix / optimize
def maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force=False):
  if split == '0':
    line_threshold = 500
  else:
    line_threshold = 5000
  norm_condition = bool(
    bool(len(dd['Unique ID']) > line_threshold) and bool(len(dd_shuffled['Unique ID']) > line_threshold))

  if norm_condition or force:
    print('Flushing, num. %s' % (num_flushed))
    df_out_fn = out_dir + '%s_%s_%s.csv' % (data_nm, split, num_flushed)
    df = pd.DataFrame(dd)
    df.to_csv(df_out_fn)

    df_out_fn = out_dir + '%s_%s_shuffled_%s.csv' % (data_nm, split, num_flushed)
    df = pd.DataFrame(dd_shuffled)
    df.to_csv(df_out_fn)

    num_flushed += 1
    dd = defaultdict(list)
    dd_shuffled = defaultdict(list)
  else:
    pass
  return dd, dd_shuffled, num_flushed


def load_ins_models(out_dir_model):
  return helper.load_pickle(out_dir_model + 'rate_model.pkl'), helper.load_pickle(out_dir_model + 'bp_model.pkl'), helper.load_pickle(out_dir_model + 'Normalizer.pkl')


def load_neural_networks(out_dir_params):
  nn_files = glob.glob(out_dir_params + "*_nn.pkl")
  nn_files.sort(reverse=True)
  nn2_files = glob.glob(out_dir_params + "*_nn2.pkl")
  nn2_files.sort(reverse=True)
  return helper.load_pickle(nn_files[0]), helper.load_pickle(nn2_files[0])


def load_predictions(pred_file):
  return helper.load_pickle(pred_file)


def load_lib_data(folder_dir, libX):
  names = []
  grna = []
  target = []
  for file in glob.glob(folder_dir + '*-' + libX + '.txt'):
    file_name = os.path.basename(file)
    data = open(file, "r").read().splitlines()
    if 'names' in file_name:
      names = data
    elif 'grna' in file_name:
      grna = data
    elif 'targets' in file_name:
      target = data
  all_data = pd.DataFrame({'name': names, 'grna': grna, 'target': target})
  return all_data


def get_args(args):
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
  helper.print_and_log("Training Neural Networks...", log_fn)
  nn_params, nn2_params = nn.create_neural_networks(data, log_fn, out_folder, exec_id)
  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  helper.print_and_log("Training KNN...", log_fn)
  total_values = helper.load_pickle(out_folder + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  rate_model, bp_model, normalizer = knn.train_knn(data, total_values, out_folder, out_folder + FOLDER_STAT_KEY)
  return nn_params, nn2_params, rate_model, bp_model, normalizer


def load_models(out_dir):
  helper.print_and_log("Loading models...", log_fn)
  nn_path = out_dir + FOLDER_PARAM_KEY
  nn, nn_2 = load_neural_networks(nn_path)
  rate, bp, norm = load_ins_models(out_dir)
  model = {'nn': nn, 'nn_2': nn_2, 'rate': rate, 'bp': bp, 'norm': norm}
  return model


def calculate_predictions(data, models, in_del, new_targets=False):
  # Getting the cutsites for human gene (approx 226,000,000)
  if in_del:
    helper.print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions = pred.predict_data_outcomes(data, models, in_del)
    predictions_file = f'{out_dir + FOLDER_PRED_KEY}in_del_distribution_mesc.pkl'
    if os.path.exists(predictions_file):
      predictions_file = f'{out_dir + FOLDER_PRED_KEY}in_del_distribution_u2os.pkl'
  else:
    helper.print_and_log("Loading Gene Cutsites...", log_fn)
    gene_data = load_sequences_from_cutsites(data, new_targets)
    # Calculating outcome using our models - only calculate approx 1,000,000
    helper.print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions = pred.predict_data_outcomes(gene_data, models, in_del)
    predictions_file = f'{out_dir + FOLDER_PRED_KEY}freq_distribution.pkl'

  helper.print_and_log("Storing Predictions...", log_fn)
  with open(predictions_file, 'wb') as out_file:
    pickle.dump(predictions, out_file)

  return predictions


def get_observed_values(data):
  unique_samples = data['Sample_Name'].unique()
  grouped_res = {}
  for sample_name in unique_samples:
    res = {}
    sample_del = data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'DELETION')]

    res[1] = sum(data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'INSERTION') & (data['Indel'].str.startswith('1+'))]['countEvents'])
    total = res[1]
    for i in range(1, 31):
      res[-i] = sum(sample_del[sample_del['Size'] == i]['countEvents'].tolist())
      total += res[-i]

    for i in range(31, 61):
      res[-i] = 0

    # Normalize
    for length, count in res.items():
      res[length] = count / total
    grouped_res[sample_name] = res

  # grouped = data.groupby('Sample_Name')['Size'].apply(list).to_dict()
  # grouped_res = {}
  # # create deletion dicts
  # for k, v in grouped.items():
  #   res = {}
  #   for i in range(1, 31):
  #     res[-i] = v.count(i)
  #   grouped_res[k] = res

  # add insertions
  # for k, v in grouped_res.items():
  #   v[1] = len(data[(data['Sample_Name'] == k) & (data['Type'] == 'INSERTION') & (data['Indel'].str.startswith('1+'))])
  #   total = sum(v.values())
  #   for length, count in v.items():
  #     v[length] = count / total

  return grouped_res


def calculate_figure_3(train_model, load_prediction, new_targets):
  fig3_predictions = None
  # Loading predictions if specified & file exists
  if load_prediction:
    files = os.listdir(out_dir + FOLDER_PRED_KEY)
    if len(files) == 1:
      fig3_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + files[0])

  if fig3_predictions is None:
    if train_model:
      # Training model
      helper.print_and_log("Loading data...", log_fn)
      all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
      models_3 = model_creation(all_data_mesc, 'fig_3/')
    else:
      # Loading model
      model_folder = out_dir + 'fig_3/'
      models_3 = load_models(model_folder)
    # Making predictions from model
    fig3_predictions = calculate_predictions(input_dir + 'genes/mart_export.txt', models_3, in_del=False, new_targets=new_targets)

  helper.print_and_log("Plotting Figure...", log_fn)
  plt_3.hist(fig3_predictions, out_dir + FOLDER_GRAPH_KEY + 'plot_3f_' + exec_id + '.pdf')
  return


def get_targets(test_mesc, test_u2os):
  libA = load_lib_data(input_dir + 'libX/', 'libA')
  test_mesc_targets = []
  for sampleName in test_mesc['Sample_Name'].unique():
    grna = sampleName.split('_')
    grna = grna[len(grna) - 1]
    sequences = libA.loc[libA['target'].str.contains(grna, case=False)]['target']
    if len(sequences) == 1:
      test_mesc_targets.append(sequences.values[0])
    else:
      test_mesc_targets.extend([seq for seq in sequences if seq.index(grna) == 10])
  test_u2os_targets = []
  for sampleName in test_u2os['Sample_Name'].unique():
    grna = sampleName.split('_')
    grna = grna[len(grna) - 1]
    sequences = libA.loc[libA['target'].str.contains(grna, case=False)]['target']
    if len(sequences) == 1:
      test_u2os_targets.append(sequences.values[0])
    else:
      test_u2os_targets.extend([seq for seq in sequences if seq.index(grna) == 10])
  return test_mesc_targets, test_u2os_targets


def calculate_figure_4(train_model, load_prediction):
  fig4a_predictions = None
  fig4b_predictions = None

  train_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}train_mesc.pkl'
  test_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}test_mesc.pkl'
  train_u2os_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}train_u2os.pkl'
  test_u2os_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}test_u2os.pkl'

  # Loading predictions if specified & file exists
  if load_prediction:
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
        fig4a_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + mesc_file)
        fig4b_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + u2os_file)
        # train_mesc = load_prediction(train_mesc_file)
        test_mesc = helper.load_pickle(test_mesc_file)
        # train_u2os = load_prediction(train_u2os_file)
        test_u2os = helper.load_pickle(test_u2os_file)
        test_mesc_targets, test_u2os_targets = get_targets(test_mesc, test_u2os)

  helper.print_and_log("Loading data...", log_fn)
  if fig4a_predictions is None or fig4b_predictions is None:
    # Loading Data
    all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1)
    all_data_mesc = all_data_mesc.reset_index()
    helper.print_and_log(f"mESC Loaded - Count(Items): {len(all_data_mesc)}", log_fn)
    all_data_u2os = pd.concat(helper.read_data(input_dir + 'U2OS.pkl'), axis=1)
    all_data_u2os = all_data_u2os.reset_index()
    all_data_u2os = all_data_u2os.rename(columns={'deletionLength': 'Size'})

    helper.print_and_log(f"u2OS Loaded - Count(Items): {len(all_data_u2os)}", log_fn)
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
    helper.print_and_log("Storing Predictions...", log_fn)

    with open(train_mesc_file, 'wb') as out_file:
      pickle.dump(train_mesc, out_file)
    with open(test_mesc_file, 'wb') as out_file:
      pickle.dump(test_mesc, out_file)

    with open(train_u2os_file, 'wb') as out_file:
      pickle.dump(train_u2os, out_file)
    with open(test_u2os_file, 'wb') as out_file:
      pickle.dump(test_u2os, out_file)

    if train_model:
      # Models for Figure 4
      models_4a = model_creation(train_mesc, 'fig_4mesc/')
      models_4b = model_creation(train_u2os, 'fig_4u20s/')
    else:
      models_4a = load_models(out_dir + 'fig_4mesc/')
      models_4b = load_models(out_dir + 'fig_4u20s/')

    test_mesc_targets, test_u2os_targets = get_targets(test_mesc, test_u2os)
    fig4a_predictions = calculate_predictions(test_mesc_targets, models_4a, in_del=True)
    fig4b_predictions = calculate_predictions(test_u2os_targets, models_4b, in_del=True)

  # Get Observed Values
  helper.print_and_log("Calculating the Observed Values...", log_fn)
  fig4a_observations = get_observed_values(test_mesc)
  fig4b_observations = get_observed_values(test_u2os)

  helper.print_and_log("Calculating Pearson Correlation...", log_fn)
  pearson_mESC = pred.get_pearson_pred_obs(fig4a_predictions, fig4a_observations)
  pearson_u2OS = pred.get_pearson_pred_obs(fig4b_predictions, fig4b_observations)

  helper.print_and_log("Plotting Figure...", log_fn)
  plt_4.box_voilin(pearson_mESC, pearson_u2OS, out_dir + FOLDER_GRAPH_KEY + 'plot_4b_' + exec_id)

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
  out_directory, log_file, execution_id = initialize_files_and_folders(user_exec_id)
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
  helper.print_and_log('Execution complete - model folder: ' + exec_id, log_fn)
  #
  #
  # # Only training / loading the models if no prediction file is found
  # if prediction_file == '':
  #   # Load LibA data for training
  #   helper.print_and_log("Loading data...", log_fn)
  #   all_data_mesc = pd.concat(read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
  #   helper.print_and_log(f"mESC Loaded - Count(Items): {len(all_data_mesc)}", log_fn)
  #   all_data_u2os = pd.concat(read_data(input_dir + 'U2OS.pkl'), axis=1).reset_index()
  #   helper.print_and_log(f"u2OS Loaded - Count(Items): {len(all_data_u2os)}", log_fn)
  #
  #   # Reshuffling the data
  #   reorder_mesc = all_data_mesc.sample(frac=1)
  #   reorder_u2os = all_data_u2os.sample(frac=1)
  #
  #   # Splitting into train test so that test can be used for predictions
  #   test_mesc = reorder_mesc.iloc[:189]
  #   train_mesc = reorder_mesc.iloc[189:]
  #   test_u2os = reorder_u2os.iloc[:185]
  #   train_u2os = reorder_u2os.iloc[185:]
  #
  #   if train_models:
  #     # Models for Figure 3
  #     models_3 = model_creation(all_data_mesc, 'fig_3/')
  #     # Models for Figure 4
  #     models_4a = model_creation(train_mesc, 'fig_4mesc/')
  #     models_4b = model_creation(train_u2os, 'fig_4u20s/')
  #   else:
  #     # TODO: loading must be changes
  #     helper.print_and_log("Loading Neural Networks...", log_fn)
  #     # models_3 = model_creation(all_data_mesc)
  #     # models_4a = model_creation(all_data_mesc)
  #     # models_4b = model_creation(all_data_u2os)
  #     nn_params, nn2_params = load_neural_networks(out_nn_param_dir)
  #     helper.print_and_log("Loading KNN...", log_fn)
  #     rate_model, bp_model, normalizer = load_ins_models(out_model_dir)
  #
  #   fig3_predictions = calculate_predictions(input_dir + 'genes/mart_export.txt', models_3, True)
  #   fig4a_predictions = calculate_predictions(test_mesc, models_4a, False)
  #   fig4b_predictions = calculate_predictions(test_u2os, models_4b, False)
  #   # Get Observed Values
  #   fig4a_observations = get_observed_values(test_mesc)
  #   fig4b_observations = get_observed_values(test_u2os)
  #
  #   pearson_mESC = pred.get_pearson_pred_obs(fig4a_predictions, fig4a_observations)
  #   pearson_u2OS = pred.get_pearson_pred_obs(fig4b_predictions, fig4b_observations)
  #
  # else:
  #   helper.print_and_log("Loading Predictions...", log_fn)
  #   fig3_predictions = load_pickle(out_dir + FOLDER_GRAPH_KEY + 'freq_distribution.pkl')
  #   fig4a_predictions = load_pickle(out_dir + FOLDER_GRAPH_KEY + 'in_del_distribution_mesc.pkl')
  #   fig4b_predictions = load_pickle(out_dir + FOLDER_GRAPH_KEY + 'in_del_distribution_u2os.pkl')
  #
  #
  # print('Plotting Graphs - 3f')
  # plt_3.hist(fig3_predictions, out_dir + FOLDER_GRAPH_KEY + 'plot_3f_' + exec_id)
  # print('Plotting Graphs - 4a')
  # plt_4.box_voilin(pearson_mESC, pearson_u2OS, out_dir + FOLDER_GRAPH_KEY + 'plot_4a_' + exec_id)
  #

















  # libX_dir = EXECUTION_PATH + '/data-libprocessing/'
  # if prediction_file == '':
  #   prediction_file = libX_dir


  #
  # # Using liBX data
  # if not only_plots:
  #   print('Prediction')
  #   lib_df = load_lib_data(libX_dir, libX)
  #   # predictions = predict_all_items(lib_df, out_dir_exin, nn_params, nn2_params, rate_model, bp_model, normalizer)
  #   # predictions.to_csv(output_predictions_file)
  #   print('Complete Predictions')
  #   print('More predictions, per sequence, for extended figure')
  #   extended_preds = pred.bulk_predict_all(lib_df, rate_model, bp_model, normalizer)
  #   with open(output_extended_predictions_file, 'wb') as out_file:
  #     pickle.dump(extended_preds, out_file)
  #   print('More predictions done!')
  # else:
  #   predictions = pd.read_csv(output_predictions_file)

  # fig_3f_data_del = predictions['Highest Del Rate'].apply(lambda x: x*100)
  # fig_3f_data_ins = predictions['Highest Ins Rate'].apply(lambda x: x*100)
  # plt.hist(fig_3f_data_del, out_dir_stat + 'del_plot_3f_' + libX)
  # plt.hist(fig_3f_data_ins, out_dir_stat + 'ins_plot_3f_' + libX)
  # Plotting Image 3f


