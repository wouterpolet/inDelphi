import argparse, datetime, os, pickle, warnings, re
from collections import defaultdict
import glob
import plot_3f as plt_3
import plot_4b as plt_4

import autograd.numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import helper
import util as util

import neural_networks as nn
import ins_network as knn
import prediction as pred

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_GRAPH_KEY = 'plots/'
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

  log_fn = out_dir + '_log_%s.out' % exec_id
  with open(log_fn, 'w') as f:
    pass
  helper.print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, exec_id


def load_pickle(file):
  return pickle.load(open(file, 'rb'))


def read_data(file):
  master_data = load_pickle(file)
  return master_data['counts'], master_data['del_features']


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
def reverse_complement(dna):
  lib = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A', 'N': 'N', 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y', 'Y': 'R'}
  new_dna = ''
  dna = dna.upper()
  for c in dna:
    if c in lib:
      new_dna += lib[c]
    else:
      new_dna += c
  new_dna = new_dna[::-1]
  return new_dna


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


def get_cutsites(chrom, sequence):
  all_cutsites = []
  for idx in range(len(sequence)):  # for each base in the sequence
    # this loop finishes only each of 5% of all found cutsites with 60-bp long sequences containing only ACGT
    seq = ''
    if sequence[idx: idx + 2] == 'CC':  # if on top strand find CC
      cutsite = idx + 6  # cut site of complementary GG is +6 away
      seq = sequence[cutsite - 30: cutsite + 30]  # get sequence 30bp L and R of cutsite
      seq = reverse_complement(seq)  # compute reverse strand (complimentary) to target with gRNA
      orientation = '-'
    if sequence[idx: idx + 2] == 'GG':  # if GG on top strand
      cutsite = idx - 4  # cut site is -4 away
      seq = sequence[cutsite - 30: cutsite + 30]  # get seq 30bp L and R of cutsite
      orientation = '+'
    if seq == '':
      continue
    if len(seq) != 60:
      continue

    # Sanitize input
    seq = seq.upper()
    if 'N' in seq:  # if N in collected sequence, return to start of for loop / skip rest
      continue
    if not re.match('^[ACGT]*$', seq):  # if there not only ACGT in seq, ^
      continue

    all_cutsites.append([chrom, seq])
  return all_cutsites


def find_cutsites_and_predict(inp_fn, use_file=''):
  # Loading Cutsites
  if use_file != '':
    all_data = read_data(use_file + 'cutsites.pkl')
  else:
    # Calculating & Loading cutsites for all files
    cutsites = []
    for file in glob.glob(inp_fn + '*.fa'):
      file_name = os.path.basename(file)
      print('Working on: ' + file_name)
      data = open(file, "r").readlines()[1:]
      sequence = ''.join(data).replace('\n', '')
      cutsites.extend(get_cutsites(file_name, sequence))

    all_data = pd.DataFrame(cutsites, columns=['Chromosome', 'Cutsite'])
    with open(inp_fn + 'cutsites.pkl', 'wb') as f:
      pickle.dump(all_data, f)

  #
  #
  # # Calculate statistics on df, saving to alldf_dict
  # # Deletion positions
  # d = defaultdict(list)            # dictionary with values as list
  # dd_shuffled = defaultdict(list)
  # num_flushed = 0
  # all_data = open(inp_fn, "r").readlines()[1:]
  # sequence = ''.join(all_data).replace('\n', '')
  #
  # bulk_predict(sequence, d)
  # generate dicts (dd and dd_shuffled) for info on each cutsite seq found in the sequence and for a shuffled cutsite sequence
  # dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)  # likely no flush
  #                               maybe flush out the dict contents into csv and return empty dicts

  # maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = True)   # will flush due to forced flushing
  return


def load_genes_cutsites(inp_fn):
  pkl_file = os.path.dirname(inp_fn) + '/cutsites.pkl'
  if os.path.exists(pkl_file):
    cutsites = load_pickle(pkl_file)
    cutsites = cutsites.rename(columns={'Cutsite': 'target'})
    return cutsites

  all_lines = open(inp_fn, "r").readlines()
  sequence, chrom = '', ''
  data, cutsites = [], []
  for line in all_lines:
    if '>' in line:
      if sequence != '':
        data.append([chrom, sequence])
        if len(data) % 100 == 0:
          print('Working on: ', len(data))
        cutsites.extend(get_cutsites(chrom, sequence))
      chrom = line.strip().split('|')[3]
      sequence = ''
    else:
      sequence += line.strip()

  data.append([chrom, sequence]) # Adding last item
  print('Last item inserted: ', len(data))
  cutsites.extend(get_cutsites(chrom, sequence))

  print('Storing to file')
  all_data = pd.DataFrame(cutsites, columns=['Cutsite', 'Chromosome', 'Location', 'Orientation'])
  with open(pkl_file, 'wb') as f:
    pickle.dump(all_data, f)
  print('Gene cutsite complete')
  return cutsites


def load_ins_models(out_dir_model):
  return load_pickle(out_dir_model + 'rate_model.pkl'), load_pickle(out_dir_model + 'bp_model.pkl'), load_pickle(out_dir_model + 'Normalizer.pkl')


def load_neural_networks(out_dir_params):
  files = os.listdir(out_dir_params)
  nn_names = files[len(files) - 3:len(files) - 1]
  return load_pickle(out_dir_params + nn_names[0]), load_pickle(out_dir_params + nn_names[1])


def load_predictions(pred_file):
  return load_pickle(pred_file)


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
  prediction_file = ''

  if args.pred_file:
    prediction_file = args.pred_file

  if args.model_folder:
    exec_id = args.model_folder
    train_models = False

  if args.exec_type:
    execution_flow = args.exec_type
  else:
    execution_flow = 'both'

  return train_models, exec_id, prediction_file, execution_flow


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
  total_values = load_pickle(out_folder + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  rate_model, bp_model, normalizer = knn.train_knn(data, total_values, out_folder, out_folder + FOLDER_STAT_KEY)
  return nn_params, nn2_params, rate_model, bp_model, normalizer


def load_models(out_dir):
  helper.print_and_log("Loading models...", log_fn)
  nn_path = out_dir + FOLDER_PARAM_KEY
  files = os.listdir(nn_path)
  nn_names = files[len(files) - 3:len(files) - 1]
  return load_pickle(nn_path + nn_names[0]), load_pickle(nn_path + nn_names[1]), load_pickle(out_dir + 'rate_model.pkl'), load_pickle(out_dir + 'bp_model.pkl'), load_pickle(out_dir + 'Normalizer.pkl')


def calculate_predictions(data, models, in_del):
  # Getting the cutsites for human gene (approx 226,000,000)
  if in_del:
    helper.print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions = pred.predict_data_outcomes(data, models, in_del)
    predictions_file = f'{out_dir + FOLDER_PRED_KEY}in_del_distribution_mesc.pkl'
    if os.path.exists(predictions_file):
      predictions_file = f'{out_dir + FOLDER_PRED_KEY}in_del_distribution_u2os.pkl'
  else:
    helper.print_and_log("Loading Gene Cutsites...", log_fn)
    gene_data = load_genes_cutsites(data)
    # Calculating outcome using our models - only calculate approx 1,000,000
    helper.print_and_log("Predicting Sequence Outcomes...", log_fn)
    predictions = pred.predict_data_outcomes(gene_data, models, in_del)
    predictions_file = f'{out_dir + FOLDER_PRED_KEY}freq_distribution.pkl'

  helper.print_and_log("Storing Predictions...", log_fn)
  with open(predictions_file, 'wb') as out_file:
    pickle.dump(predictions, out_file)

  return predictions


def get_observed_values(data):
  grouped = data.groupby('Sample_Name')['Size'].apply(list).to_dict()
  grouped_res = {}
  # create deletion dicts
  for k, v in grouped.items():
    res = {}
    for i in range(1, 31):
      res[-i] = v.count(i)
    grouped_res[k] = res

  # add insertions
  for k, v in grouped_res.items():
    v[1] = len(data[(data['Sample_Name'] == k) & (data['Type'] == 'INSERTION') & (
      data['Indel'].str.startswith('1+'))])
    total = sum(v.values())
    for length, count in v.items():
      v[length] = count / total

  return grouped_res


def calculate_figure_3(train_model, prediction_file):
  fig3_predictions = None
  # Loading predictions if specified & file exists
  if prediction_file != '':
    if os.path.isfile(prediction_file):
      fig3_predictions = load_predictions(prediction_file)
    elif os.path.exists(out_dir + FOLDER_PRED_KEY + prediction_file):
      fig3_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + prediction_file)

  if fig3_predictions is None:
    if train_model:
      # Training model
      helper.print_and_log("Loading data...", log_fn)
      all_data_mesc = pd.concat(read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
      models_3 = model_creation(all_data_mesc, 'fig_3/')
    else:
      # Loading model
      model_folder = out_dir + 'fig_3/'
      models_3 = load_models(model_folder)
    # Making predictions from model
    fig3_predictions = calculate_predictions(input_dir + 'genes/mart_export.txt', models_3, False)

  helper.print_and_log("Plotting Figure...", log_fn)
  plt_3.hist(fig3_predictions, out_dir + FOLDER_GRAPH_KEY + 'plot_3f_' + exec_id + '.pdf')
  return


def calculate_figure_4(train_model, prediction_files):
  fig4a_predictions = None
  fig4b_predictions = None
  # Loading predictions if specified & file exists
  if prediction_files != '':
    prediction_file = [x.strip() for x in prediction_files.split(',')]
    if os.path.isfile(prediction_file[0]):
      fig4a_predictions = load_predictions(prediction_file[0])
    elif os.path.exists(out_dir + FOLDER_PRED_KEY + prediction_file[0]):
      fig4a_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + prediction_file[0])

    if os.path.isfile(prediction_file[1]):
      fig4b_predictions = load_predictions(prediction_file[1])
    elif os.path.exists(out_dir + FOLDER_PRED_KEY + prediction_file[1]):
      fig4b_predictions = load_predictions(out_dir + FOLDER_PRED_KEY + prediction_file[1])

  helper.print_and_log("Loading data...", log_fn)
  all_data_mesc = pd.concat(read_data(input_dir + 'dataset.pkl'), axis=1)
  all_data_mesc = all_data_mesc.reset_index()
  helper.print_and_log(f"mESC Loaded - Count(Items): {len(all_data_mesc)}", log_fn)
  all_data_u2os = pd.concat(read_data(input_dir + 'U2OS.pkl'), axis=1)
  all_data_u2os = all_data_u2os.reset_index()
  all_data_u2os = all_data_u2os.rename(columns={'deletionLength': 'Size'})

  helper.print_and_log(f"u2OS Loaded - Count(Items): {len(all_data_u2os)}", log_fn)
  # Reshuffling the data
  unique_mesc = np.random.choice(all_data_mesc['Sample_Name'].unique(), size=189, replace=False)
  test_mesc = all_data_mesc[all_data_mesc['Sample_Name'].isin(unique_mesc)]
  train_mesc = pd.merge(all_data_mesc, test_mesc, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

  # removing exception cases - aka deletions, with Homology length not 0 and no counter events
  wrong_grna = all_data_u2os[(all_data_u2os['Type'] == 'DELETION') & (all_data_u2os['homologyLength'] != 0)].groupby('Sample_Name').sum()
  wrong_grna = wrong_grna.reset_index()
  wrong_grna = wrong_grna[wrong_grna['countEvents'] == 0]['Sample_Name']
  all_data_u2os = all_data_u2os[all_data_u2os["Sample_Name"].isin(wrong_grna) == False]
  unique_u2os = np.random.choice(all_data_u2os['Sample_Name'].unique(), size=185, replace=False)
  test_u2os = all_data_u2os[all_data_u2os['Sample_Name'].isin(unique_u2os)]
  train_u2os = pd.merge(all_data_u2os, test_mesc, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

  # TODO: Discuss - unsure about this, cause if we're loading the predictions
  #  we'll be loading new train/test data so might have overlapping new test sets (which were used during training)
  #  Alternative - store & load observations
  if fig4a_predictions is None or fig4b_predictions is None:
    if train_model:
      # Models for Figure 4
      models_4a = model_creation(train_mesc, 'fig_4mesc/')
      models_4b = model_creation(train_u2os, 'fig_4u20s/')
    else:
      models_4a = load_models(out_dir + 'fig_4mesc/')
      models_4b = load_models(out_dir + 'fig_4u20s/')

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

    fig4a_predictions = calculate_predictions(test_mesc_targets, models_4a, True)
    fig4b_predictions = calculate_predictions(test_u2os_targets, models_4b, True)

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
  parser.add_argument('--pred_file', dest='pred_file', type=str, help='File name used to predict outcomes')
  args = parser.parse_args()
  train_models, user_exec_id, prediction_file, execution_flow = get_args(args)

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
    calculate_figure_3(train_models, prediction_file)
  elif execution_flow == '4b':
    calculate_figure_4(train_models, prediction_file)
  else:
    calculate_figure_3(train_models, prediction_file)
    calculate_figure_4(train_models, prediction_file)

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


