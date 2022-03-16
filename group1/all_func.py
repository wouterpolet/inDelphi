import argparse, datetime, os, pickle, warnings, re
from collections import defaultdict
import glob
import plot_3f as plt

import autograd.numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from scipy.stats import entropy

import helper
import util as util

import neural_networks as nn
import ins_network as knn
import prediction as pred

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)


DELETION_LEN_LIMIT = 28
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_STAT_KEY = 'statistics/'
FOLDER_MODEL_KEY = 'model/'


def initialize_files_and_folders(user_exec_id):
  # Set output location of model & params
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
  util.ensure_dir_exists(out_place)

  # num_folds = helper.count_num_folders(out_place)
  if user_exec_id == '' or helper.count_num_folders(out_place) < 1:
    exec_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
  else:
    latest = datetime.datetime.strptime('1990/01/01', '%Y/%m/%d')
    for name in os.listdir(out_place):
      date_time_obj = datetime.datetime.strptime(name, "%Y%m%d_%H%M")
      if name == user_exec_id:
        latest = date_time_obj
        break
      if latest < date_time_obj:
        latest = date_time_obj
    exec_id = latest.strftime("%Y%m%d_%H%M")

  # if use_prev and num_folds >= 1:
  #   out_letters = helper.alphabetize(num_folds - 1)
  # else:
  #   out_letters = helper.alphabetize(num_folds)

  out_dir = out_place + exec_id + '/'
  util.ensure_dir_exists(out_dir + FOLDER_PARAM_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_STAT_KEY)
  util.ensure_dir_exists(out_dir + FOLDER_MODEL_KEY)

  log_fn = out_dir + '_log_%s.out' % exec_id
  with open(log_fn, 'w') as f:
    pass
  helper.print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, exec_id


def _pickle_load(file):
  data = pickle.load(open(file, 'rb'))
  return data


def load_model(file):
  return _pickle_load(file)


def read_data(file):
  master_data = _pickle_load(file)
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
    cutsites = _pickle_load(pkl_file)
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
  return load_model(out_dir_model + 'rate_model.pkl'), load_model(out_dir_model + 'bp_model.pkl'), load_model(out_dir_model + 'Normalizer.pkl')


def load_neural_networks(out_dir_params):
  files = os.listdir(out_dir_params)
  nn_names = files[len(files) - 3:len(files) - 1]
  return load_model(out_dir_params + nn_names[0]), load_model(out_dir_params + nn_names[1])


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
  if args.model_folder:
    exec_id = args.model_folder
    train_models = False
  else:
    exec_id = ''
    train_models = True

  if args.only_fig:
    only_plots = args.only_fig == 'True'
  else:
    only_plots = False

  if args.pred_file:
    prediction_file = args.pred_file
    libX = 'libA'
  else:
    prediction_file = ''
    libX = 'libB'

    return train_models, exec_id, only_plots, prediction_file, libX


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execution Details')
  parser.add_argument('--model_folder', dest='model_folder', type=str, help='Variable indicating the execution id of the trained neural network and knn')
  parser.add_argument('--pred_file', dest='pred_file', type=str, help='File name used to predict outcomes')
  parser.add_argument('--plot_fig_only', dest='only_fig', type=str, help='TODO fill here')

  args = parser.parse_args()
  train_models, user_exec_id, only_plots, prediction_file, libX = get_args(args)

  execution_path = os.path.dirname(os.path.dirname(__file__))
  input_dir = execution_path + '/in/'
  libX_dir = execution_path + '/data-libprocessing/'
  if prediction_file == '':
    prediction_file = libX_dir

  out_dir, log_fn, exec_id = initialize_files_and_folders(user_exec_id)

  helper.print_and_log("Loading data...", log_fn)

  counts, del_features = read_data(input_dir + 'dataset.pkl')
  # counts, del_features = read_data(input_dir + 'U2OS.pkl')
  all_data = pd.concat([counts, del_features], axis=1)
  all_data = all_data.reset_index()
  helper.print_and_log(f"Data Loaded - Items in Dataframe: {len(all_data)}", log_fn)

  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  if train_models:
    helper.print_and_log("Training Neural Networks...", log_fn)
    nn_params, nn2_params = nn.create_neural_networks(all_data, log_fn, out_dir, exec_id)

    helper.print_and_log("Training KNN...", log_fn)
    total_values = load_model(out_dir + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
    rate_model, bp_model, normalizer = knn.train_knn(all_data, total_values, out_dir + FOLDER_MODEL_KEY, out_dir + FOLDER_STAT_KEY)
  else:
    helper.print_and_log("Loading Neural Networks...", log_fn)
    nn_params, nn2_params = load_neural_networks(out_dir + FOLDER_PARAM_KEY)
    helper.print_and_log("Loading KNN...", log_fn)
    rate_model, bp_model, normalizer = load_ins_models(out_dir + FOLDER_MODEL_KEY)

  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''

  # Using Chromosome Data
  # inp_fn = input_dir + 'Homo_sapiens.GRCh38.dna.chromosome.1.fa'
  # inp_fn = input_dir + 'chromosome/'
  # find_cutsites_and_predict(inp_fn, use_file='')
  helper.print_and_log("Loading Gene Cutsites...", log_fn)
  inp_fn = input_dir + 'genes/mart_export.txt'
  gene_data = load_genes_cutsites(inp_fn)
  helper.print_and_log("Predicting Sequence Outcomes from cutsites...", log_fn)
  predictions = pred.bulk_predict_all(gene_data, nn_params, nn2_params, rate_model, bp_model, normalizer)
  helper.print_and_log("Storing Predictions...", log_fn)

  output_extended_predictions_file = f'{out_dir + FOLDER_STAT_KEY}extended_prediction_output_{exec_id}.pkl'
  with open(output_extended_predictions_file, 'wb') as out_file:
    pickle.dump(predictions, out_file)

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

  print('Plotting Graphs - 3f')
  # fig_3f_data_del = predictions['Highest Del Rate'].apply(lambda x: x*100)
  # fig_3f_data_ins = predictions['Highest Ins Rate'].apply(lambda x: x*100)
  # plt.hist(fig_3f_data_del, out_dir_stat + 'del_plot_3f_' + libX)
  # plt.hist(fig_3f_data_ins, out_dir_stat + 'ins_plot_3f_' + libX)
  # Plotting Image 3f
  # plt.hist(predictions, out_dir_stat + 'plot_3f_' + libX)

  print('Plotting Graphs')

# Run bulk prediction once out of 300 times
# mart_export.txtcutsites.pkl