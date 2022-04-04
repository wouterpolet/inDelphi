import os
import glob
import pickle
import datetime
import pandas as pd
import autograd.numpy as np
import functionality.author_helper as ah

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_GRAPH_KEY = 'plots/'
FOLDER_LOG_KEY = 'logs/'
FOLDER_PRED_KEY = 'predictions/'
FOLDER_INPUT_KEY = '/in/'

RATE_MODEL_NAME = 'rate_model.pkl'
BP_MODEL_NAME = 'bp_model.pkl'
NORMALIZER_NAME = 'Normalizer.pkl'

EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))
INPUT_DIRECTORY = EXECUTION_PATH + FOLDER_INPUT_KEY
SAMPLE_SIZE = 1003524
BATCH_SIZE = 20000000

PREDICTION_FILE_3 = 'freq_distribution'
PREDICTION_FILE_4 = 'in_del_distribution_'

def load_pickle(file):
  return pickle.load(open(file, 'rb'))


def read_data(file):
  master_data = load_pickle(file)
  return master_data['counts'], master_data['del_features']


def initialize_files_and_folders(user_exec_id):
  """
  Initialize output folders and needed files (for logging)
  @param user_exec_id: the unique execution id of the model
  @return: out_dir: output directory of the model
           log_fn: log file
           exec_id: execution id pointing to the output directory
  """
  # Set output location of model & params
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
  ah.ensure_dir_exists(out_place)
  exec_id = ''
  output_date_format = "%Y%m%d_%H%M"
  # num_folds = helper.count_num_folders(out_place)
  if user_exec_id == '' or ah.count_num_folders(out_place) < 1:
    exec_id = datetime.datetime.now().strftime(output_date_format)
  else:
    latest = datetime.datetime.strptime('1990/01/01', '%Y/%m/%d')
    for name in os.listdir(out_place):
      try:
        datetime.datetime.strptime(name, output_date_format)
      except ValueError:
        if name == user_exec_id:
          exec_id = name
          break
        else:
          continue
      date_time_obj = datetime.datetime.strptime(name, output_date_format)
      if name == user_exec_id:
        latest = date_time_obj
        break
      if latest < date_time_obj:
        latest = date_time_obj
    if exec_id == '':
      exec_id = latest.strftime(output_date_format)

  out_dir = out_place + exec_id + '/'
  ah.ensure_dir_exists(out_dir + FOLDER_PRED_KEY)
  ah.ensure_dir_exists(out_dir + FOLDER_GRAPH_KEY)
  ah.ensure_dir_exists(out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY)
  # ah.ensure_dir_exists(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY)
  ah.ensure_dir_exists(out_dir + FOLDER_LOG_KEY)

  log_fn = out_dir + FOLDER_LOG_KEY + '_log_%s.out' % datetime.datetime.now().strftime("%Y%m%d_%H%M")
  with open(log_fn, 'w') as f:
    pass
  ah.print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, exec_id


def load_lib_data(folder_dir, libX):
  """
  Load libA or libB data from text files
  @param folder_dir: folder directory from where to read lib
  @param libX: libA or libB
  @return: dataframe made up of names, grnas and targets
  """
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


def get_targets(libX, data):
  """
  get full target(s) using the sample name from our dataset using the LibA/LibB grnas
  @rtype: dictionary of all sequences
  """
  result = {}
  for sampleName in data['Sample_Name'].unique():
    grna = sampleName.split('_')
    grna = grna[len(grna) - 1]
    sequences = libX.loc[libX['target'].str.contains(grna, case=False)]['target']
    if len(sequences) == 1:
      result[grna] = [sequences.values[0]]
    else:
      all_seqs = [seq for seq in sequences if seq.index(grna) == 10]
      result[grna] = all_seqs
  return result


def load_ins_models(out_dir_model):
  """
  Load pretrained insertion models
  @param out_dir_model: output directory of the trained models
  @return: rate model, bp model, normalizer
  """
  return load_pickle(out_dir_model + RATE_MODEL_NAME), load_pickle(out_dir_model + BP_MODEL_NAME), load_pickle(out_dir_model + NORMALIZER_NAME)


def load_neural_networks(out_dir_params):
  """
  Load both neural networks from the output directory specified
  Always take the ones generated from the last epoch
  @param out_dir_params:
  @return: NN1: MH Deletions Network
           NN2: MH-Less Deletions Network
  """
  nn_files = glob.glob(out_dir_params + "*_nn.pkl")
  nn_files.sort(reverse=True)
  nn2_files = glob.glob(out_dir_params + "*_nn2.pkl")
  nn2_files.sort(reverse=True)
  return load_pickle(nn_files[0]), load_pickle(nn2_files[0])


def load_models(out_dir):
  """
  Load pretrained model from directory
  @param out_dir: model directory
  @return: dictionary of models
  """
  nn_path = out_dir + FOLDER_PARAM_KEY
  nn, nn_2 = load_neural_networks(nn_path)
  rate, bp, norm = load_ins_models(out_dir)
  model = {'nn': nn, 'nn_2': nn_2, 'rate': rate, 'bp': bp, 'norm': norm}
  return model


def load_total_phi_delfreq(out_dir):
  return load_pickle(out_dir + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')


def store_predictions(out_dir, file, predictions):
  with open(out_dir + FOLDER_PRED_KEY + file, 'wb') as out_file:
    pickle.dump(predictions, out_file)


def load_predictions(out_dir, in_del):
  # files = os.listdir(out_dir + FOLDER_PRED_KEY)
  files = glob.glob(out_dir + FOLDER_PRED_KEY + '*.pkl')
  if in_del:
    mesc_file = glob.glob(out_dir + FOLDER_PRED_KEY + PREDICTION_FILE_4 + 'mesc.pkl')
    u2os_files = glob.glob(out_dir + FOLDER_PRED_KEY + PREDICTION_FILE_4 + 'u2os.pkl')
    if len(mesc_file) == 1 and len(u2os_files) == 1:
      return load_pickle(mesc_file[0]), load_pickle(u2os_files[0])
  else:
    distribution = glob.glob(out_dir + FOLDER_PRED_KEY + PREDICTION_FILE_3 + '.pkl')
    if len(distribution) == 1:
      return load_pickle(distribution[0])
  return None


def load_statistics(out_dir_stat):
  ins_stat_dir = out_dir_stat + 'ins_stat.csv'
  bp_stat_dir = out_dir_stat + 'bp_stat.csv'

  if os.path.isfile(ins_stat_dir) and os.path.isfile(bp_stat_dir):
    ins_stat = pd.read_csv(ins_stat_dir, index_col=0)
    bp_stat = pd.read_csv(bp_stat_dir, index_col=0)
  else:
    raise Exception('Invalid insertion and bp statistics files')
  return ins_stat, bp_stat


def convert_oh_string_to_nparray(input):
  input = str(input).replace('[', '').replace(']', '')
  nums = input.split(' ')
  return np.array([int(s) for s in nums])

