import os, datetime
import pickle
from collections import defaultdict

import pandas as pd
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import _config, _predict
from mylib import util
from d2_model import alphabetize, count_num_folders, print_and_log, save_train_test_names \
  , init_random_params, main_objective, rsq, save_parameters, adam_minmin

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# import fi2_ins_ratio
# import fk_1bpins


def parse_data(merged):
  deletions = merged[merged['Type'] == 'DELETION']
  deletions = deletions.reset_index()

  # A single value GRNA -Train until 1871
  exps = deletions['Sample_Name'].unique()[:5]
  # Q & A: How to determine if a deletion is MH or MH-less - length != 0
  # Question: Do we need to distinguish between MH and MH-less, if yes, how to pass diff del_len to MH-less NN

  microhomologies = deletions[deletions['homologyLength'] != 0]
  # mh_less = deletions[deletions['homologyLength'] == 0]
  mh_lens, gc_fracs, del_lens, freqs, dl_freqs = [], [], [], [], []
  for id, exp in enumerate(exps):
    # Microhomology computation
    mh_exp_data = microhomologies[microhomologies['Sample_Name'] == exp][
      ['countEvents', 'homologyLength', 'homologyGCContent', 'Size']]

    # Normalize Counts
    total_count = sum(mh_exp_data['countEvents'])
    mh_exp_data['countEvents'] = mh_exp_data['countEvents'].div(total_count)

    freqs.append(mh_exp_data['countEvents'])
    mh_lens.append(mh_exp_data['homologyLength'].astype('int32'))
    gc_fracs.append(mh_exp_data['homologyGCContent'])
    del_lens.append(mh_exp_data['Size'].astype('int32'))

    curr_dl_freqs = []
    dl_freq_df = mh_exp_data[mh_exp_data['Size'] <= 28]
    for del_len in range(1, 28 + 1):
      dl_freq = sum(dl_freq_df[dl_freq_df['Size'] == del_len]['countEvents'])
      curr_dl_freqs.append(dl_freq)
    dl_freqs.append(curr_dl_freqs)

    # # Microhomology-less computation
    # mh_less_exp_data = mh_less[mh_less['Sample_Name'] == exp][['countEvents', 'Size']]
    # del_lens.append(mh_exp_data['Size'].astype('int32'))

  return exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs


def initialize_files_and_folders():
  # Set output location of model & params
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
  util.ensure_dir_exists(out_place)

  num_folds = count_num_folders(out_place)
  out_letters = alphabetize(num_folds + 1)
  out_dir = out_place + out_letters + '/'

  out_dir_params = out_place + out_letters + '/parameters/'
  util.ensure_dir_exists(out_dir_params)

  log_fn = out_dir + '_log_%s.out' % out_letters
  with open(log_fn, 'w') as f:
    pass
  print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, out_dir_params, out_letters


def initialize_model():
  # Model Settings & params
  seed = npr.RandomState(1)

  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]
  return seed, nn_layer_sizes, nn2_layer_sizes


def read_data(file):
  master_data = pickle.load(open(file, 'rb'))
  return master_data['counts'], master_data['del_features']


def train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, out_dir_params, out_letters):
  param_scale = 0.1
  # num_epochs = 7*200 + 1
  num_epochs = 50
  step_size = 0.10
  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs=seed)
  init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans

  batch_size = 200
  num_batches = int(np.ceil(len(INP_train) / batch_size))

  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)

  def objective(nn_params, nn2_params, iter):
    idx = batch_indices(iter)
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)

  both_objective_grad = grad(objective, argnum=[0, 1])

  def print_perf(nn_params, nn2_params, iter):
    print_and_log(str(iter), log_fn)
    if iter % 5 != 0:
      return None

    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size,
                                seed)
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test),
                               seed)

    tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)

    out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (
      iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
    print_and_log(out_line, log_fn)

    if iter % 20 == 0:
      letters = alphabetize(int(iter / 10))
      print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2", log_fn)
      print_and_log('%s %s %s' % (datetime.datetime.now(), out_letters, letters), log_fn)
      save_parameters(nn_params, nn2_params, out_dir_params, letters)
      if iter >= 10:
        pass
        # plot_mh_score_function(nn_params, out_dir, letters + '_nn')
        # plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train, 'train', letters)
        # plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test, 'test', letters)

    return None

  optimized_params = adam_minmin(both_objective_grad, init_nn_params, init_nn2_params, step_size=step_size,
                                 num_iters=num_epochs, callback=print_perf)


def neural_networks(merged):
  seed, nn_layer_sizes, nn2_layer_sizes = initialize_model()
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(merged)

  print_and_log("Parsing data...", log_fn)
  INP = []
  for mhl, gcf in zip(mh_lens, gc_fracs):
    inp_point = np.array([mhl, gcf]).T  # N * 2
    INP.append(inp_point)
  INP = np.array(INP)  # 2000 * N * 2
  # Neural network considers each N * 2 input, transforming it into N * 1 output.
  OBS = np.array(freqs)
  OBS2 = np.array(dl_freqs)
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  print_and_log("Training model...", log_fn)
  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  save_train_test_names(NAMES_train, NAMES_test, out_dir)
  train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, out_dir_params, out_letters)


def load_statistics(data_nm):
  stats_csv_1, stats_csv_2 = prepare_statistics(data_nm)

  # if not os.path.isfile(stats_csv_fn):
  #   print('Running statistics from scratch...')
  #   stats_csv_1, stats_csv_2 = prepare_statistics(data_nm)
  #   # TODO: fix here - file override eachother
  #   stats_csv_1.to_csv(stats_csv_fn)
  #   stats_csv_2.to_csv(stats_csv_fn)
  # else:
  #   print('Getting statistics from file...')
  #   stats_csv = pd.read_csv(stats_csv_fn, index_col=0)
  print('Done')
  return stats_csv_1, stats_csv_2


def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name
  bp_ins_df = defaultdict(list)
  ins_ratio_df = defaultdict(list)

  timer = util.Timer(total=len(data_nm))
  exps = data_nm['Sample_Name'].unique()[:10]

  data_nm['delta'] = data_nm['Indel'].str.extract(r'(\d+)', expand=True)
  data_nm['nucleotide'] = data_nm['Indel'].str.extract(r'([A-Z]+)', expand=True)
  data_nm['delta'] = data_nm['delta'].astype('int32')

  for id, exp in enumerate(exps):
    exp_data = data_nm[data_nm['Sample_Name'] == exp]
    calc_ins_ratio_statistics(exp_data, exp, ins_ratio_df)
    calc_1bp_ins_statistics(exp_data, exp, bp_ins_df)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  ins_stat = pd.DataFrame(ins_ratio_df)
  bp_stat = pd.DataFrame(bp_ins_df)
  return ins_stat, bp_stat


def calc_ins_ratio_statistics(all_data, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions
  total_ins_del_counts = sum(all_data['countEvents'])
  if total_ins_del_counts <= 1000:
    return

  editing_rate = 1 # always 1 since sum(in or del) / sum(in or del which aren't noise)
  ins_count = sum(all_data[(all_data['Type'] == 'INSERTION') & (all_data['delta'] == 1)]['countEvents'])
  del_count = sum(all_data[all_data['Type'] == 'DELETION']['countEvents']) # need to check - Indel with Mismatches
  mhdel_count = sum(all_data[(all_data['Type'] == 'DELETION') & (all_data['homologyLength'] != 0)]['countEvents']) # need to check - Indel with Mismatches

  ins_ratio = ins_count / total_ins_del_counts
  fivebase = exp[len(exp)-4]

  del_score = 0.02 #TODO need to find deletion score function(total_deletion_score) - maybe c6_polish.py?
  norm_entropy = 0.02 #TODO need to find deletion length distribution function(deletion_length_distribution) - maybe c6_polish.py?

  # local_seq = exp[len(exp) - 4:len(exp) + 4] # TODO - fix - +4 will fail - need to get sequence from libA.txt
  local_seq = exp[len(exp) - 4:len(exp)]
  gc = (local_seq.count('C') + local_seq.count('G')) / len(local_seq)

  if fivebase == 'A':
    fivebase_oh = np.array([1, 0, 0, 0])
  if fivebase == 'C':
    fivebase_oh = np.array([0, 1, 0, 0])
  if fivebase == 'G':
    fivebase_oh = np.array([0, 0, 1, 0])
  if fivebase == 'T':
    fivebase_oh = np.array([0, 0, 0, 1])

  threebase = exp[len(exp)-3]

  if threebase == 'A':
    threebase_oh = np.array([1, 0, 0, 0])
  if threebase == 'C':
    threebase_oh = np.array([0, 1, 0, 0])
  if threebase == 'G':
    threebase_oh = np.array([0, 0, 1, 0])
  if threebase == 'T':
    threebase_oh = np.array([0, 0, 0, 1])

  alldf_dict['Editing Rate'].append(editing_rate)
  alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))
  alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
  alldf_dict['Ins1bp Ratio'].append(ins_ratio)
  alldf_dict['Fivebase'].append(fivebase)
  alldf_dict['Del Score'].append(del_score)
  alldf_dict['Entropy'].append(norm_entropy)
  alldf_dict['GC'].append(gc)
  alldf_dict['Fivebase_OH'].append(fivebase_oh)
  alldf_dict['Threebase'].append(threebase)
  alldf_dict['Threebase_OH'].append(threebase_oh)
  alldf_dict['_Experiment'].append(exp)
  return alldf_dict


def calc_1bp_ins_statistics(all_data, exp, alldf_dict):
  # Normalize Counts
  total_count = sum(all_data['countEvents'])
  all_data['Frequency'] = all_data['countEvents'].div(total_count)

  insertions = all_data[all_data['Type'] == 'INSERTION']
  insertions = insertions[insertions['delta'] == 1]

  if sum(insertions['countEvents']) <= 100:
    return

  freq = sum(insertions['Frequency']) # TODO check if Frequency can be removed
  a_frac = sum(insertions[insertions['nucleotide'] == 'A']['Frequency']) / freq
  c_frac = sum(insertions[insertions['nucleotide'] == 'C']['Frequency']) / freq
  g_frac = sum(insertions[insertions['nucleotide'] == 'G']['Frequency']) / freq
  t_frac = sum(insertions[insertions['nucleotide'] == 'T']['Frequency']) / freq
  alldf_dict['Frequency'].append(freq)
  alldf_dict['A frac'].append(a_frac)
  alldf_dict['C frac'].append(c_frac)
  alldf_dict['G frac'].append(g_frac)
  alldf_dict['T frac'].append(t_frac)

  fivebase = exp[len(exp)-4]
  alldf_dict['Base'].append(fivebase)

  alldf_dict['_Experiment'].append(exp) # TODO check if _Experiment can be removed
  return alldf_dict


def featurize(rate_stats, Y_nm):
  fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
  threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

  ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
  del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
  print(ent.shape, fivebases.shape, del_scores.shape)

  Y = np.array(rate_stats[Y_nm])
  print(Y_nm)

  Normalizer = [(np.mean(fivebases.T[2]),
                 np.std(fivebases.T[2])),
                (np.mean(fivebases.T[3]),
                 np.std(fivebases.T[3])),
                (np.mean(threebases.T[0]),
                 np.std(threebases.T[0])),
                (np.mean(threebases.T[2]),
                 np.std(threebases.T[2])),
                (np.mean(ent),
                 np.std(ent)),
                (np.mean(del_scores),
                 np.std(del_scores)),
                ]

  fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
  fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
  threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
  threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
  gtag = np.array([fiveG, fiveT, threeA, threeG]).T

  ent = (ent - np.mean(ent)) / np.std(ent)
  del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)

  X = np.concatenate((gtag, ent, del_scores), axis=1)
  X = np.concatenate((gtag, ent, del_scores), axis=1)
  feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
  print('Num. samples: %s, num. features: %s' % X.shape)

  return X, Y, Normalizer


def generate_models(X, Y, bp_stats, Normalizer):
  # Train rate model
  model = KNeighborsRegressor()
  model.fit(X, Y)
  with open(out_dir + 'rate_model_v2.pkl', 'w') as f:
    pickle.dump(model, f)

  # Obtain bp stats
  bp_model = dict()
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
  t_melt = pd.melt(bp_stats,
                   id_vars = ['Base'],
                   value_vars = ins_bases,
                   var_name = 'Ins Base',
                   value_name = 'Fraction')
  for base in list('ACGT'):
    bp_model[base] = dict()
    mean_vals = []
    for ins_base in ins_bases:
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
      mean_vals.append(float(np.mean(t_melt[crit])))
    for bp, freq in zip(list('ACGT'), mean_vals):
      bp_model[base][bp] = freq / sum(mean_vals)

  with open(out_dir + 'bp_model_v2.pkl', 'w') as f:
    pickle.dump(bp_model, f)

  with open(out_dir + 'Normalizer_v2.pkl', 'w') as f:
    pickle.dump(Normalizer, f)

  return


def knn(merged):
  rate_stats, bp_stats = load_statistics(merged)
  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  X, Y, Normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, bp_stats, Normalizer)


def predict_all_items():
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'

  # _predict.init_model(run_iter='aax', param_iter='aag')
  # _predict.predict_all()


if __name__ == '__main__':
  out_dir, log_fn, out_dir_params, out_letters = initialize_files_and_folders()
  print_and_log("Loading data...", log_fn)
  counts, del_features = read_data('../in/' + 'dataset.pkl')
  merged = pd.concat([counts, del_features], axis=1)
  merged = merged.reset_index()
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  # neural_networks(merged)

  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  knn(merged)