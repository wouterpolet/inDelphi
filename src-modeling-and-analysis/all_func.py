import argparse
import copy
import datetime
import os
import pickle
import random
import warnings
import re
from collections import defaultdict

import autograd.numpy as np
import autograd.numpy.random as npr
import pandas as pd
from autograd import grad
from autograd.misc import flatten
from pandas.core.common import SettingWithCopyWarning
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# import _config, _predict
import util as util
from d2_model import alphabetize, count_num_folders, print_and_log, save_train_test_names \
  , init_random_params, rsq, save_parameters, nn_match_score_function

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_data(merged):
  deletions = merged[merged['Type'] == 'DELETION']
  deletions = deletions.reset_index()

  # A single value GRNA -Train until 1871
  exps = deletions['Sample_Name'].unique()
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


def initialize_files_and_folders(use_prev):
  # Set output location of model & params
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
  util.ensure_dir_exists(out_place)

  num_folds = count_num_folders(out_place)
  if use_prev and num_folds >= 1:
    out_letters = alphabetize(num_folds - 1)
  else:
    out_letters = alphabetize(num_folds)

  out_dir = out_place + out_letters + '/'
  out_dir_params = out_place + out_letters + '/parameters/'
  out_dir_stat = out_place + out_letters + '/statistics/'
  out_dir_model = out_place + out_letters + '/model/'
  out_dir_exin = out_place + out_letters + '/exon_intron/'
  util.ensure_dir_exists(out_dir_params)
  util.ensure_dir_exists(out_dir_stat)
  util.ensure_dir_exists(out_dir_model)
  util.ensure_dir_exists(out_dir_exin)

  log_fn = out_dir + '_log_%s.out' % out_letters
  with open(log_fn, 'w') as f:
    pass
  print_and_log('out dir: ' + out_dir, log_fn)

  return out_dir, log_fn, out_dir_params, out_dir_stat, out_dir_model, out_dir_exin, out_letters


def initialize_model():
  # Model Settings & params
  seed = npr.RandomState(1)

  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]
  return seed, nn_layer_sizes, nn2_layer_sizes


def _pickle_load(file):
  data = pickle.load(open(file, 'rb'))
  return data


def load_model(file):
  return _pickle_load(file)


def read_data(file):
  master_data = _pickle_load(file)
  return master_data['counts'], master_data['del_features']


##
# Objective
##
def main_objective(nn_params, nn2_params, inp, obs, obs2, del_lens, num_samples, rs, iter=0):
  LOSS = 0
  test1 = []
  total_phi_del_freq = []  # 1961 x 1
  for idx in range(len(inp)):  # for each gRNA
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25 * dl)
        mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    # Pearson correlation squared loss
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(obs[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean) * (obs[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean) ** 2))
    pearson_denom_y = np.sqrt(np.sum((obs[idx] - y_mean) ** 2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    rsq = (pearson_numerator / pearson_denom) ** 2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

    #
    # I want to make sure nn2 never outputs anything negative.
    # Sanity check during training.
    #

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28 + 1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25 * np.arange(1, 28 + 1))
    mh_less_phi_total = np.sum(unnormalized_nn2, dtype=np.float64)

    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    mh_contribution = np.zeros(28, )
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25 * dl)
      mask = np.concatenate([np.zeros(dl - 1, ), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1, )])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))


    # Pearson correlation squared loss
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(obs2[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean) * (obs2[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean) ** 2))
    pearson_denom_y = np.sqrt(np.sum((obs2[idx] - y_mean) ** 2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    rsq = (pearson_numerator / pearson_denom) ** 2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

    if not isinstance(mh_phi_total, float):
      mh_phi_total = mh_phi_total._value
    if not isinstance(mh_less_phi_total, float):
      mh_less_phi_total = mh_less_phi_total._value

    # Calculate norm entropy
    mh_total = mh_phi_total + mh_less_phi_total
    # Convert to list from arraybox
    normalized_fq_list = []
    for item in normalized_fq:
      value = item
      if not isinstance(item, float):
        value = item._value
      normalized_fq_list.append(value)
    # normalized_fq_list = [item._value for item in normalized_fq]
    norm_entropy = entropy(normalized_fq_list) / np.log(len(normalized_fq_list))

    # Append to list for storing
    total_phi_del_freq.append([NAMES[idx], mh_total, norm_entropy])

  if iter == num_epochs - 1:
    column_names = ["exp", "total_phi", "norm_entropy"]
    df = pd.DataFrame(total_phi_del_freq, columns=column_names)
    df.to_pickle(out_dir_params + 'total_phi_delfreq.pkl')

  return LOSS / num_samples

##
# ADAM Optimizer
##
def exponential_decay(step_size):
  if step_size > 0.001:
      step_size *= 0.999
  return step_size


def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9,
                b2=0.999, eps=10 ** -8):
  x_nn, unflatten_nn = flatten(init_params_nn)
  x_nn2, unflatten_nn2 = flatten(init_params_nn2)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
  for i in range(num_iters):
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    g_nn, _ = flatten(g_nn_uf)
    g_nn2, _ = flatten(g_nn2_uf)

    if callback:
      callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)

    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn ** 2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1 ** (i + 1))  # Bias correction.
    vhat_nn = v_nn / (1 - b2 ** (i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)

    # Update parameters
    m_nn2 = (1 - b1) * g_nn2 + b1 * m_nn2  # First  moment estimate.
    v_nn2 = (1 - b2) * (g_nn2 ** 2) + b2 * v_nn2  # Second moment estimate.
    mhat_nn2 = m_nn2 / (1 - b1 ** (i + 1))  # Bias correction.
    vhat_nn2 = v_nn2 / (1 - b2 ** (i + 1))
    x_nn2 = x_nn2 - step_size * mhat_nn2 / (np.sqrt(vhat_nn2) + eps)
  return unflatten_nn(x_nn), unflatten_nn2(x_nn2)


def train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, out_dir_params, out_letters):
  param_scale = 0.1
  # num_epochs = 7*200 + 1
  global num_epochs
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
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed, iter=iter)

  both_objective_grad = grad(objective, argnum=[0, 1])

  def print_perf(nn_params, nn2_params, iter):
    print_and_log(str(iter), log_fn)
    if iter % 5 != 0:
      return None

    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

    tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)

    out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (
      iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
    print_and_log(out_line, log_fn)

    if iter % 10 == 0:
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
  return optimized_params


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
  global NAMES
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  print_and_log("Training model...", log_fn)
  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  save_train_test_names(NAMES_train, NAMES_test, out_dir)
  return train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, out_dir_params, out_letters)


def load_statistics(data_nm, total_values):
  ins_stat_dir = out_dir_stat + 'ins_stat.csv'
  bp_stat_dir = out_dir_stat + 'bp_stat.csv'
  if os.path.isfile(ins_stat_dir) and os.path.isfile(bp_stat_dir):
    print('Loading statistics...')
    ins_stat = pd.read_csv(ins_stat_dir, index_col=0)
    bp_stat = pd.read_csv(bp_stat_dir, index_col=0)
  else:
    print('Creating statistics...')
    ins_stat, bp_stat = prepare_statistics(data_nm, total_values)
    ins_stat.to_csv(ins_stat_dir)
    bp_stat.to_csv(bp_stat_dir)
  return ins_stat, bp_stat


def prepare_statistics(data_nm, total_values):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name
  bp_ins_df = defaultdict(list)
  ins_ratio_df = defaultdict(list)

  timer = util.Timer(total=len(data_nm))
  exps = data_nm['Sample_Name'].unique()

  data_nm['delta'] = data_nm['Indel'].str.extract(r'(\d+)', expand=True)
  data_nm['nucleotide'] = data_nm['Indel'].str.extract(r'([A-Z]+)', expand=True)
  data_nm['delta'] = data_nm['delta'].astype('int32')

  for id, exp in enumerate(exps):
    exp_data = data_nm[data_nm['Sample_Name'] == exp]
    calc_ins_ratio_statistics(exp_data, exp, ins_ratio_df, total_values)
    calc_1bp_ins_statistics(exp_data, exp, bp_ins_df)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  ins_stat = pd.DataFrame(ins_ratio_df)
  bp_stat = pd.DataFrame(bp_ins_df)
  return ins_stat, bp_stat


def sigmoid(x):
  return 0.5 * (np.tanh(x) + 1.0)


def calc_ins_ratio_statistics(all_data, exp, alldf_dict, total_values):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions
  total_ins_del_counts = sum(all_data['countEvents'])
  if total_ins_del_counts <= 1000:
    return

  editing_rate = 1  # always 1 since sum(in or del) / sum(in or del which aren't noise)
  ins_count = sum(all_data[(all_data['Type'] == 'INSERTION') & (all_data['delta'] == 1)]['countEvents'])
  del_count = sum(all_data[all_data['Type'] == 'DELETION']['countEvents'])
  mhdel_count = sum(all_data[(all_data['Type'] == 'DELETION') & (all_data['homologyLength'] != 0)]['countEvents'])

  ins_ratio = ins_count / total_ins_del_counts
  fivebase = exp[len(exp) - 4]

  if len(total_values[total_values['exp'] == exp]) > 0:
    del_score = total_values[total_values['exp'] == exp]['total_phi'].values[0]
    norm_entropy = total_values[total_values['exp'] == exp]['norm_entropy'].values[0]
  else:
    del_score = 0
    norm_entropy = 0

  # local_seq = exp[len(exp) - 4:len(exp) + 4] # TODO - fix - +4 will fail - need to get sequence from libA.txt
  # This is not needed
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

  threebase = exp[len(exp) - 3]

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

  freq = sum(insertions['Frequency'])  # TODO check if Frequency can be removed
  a_frac = sum(insertions[insertions['nucleotide'] == 'A']['Frequency']) / freq
  c_frac = sum(insertions[insertions['nucleotide'] == 'C']['Frequency']) / freq
  g_frac = sum(insertions[insertions['nucleotide'] == 'G']['Frequency']) / freq
  t_frac = sum(insertions[insertions['nucleotide'] == 'T']['Frequency']) / freq
  alldf_dict['Frequency'].append(freq)
  alldf_dict['A frac'].append(a_frac)
  alldf_dict['C frac'].append(c_frac)
  alldf_dict['G frac'].append(g_frac)
  alldf_dict['T frac'].append(t_frac)

  fivebase = exp[len(exp) - 4]
  alldf_dict['Base'].append(fivebase)

  alldf_dict['_Experiment'].append(exp)  # TODO check if _Experiment can be removed
  return alldf_dict


def convert_oh_string_to_nparray(input):
  input = input.replace('[', '').replace(']', '')
  nums = input.split(' ')
  return np.array([int(s) for s in nums])


def featurize(rate_stats, Y_nm):
  fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
  threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

  ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
  del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
  print('Entropy Shape: %s, Fivebase Shape: %s, Deletion Score Shape: %s' % (ent.shape, fivebases.shape, del_scores.shape))
  Y = np.array(rate_stats[Y_nm])
  print('Y_nm: %s' % Y_nm)

  Normalizer = [(np.mean(fivebases.T[2]), np.std(fivebases.T[2])),
                (np.mean(fivebases.T[3]), np.std(fivebases.T[3])),
                (np.mean(threebases.T[0]), np.std(threebases.T[0])),
                (np.mean(threebases.T[2]), np.std(threebases.T[2])),
                (np.mean(ent), np.std(ent)),
                (np.mean(del_scores), np.std(del_scores)),
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
  with open(out_dir_model + '%s_rate_model.pkl' % out_letters, 'wb') as f:
    pickle.dump(model, f)

  # Obtain bp stats
  bp_model = dict()
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
  t_melt = pd.melt(bp_stats, id_vars=['Base'], value_vars=ins_bases, var_name='Ins Base', value_name='Fraction')
  for base in list('ACGT'):
    bp_model[base] = dict()
    mean_vals = []
    for ins_base in ins_bases:
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
      mean_vals.append(float(np.mean(t_melt[crit])))
    for bp, freq in zip(list('ACGT'), mean_vals):
      bp_model[base][bp] = freq / sum(mean_vals)

  with open(out_dir_model + '%s_bp_model.pkl' % out_letters, 'wb') as f:
    pickle.dump(bp_model, f)

  with open(out_dir_model + '%s_Normalizer.pkl' % out_letters, 'wb') as f:
    pickle.dump(Normalizer, f)

  return model, bp_model, Normalizer


def knn(merged, total_values):
  rate_stats, bp_stats = load_statistics(merged, total_values)
  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  X, Y, Normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
  return generate_models(X, Y, bp_stats, Normalizer)


# TODO fix / optimize
def parse_header(header):
  w = header.split('_')
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
def get_indel_len_pred(pred_all_df):
  indel_len_pred = dict()

  # 1 bp insertions
  crit = (pred_all_df['Category'] == 'ins')                                 # for all insertions
  indel_len_pred[1] = float(sum(pred_all_df[crit]['Predicted_Frequency']))  # predicted frequency of 1bp ins over all indel products
                                                                            # store for +1 key in dictionary
  # Deletions
  for del_len in range(1, 60):
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Length'] == del_len)    # for each deletion length
    freq = float(sum(pred_all_df[crit]['Predicted_Frequency']))                       #   get pred freq of del with that len over all indel products
    dl_key = -1 * del_len                                                             #   give -dl key in dict
    indel_len_pred[dl_key] = freq                                                     #   store as -dl key in dict

                                                                            # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}

  # Frameshifts, insertion-orientation
  fs = {'+0': 0, '+1': 0, '+2': 0}
  for indel_len in indel_len_pred:              # for each predicted frequency of +1, -1, ..., -60
    fs_key = '+%s' % (indel_len % 3)            #   calculate the resulting frameshift +0, +1 or +2 by remainder division
    fs[fs_key] += indel_len_pred[indel_len]     #   and accumulate the predicted frequency of frame shifts
  return indel_len_pred, fs                     # return dict: {+1 = [..], -1 = [..], ..., -60 = [..]} and fs = {'+0': [..], '+1': [..], '+2': [..]}


# TODO fix / optimize
def predict_all(seq, cutsite, rate_model, bp_model, normalizer):
  # Predict 1 bp insertions and all deletions (MH and MH-less)
  # Most complete "version" of inDelphi
  # Requires rate_model (k-NN) to predict 1 bp insertion rate compared to deletion rate
  # Also requires bp_model to predict 1 bp insertion genotype given -4 nucleotide

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH deletions

  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)
  # for this sequence context and cutsite: return all MHs for each resection length, their gc fractions, deltas and deletion lengths

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T  # input to MH-NN
  del_lens = np.array(del_len).T  # input to MH-less NN

  # Predict
  mh_scores = nn_match_score_function(nn_params, pred_input)  # nn_params are the trained MH-NN params
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unfq = np.exp(
    mh_scores - 0.25 * Js)  # unnormalised MH-NN phi for each MH (each of which corresponds to a unique genotype)

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = nn_match_score_function(nn2_params, np.array(dl))  # trained nn2_params
      mhless_score = np.exp(mhless_score - 0.25 * dl)
      mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
      mhfull_contribution = mhfull_contribution + mask
  mhfull_contribution = mhfull_contribution.reshape(-1, 1)
  unfq = unfq + mhfull_contribution  # unnormalised MH deletion genotype freq distribution

  # Store predictions to combine with mh-less deletion predictions
  pred_del_len = copy.copy(del_len)  # prediction deletion lenghts
  pred_gt_pos = copy.copy(gt_pos)  # prediction deltas             these 2 together correspond to a unique genotype

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH-less deletions
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)  # same results as previously

  unfq = list(unfq)  # unnormalised MH deletion genotype freq distribution

  pred_mhless_d = defaultdict(list)
  # Include MH-less contributions at non-full MH deletion lengths
  nonfull_dls = []
  for dl in range(1, 60):
    if dl not in del_len:  # for a deletion length that a MH-based deletion doesn't correspond to
      nonfull_dls.append(dl)
    elif del_len.count(dl) == 1:  # for a deletion length that occurs once for a MH-based deletion...
      idx = del_len.index(dl)
      if mh_len[idx] != dl:  # and is not a full-MH (MH-length = deletion length)
        nonfull_dls.append(dl)
    else:  # e.g. if delebution length occurs but occurs more than once?
      nonfull_dls.append(dl)

  mh_vector = np.array(mh_len)
  for dl in nonfull_dls:  # for each deletion length 1- 60 unaccounted for by MH-NN predictions
    mhless_score = nn_match_score_function(nn2_params,
                                                 np.array(dl))  # nn2_params are the trained MH-less NN parameters
    mhless_score = np.exp(mhless_score - 0.25 * dl)  # get its the MH-less phi

    unfq.append(mhless_score)  # unnormalised scores for MH-based deletion genotypes
    #     + unnormalised scores for each unacccounted for MH-less based genotype
    pred_gt_pos.append('e')  # gtpos = delta, but delta position = e?
    pred_del_len.append(dl)  # deletion length

  unfq = np.array(unfq)
  total_phi_score = float(sum(unfq))

  nfq = np.divide(unfq, np.sum(unfq))  # normalised scores for MH-based and MH-less based deletion genotypes
  pred_freq = list(nfq.flatten())  # convert into 1D: number of all deletion genotypes x 1 list

  d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
  pred_del_df = pd.DataFrame(d)
  pred_del_df['Category'] = 'del'  # dataframe of all predicted deletion products:
  # 'Length'                predicted deletion length
  # 'Genotype Position'     predicted delta
  # 'Predicted_Frequency'   predicted normalised frequency
  # 'Category'              deletion

  ################################################################
  #####
  ##### Predict Insertions
  #####
  # Predict 1 bp insertions
  del_score = total_phi_score  # <- input to k-nn
  dlpred = []
  for dl in range(1, 28 + 1):  # for each deletion length 1:28
    crit = (pred_del_df['Length'] == dl)  # select the predicted dels with that del length
    dlpred.append(
      sum(pred_del_df[crit]['Predicted_Frequency']))  # store the predicted freq of all dels with that length
  dlpred = np.array(dlpred) / sum(dlpred)  # normalised frequency distribution of deletion lengths
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))  # precision score of ^ <- input to k-nn

  # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
  fiveohmapper = {'A': [0, 0], 'C': [0, 0],  # no difference between A and C
                  'G': [1, 0], 'T': [0, 1]}
  threeohmapper = {'A': [1, 0], 'C': [0, 0],  # no difference between C and T
                   'G': [0, 1], 'T': [0, 0]}
  fivebase = seq[cutsite - 1]  # the -4 base, e.g. T
  threebase = seq[cutsite]  # the -3 base
  onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [norm_entropy] + [del_score]  # all inputs to knn
  for idx in range(len(onebp_features)):  # for each G, T, A, G, norm-entropy, del-scoer
    val = onebp_features[idx]
    onebp_features[idx] = (val - normalizer[idx][0]) / normalizer[idx][1]  # normalise acc. to set normaliser
  onebp_features = np.array(onebp_features).reshape(1, -1)
  rate_1bpins = float(rate_model.predict(onebp_features))  # based on 1bp features of this sequence context, predict
  #   the fraction frequency of 1bp ins over all ins and dels
  # Predict 1 bp genotype frequencies
  pred_1bpins_d = defaultdict(list)  # empty dict
  for ins_base in bp_model[
    fivebase]:  # structure of bp_model in e5 line 107     e.g. -4 base = T, bp_model[T] retuns e5 line 112
    # for each base {A,C,G,T,} when -4 base is T:
    freq = bp_model[fivebase][ins_base]  # e.g. freq = avg. freq of A when -4 base is T
    freq *= rate_1bpins / (
          1 - rate_1bpins)  # e.g. freq of ins_base A =  ratio between fraction frequency of A as 1bp ins when -4 base is T
    #                                          and the fraction frequency of all deletions
    # the division by denominator is required to normalise properly at the last line before return
    pred_1bpins_d['Category'].append('ins')
    pred_1bpins_d['Length'].append(1)
    pred_1bpins_d['Inserted Bases'].append(ins_base)  # store 'A'
    pred_1bpins_d['Predicted_Frequency'].append(freq)  # and freq of 'A' when -4 base is T

  pred_1bpins_df = pd.DataFrame(pred_1bpins_d)  # dict -> df
  pred_all_df = pred_del_df.append(pred_1bpins_df,
                                   ignore_index=True)  # to dataframe of all unique predicted deletion products, append unique insertion products and rename
  pred_all_df['Predicted_Frequency'] /= sum(pred_all_df[
                                              'Predicted_Frequency'])  # normalised frequency of all unique indel products for given sequence and cutsite

  return pred_del_df, pred_all_df, total_phi_score, rate_1bpins  # predicted: df of uniq pred'd del products, df of all uniq pred in+del products, total NN1+2 phi score, fraction freq of 1bp ins over all indels


# TODO fix / optimize
def bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir):
  # Input: A specific sequence
  # Find all Cas9 cutsites, gather metadata, and run inDelphi
  try:
    # header is of FASTA type from NCBI
    ans = parse_header(header)
    # gene id in database / chromosome number / start of seq - 30 / end of sequence + 30
    gene_kgid, chrom, start, end = ans
  except:
    return

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

    # Randomly query subset for broad shallow coverage
    r = np.random.random()
    if r > 0.05:
      continue  # randomly decide if will predict on the found cutsite or not. 5% of time will

    # Shuffle everything but GG
    seq_nogg = list(seq[:34] + seq[36:])
    random.shuffle(seq_nogg)
    shuffled_seq = ''.join(seq_nogg[:34]) + 'GG' + ''.join(seq_nogg[36:])  # a sort of -ve control

    # for one set of sequence context and its shuffled counterpart
    for d, seq_context, shuffled_nm in zip([dd, dd_shuffled],
                                           # initially empty dicts (values as list) for each full exon/intron
                                           [seq, shuffled_seq],
                                           # sub-exon/intron cutsite sequence and shuffled sequence
                                           ['wt', 'shuffled']):
      #
      # Store metadata statistics
      #
      local_cutsite = 30
      grna = seq_context[13:33]
      cutsite_coord = start + idx
      unique_id = '%s_%s_hg38_%s_%s_%s' % (gene_kgid, grna, chrom, cutsite_coord, orientation)

      # the SpCas9 gRNAs targeting exons and introns
      d['Sequence Context'].append(seq_context)
      d['Local Cutsite'].append(local_cutsite)
      d['Chromosome'].append(chrom)
      d['Cutsite Location'].append(cutsite_coord)
      d['Orientation'].append(orientation)
      d['Cas9 gRNA'].append(grna)
      d['Gene kgID'].append(gene_kgid)
      d['Unique ID'].append(unique_id)

      # Make predictions for each SpCas9 gRNA targeting exons and introns
      ans = predict_all(seq_context, local_cutsite,  # seq_context is a tuple/pair? of seq and shuffled_seq
                                 rate_model, bp_model, normalizer)  # trained k-nn, bp summary dict, normalizer
      pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans  #
      # predict all receives seq_context = the gRNA sequence and local_cutsite = the -3 base index
      # pred_del_df = df of predicted unique del products             for sequence context and cutsite
      # pred_all_df = df of all predicted unique in+del products          ^
      # total_phi_score = total NN1+2 phi score                           ^
      # ins_del_ratio = fraction frequency of 1bp ins over all indels     ^

      # pred_all_df ( pred_del_df only has the first 4 columns, and only with info for dels):
      #   'Length'                predicted in/del length
      #   'Genotype Position'     predicted delta (useful only for dels)
      #   'Predicted_Frequency'   predicted normalised in/del frequency
      #   'Category'              deletion/insertion
      #   'Inserted Bases'        predicted inserted base (useful only for ins)

      # Save predictions
      # del_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'dels', shuffled_nm)
      # pred_del_df.to_csv(del_df_out_fn)
      # all_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'all', shuffled_nm)
      # pred_all_df.to_csv(all_df_out_fn)

      ## Translate predictions to indel length frequencies
      indel_len_pred, fs = get_indel_len_pred(pred_all_df)  # normalised frequency distributon on indel lengths
      # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}
      #   and normalised frequency distribution of frameshifts
      #   fs = {'+0': [..], '+1': [..], '+2': [..]}
      # d = zip[dd, dd_shuffled]:
      # 'Sequence Context'
      # 'Local Cutsite'
      # 'Chromosome'
      # 'Cutsite Location'
      # 'Orientation'
      # 'Cas9 gRNA'
      # 'Gene kgID'
      # 'Unique ID'

      #
      # Store prediction statistics
      #
      d['Total Phi Score'].append(total_phi_score)
      d['1ins/del Ratio'].append(ins_del_ratio)

      d['1ins Rate Model'].append(rate_model)
      d['1ins bp Model'].append(bp_model)
      d['1ins normalizer'].append(normalizer)

      d['Frameshift +0'].append(fs['+0'])
      d['Frameshift +1'].append(fs['+1'])
      d['Frameshift +2'].append(fs['+2'])
      d['Frameshift'].append(fs['+1'] + fs['+2'])

      crit = (pred_del_df['Genotype Position'] != 'e')
      s = pred_del_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)
      del_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - Del Genotype'].append(del_gt_precision)

      dls = []
      for del_len in range(1, 60):
        dlkey = -1 * del_len
        dls.append(indel_len_pred[dlkey])
      dls = np.array(dls) / sum(dls)
      del_len_precision = 1 - entropy(dls) / np.log(len(dls))
      d['Precision - Del Length'].append(del_len_precision)

      crit = (pred_all_df['Genotype Position'] != 'e')
      s = pred_all_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)
      all_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - All Genotype'].append(all_gt_precision)

      negthree_nt = seq_context[local_cutsite - 1]
      negfour_nt = seq_context[local_cutsite]
      d['-4 nt'].append(negfour_nt)
      d['-3 nt'].append(negthree_nt)

      crit = (pred_all_df['Category'] == 'ins')
      highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
      highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      d['Highest Ins Rate'].append(highest_ins_rate)
      d['Highest Del Rate'].append(highest_del_rate)

  return


# TODO fix / optimize
def maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = False):
  if split == '0':
    line_threshold = 500
  else:
    line_threshold = 5000
  norm_condition = bool(bool(len(dd['Unique ID']) > line_threshold) and bool(len(dd_shuffled['Unique ID']) > line_threshold))

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


# TODO fix / optimize
def predict_all_items(prediction_file, df_out_dir, nn_params, nn2_params, rate_model, bp_model, normalizer):
  dd = defaultdict(list)
  dd_shuffled = defaultdict(list)

  num_flushed = 0
  timer = util.Timer(total = util.line_count(prediction_file))
  with open(prediction_file) as f:
    for i, line in enumerate(f):
      if i % 2 == 0:
        header = line.strip()
      if i % 2 == 1:
        sequence = line.strip()

        if len(sequence) < 60:
          continue
        if len(sequence) > 500000:
          continue
        # predict for a single exon/intron
        bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir)

        dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)

      if (i - 1) % 50 == 0 and i > 1:
        print('%s pct, %s' % (i / 500, datetime.datetime.now()))

      timer.update()

  maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force=True)


def load_ins_models(out_letters):
  rate_model = load_model(out_dir_model + '%s_rate_model.pkl' % out_letters)
  bp_model = load_model(out_dir_model + '%s_bp_model.pkl' % out_letters)
  normalizer = load_model(out_dir_model + '%s_Normalizer.pkl' % out_letters)
  return rate_model, bp_model, normalizer


def load_neural_networks(out_letters):
  nn_params = load_model(out_dir_params + '%s_nn2.pkl' % out_letters)
  nn2_params = load_model(out_dir_params + '%s_nn2.pkl' % out_letters)
  return nn_params, nn2_params


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execution Details')
  parser.add_argument('--cached_nn', dest='use_prev_nn_model', type=str, help='Boolean variable indicating if to use cached model or recalculate neural network')
  parser.add_argument('--cached_knn', dest='use_prev_knn_model', type=str, help='Boolean variable indicating if to use cached model or recalculate knn')
  parser.add_argument('--pred_file', dest='pred_file', type=str, help='File name used to predict outcomes')

  args = parser.parse_args()

  if args.use_prev_nn_model:
    use_nn_model = args.use_prev_nn_model == 'True'
  else:
    use_nn_model = False

  if args.pred_file:
    prediction_file = args.pred_file
  else:
    prediction_file = ''


  if args.pred_file:
    use_knn_model = args.use_prev_knn_model == 'True'
  else:
    use_knn_model = False


  out_dir, log_fn, out_dir_params, out_dir_stat, out_dir_model, out_dir_exin, out_letters = initialize_files_and_folders(use_nn_model)
  print_and_log("Loading data...", log_fn)
  input_dir = os.path.dirname(os.path.dirname(__file__)) + '/in/'

  counts, del_features = read_data(input_dir + 'dataset.pkl')
  merged = pd.concat([counts, del_features], axis=1)
  merged = merged.reset_index()
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  if not use_nn_model:
    print_and_log("Training Neural Networks...", log_fn)
    nn_params, nn2_params = neural_networks(merged)
  else:
    print_and_log("Loading Neural Networks...", log_fn)
    nn_params, nn2_params = load_neural_networks(out_letters)

  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  if not use_knn_model:
    print_and_log("Training KNN...", log_fn)
    total_values = load_model(out_dir_params + 'total_phi_delfreq.pkl')
    rate_model, bp_model, normalizer = knn(merged, total_values)
  else:
    print_and_log("Loading KNN...", log_fn)
    rate_model, bp_model, normalizer = load_ins_models(out_letters)

  # TODO predict function using models above
  print('Prediction')
  predict_all_items(prediction_file, out_dir_exin, nn_params, nn2_params, rate_model, bp_model, normalizer)
