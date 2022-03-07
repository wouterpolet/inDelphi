import os, datetime
import pickle
import pandas as pd
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.model_selection import train_test_split

import _config, _predict
from mylib import util
from d2_model import alphabetize, count_num_folders, print_and_log, save_train_test_names\
  , init_random_params, main_objective, rsq, save_parameters, adam_minmin

import fi2_ins_ratio
import fk_1bpins


def parse_data(counts, del_features):
  merged = pd.concat([counts, del_features], axis=1)
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

  optimized_params = adam_minmin(both_objective_grad, init_nn_params, init_nn2_params, step_size=step_size, num_iters=num_epochs, callback=print_perf)

def neural_networks():
  seed, nn_layer_sizes, nn2_layer_sizes = initialize_model()

  print_and_log("Loading data...", log_fn)
  counts, del_features = read_data('../in/' + 'dataset.pkl')
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(counts, del_features)

  print_and_log("Parsing data...", log_fn)
  INP = []
  for mhl, gcf in zip(mh_lens, gc_fracs):
    inp_point = np.array([mhl, gcf]).T   # N * 2
    INP.append(inp_point)
  INP = np.array(INP)   # 2000 * N * 2
  # Neural network considers each N * 2 input, transforming it into N * 1 output.
  OBS = np.array(freqs)
  OBS2 = np.array(dl_freqs)
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  print_and_log("Training model...", log_fn)
  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  save_train_test_names(NAMES_train, NAMES_test, out_dir)
  train_parameters(ans, out_dir_params, seed, nn_layer_sizes, nn2_layer_sizes, out_letters)

def knn():
  exps = ['VO-spacers-HEK293-48h-controladj',
          'VO-spacers-K562-48h-controladj',
          'DisLib-mES-controladj',
          'DisLib-U2OS-controladj',
          'Lib1-mES-controladj'
         ]

  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()
  for exp in exps:
    # TODO: Check re statistics - this might be an issue
    rate_stats = fi2_ins_ratio.load_statistics(exp)
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    bp_stats = fk_1bpins.load_statistics(exp)
    exps = rate_stats['_Experiment']

    if 'DisLib' in exp:
      crit = (rate_stats['_Experiment'] >= 73) & (rate_stats['_Experiment'] <= 300)
      rs = rate_stats[crit]
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)

      crit = (rate_stats['_Experiment'] >= 16) & (rate_stats['_Experiment'] <= 72)
      rs = rate_stats[crit]
      rs = rs[rs['Ins1bp Ratio'] < 0.3] # remove outliers
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)

      crit = (bp_stats['_Experiment'] >= 73) & (bp_stats['_Experiment'] <= 300)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)

      crit = (bp_stats['_Experiment'] >= 16) & (bp_stats['_Experiment'] <= 72)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)

    elif 'VO' in exp or 'Lib1' in exp:
      all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
      all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

    print(exp, len(all_rate_stats))

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, all_bp_stats, Normalizer)


def predict_all_items():
  out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'

  # _predict.init_model(run_iter='aax', param_iter='aag')
  # _predict.predict_all()




if __name__ == '__main__':
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  out_dir, log_fn, out_dir_params, out_letters = initialize_files_and_folders()
  neural_networks()

  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  knn()




