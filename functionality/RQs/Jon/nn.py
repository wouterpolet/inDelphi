import datetime

import pandas as pd
from autograd import grad
from autograd.misc import flatten
import autograd.numpy as np
from scipy.stats import entropy, pearsonr
from sklearn.model_selection import train_test_split

from functionality.neural_networks import parse_data
from functionality.author_helper import nn_match_score_function, init_random_params, rsq, print_and_log, alphabetize, save_parameters, exponential_decay, save_train_test_names, ensure_dir_exists, save_parameter
import functionality.RQs.Jon.helper as jrq

"""
Python file was not converted to class because of callback function and nested functions
"""


def initialize_model():
  """
  Neural Network settings creation
  @return: layering for nn1, layering for nn2
  """
  # Model Settings & params
  # seed = npr.RandomState(1) - Removed since using the same seed as the original model
  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]
  return nn_layer_sizes, nn2_layer_sizes


def main_objective(nn_params, nn2_params, inp, obs, obs2, del_lens, store=False):
  LOSS = 0
  nn_loss = 0
  nn2_loss = 0
  total_phi_del_freq = []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)   # vector of 0s, one for each MH
    for jdx in range(len(mh_vector)):
      # if the deletion length for the indexed MH of that gRNA = MH length (i.e. full MH)
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))  # predict MH-less psi
        mhless_score = np.exp(mhless_score - 0.25 * dl)  # conver to MH-less phi
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
    nn1_neg_rsq = rsq * -1
    nn_loss += nn1_neg_rsq

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
    nn2_neg_rsq = rsq * -1
    nn2_loss += nn2_neg_rsq
    if use_max:
      LOSS += max(nn1_neg_rsq, nn2_neg_rsq)
    else:
      LOSS += (nn1_neg_rsq + nn2_neg_rsq)

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

  if store:
    column_names = ["exp", "total_phi", "norm_entropy"]
    df = pd.DataFrame(total_phi_del_freq, columns=column_names)
    df.to_pickle(out_dir_params + 'total_phi_delfreq.pkl')

  current_statistics['nn_loss'] = nn_loss
  current_statistics['nn2_loss'] = nn2_loss
  return LOSS


def train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, exec_id):
  param_scale = 0.1
  global num_epochs
  num_epochs = 51

  step_size = 0.10
  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs=seed)
  init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans

  def objective(nn_params, nn2_params):
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train)

  both_objective_grad = grad(objective, argnum=[0, 1])

  def print_perf(nn_params, nn2_params, iter):
    train_size = len(INP_train)
    test_size = len(INP_test)

    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train) / train_size
    # Jon - Research Question - Start
    loss_statistics = {
      'nn_train_loss': current_statistics['nn_loss']/train_size,
      'nn2_train_loss': current_statistics['nn2_loss']/train_size
    }
    # Jon - Research Question - End
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test) / test_size
    # Jon - Research Question - Start
    loss_statistics['nn_test_loss'] = current_statistics['nn_loss']/test_size
    loss_statistics['nn2_test_loss'] = current_statistics['nn2_loss']/test_size
    # Jon - Research Question - End

    tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train)
    te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test)

    tr1_rsq_mean = np.mean(tr1_rsq)
    tr2_rsq_mean = np.mean(tr2_rsq)
    te1_rsq_mean = np.mean(te1_rsq)
    te2_rsq_mean = np.mean(te2_rsq)

    out_line = ' %s\t\t| %.3f\t\t| %.3f\t\t\t| %.3f\t\t\t| %.3f\t| %.3f\t\t| %.3f\t\t|' % (iter, train_loss, tr1_rsq_mean, tr2_rsq_mean, test_loss, te1_rsq_mean, te2_rsq_mean)
    print_and_log(out_line, log_fn)

    # Jon - Research Question - Start
    statistics = {'iteration': iter, 'seed': seed,
                  'train_loss': train_loss, 'train_rsq1': tr1_rsq_mean, 'train_rsq2': tr2_rsq_mean, 'train_sample_size':train_size,
                  'test_loss': test_loss, 'test_rsq1': te1_rsq_mean, 'test_rsq2': te2_rsq_mean, 'test_sample_size':test_size}
    statistics.update(loss_statistics)
    execution_statistics.append(statistics)
    # Jon - Research Question - End

    if iter % 10 == 0:
      letters = alphabetize(int(iter / 10))
      # helper.print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
      print_and_log('...Saving parameters... | Timestamp: %s | Execution ID: %s | Parameter ID: %s' % (datetime.datetime.now(), exec_id, letters), log_fn)
      save_parameters(nn_params, nn2_params, out_dir_params, letters)
      if iter == num_epochs - 1:
        pass

    return None

  optimized_params = adam_minmin(both_objective_grad, init_nn_params, init_nn2_params, step_size=step_size,
                                 num_iters=num_epochs, callback=print_perf)
  return optimized_params


def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
  x_nn, unflatten_nn = flatten(init_params_nn)
  x_nn2, unflatten_nn2 = flatten(init_params_nn2)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
  for i in range(num_iters):
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2))
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


def format_data(exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs):
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
  return INP, OBS, OBS2, NAMES, DEL_LENS


def single_objective_nn1(nn_params, inp, obs, del_lens):
  LOSS = 0
  nn_loss = 0
  nn2_loss = 0
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
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
    nn_loss += neg_rsq

  current_statistics['nn_loss'] = nn_loss
  current_statistics['nn2_loss'] = nn2_loss
  return LOSS


def single_objective_nn2(nn2_params, inp, obs2, del_lens, store=False):
  LOSS = 0
  nn_loss = 0
  nn2_loss = 0
  total_phi_del_freq = []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(trained_nn1, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)   # vector of 0s, one for each MH
    for jdx in range(len(mh_vector)):
      # if the deletion length for the indexed MH of that gRNA = MH length (i.e. full MH)
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))  # predict MH-less psi
        mhless_score = np.exp(mhless_score - 0.25 * dl)  # conver to MH-less phi
        mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution

    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)

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
    nn2_neg_rsq = rsq * -1
    LOSS += nn2_neg_rsq
    nn2_loss += nn2_neg_rsq

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

  if store:
    column_names = ["exp", "total_phi", "norm_entropy"]
    df = pd.DataFrame(total_phi_del_freq, columns=column_names)
    df.to_pickle(out_dir_params + 'total_phi_delfreq.pkl')

  current_statistics['nn_loss'] = nn_loss
  current_statistics['nn2_loss'] = nn2_loss
  return LOSS


def adam_min(grad_single, init_params_nn, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
  x_nn, unflatten_nn = flatten(init_params_nn)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  for i in range(num_iters):
    g_nn_uf = grad_single(unflatten_nn(x_nn))
    g_nn, _ = flatten(g_nn_uf)

    if callback:
      callback(unflatten_nn(x_nn), i)

    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn ** 2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1 ** (i + 1))  # Bias correction.
    vhat_nn = v_nn / (1 - b2 ** (i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)
  return unflatten_nn(x_nn)


def singular_rsq_nn1(nn_params, inp, obs, del_lens, names=None):
  if names is not None:
    rsqs = {}
  else:
    rsqs = []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    rsq1 = pearsonr(normalized_fq, obs[idx])[0] ** 2
    if names is not None:
      rsqs[names[idx]] = rsq1
    else:
      rsqs.append(rsq1)
  return rsqs


def singular_rsq_nn2(nn2_params, inp, obs2, del_lens, names=None):
  if names is not None:
    rsqs = {}
  else:
    rsqs = []
  for idx in range(len(inp)):
    mh_scores = nn_match_score_function(trained_nn1, inp[idx])
    Js = np.array(del_lens[idx])
    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28 + 1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25 * np.arange(1, 28 + 1))

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

    rsq = pearsonr(normalized_fq, obs2[idx])[0] ** 2
    if names is not None:
      rsqs[names[idx]] = rsq
    else:
      rsqs.append(rsq)
  return rsqs


def train_parameter(ans, seed, nn_layer_sizes, exec_id, is_nn1=True):
  param_scale = 0.1
  global num_epochs
  num_epochs = 51

  step_size = 0.10
  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs=seed)
  INP_train, INP_test, OBS_train, OBS_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans

  def objective(nn_params):
    if is_nn1:
      return single_objective_nn1(nn_params, INP_train, OBS_train, DEL_LENS_train)
    else:
      return single_objective_nn2(nn_params, INP_train, OBS_train, DEL_LENS_train)

  def print_perf(nn_params, iter):
    train_size = len(INP_train)
    test_size = len(INP_test)

    if is_nn1:
      train_loss = single_objective_nn1(nn_params, INP_train, OBS_train, DEL_LENS_train) / train_size
    else:
      train_loss = single_objective_nn2(nn_params, INP_train, OBS_train, DEL_LENS_train) / train_size
    # Jon - Research Question - Start
    loss_statistics = {
      'nn_train_loss': current_statistics['nn_loss']/train_size,
      'nn2_train_loss': current_statistics['nn2_loss']/train_size
    }
    # Jon - Research Question - End
    if is_nn1:
      test_loss = single_objective_nn1(nn_params, INP_test, OBS_test, DEL_LENS_test) / test_size
    else:
      test_loss = single_objective_nn2(nn_params, INP_test, OBS_test, DEL_LENS_test) / test_size
    # Jon - Research Question - Start
    loss_statistics['nn_test_loss'] = current_statistics['nn_loss']/test_size
    loss_statistics['nn2_test_loss'] = current_statistics['nn2_loss']/test_size
    # Jon - Research Question - End
    if is_nn1:
      tr_rsq = singular_rsq_nn1(nn_params, INP_train, OBS_train, DEL_LENS_train)
      te_rsq = singular_rsq_nn1(nn_params, INP_test, OBS_test, DEL_LENS_test)
    else:
      tr_rsq = singular_rsq_nn2(nn_params, INP_train, OBS_train, DEL_LENS_train)
      te_rsq = singular_rsq_nn2(nn_params, INP_test, OBS_test, DEL_LENS_test)

    tr_rsq_mean = np.mean(tr_rsq)
    te_rsq_mean = np.mean(te_rsq)

    out_line = ' %s\t\t| %.3f\t\t| %.3f\t\t\t| %.3f\t\t\t| %.3f\t|' % (iter, train_loss, tr_rsq_mean, test_loss, te_rsq_mean)
    print_and_log(out_line, log_fn)

    # Jon - Research Question - Start
    statistics = {'iteration': iter, 'seed': seed,
                  'train_loss': train_loss, 'train_rsq': tr_rsq_mean, 'train_sample_size':train_size,
                  'test_loss': test_loss, 'test_rsq': te_rsq_mean, 'test_sample_size':test_size}
    statistics.update(loss_statistics)
    execution_statistics.append(statistics)
    # Jon - Research Question - End

    if iter % 10 == 0:
      letters = alphabetize(int(iter / 10))
      # helper.print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
      print_and_log('...Saving parameters... | Timestamp: %s | Execution ID: %s | Parameter ID: %s' % (datetime.datetime.now(), exec_id, letters), log_fn)
      if is_nn1:
        save_parameter(nn_params, out_dir_params, letters, 'nn')
      else:
        save_parameter(nn_params, out_dir_params, letters, 'nn2')
      if iter == num_epochs - 1:
        pass

    return None

  optimized_params = adam_min(grad(objective, argnum=[0]), init_nn_params, step_size=step_size, num_iters=num_epochs, callback=print_perf)
  return optimized_params


def create_neural_networks(merged, log, out_directory, exec_id, seed):
  """
  Create and Train the Nueral Networks (Microhomology and microhomology less networks)
  @param merged: all the data (del_features and counts) provided in the file
  @param log: log file
  @param out_directory: Output directory
  @param exec_id: Execution ID
  @return: the trained neural networks (2)
  """
  global log_fn
  log_fn = log
  global out_dir
  out_dir = out_directory
  global out_dir_params
  out_dir_params = out_dir + 'parameters/'
  ensure_dir_exists(out_dir_params)

  # Jon - Research Question - Start
  global current_statistics
  current_statistics = {}
  global execution_statistics
  execution_statistics = []
  # Jon - Research Question - End

  nn_layer_sizes, nn2_layer_sizes = initialize_model()
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(merged)

  print_and_log("Parsing data...", log_fn)
  global NAMES
  INP, OBS, OBS2, NAMES, DEL_LENS = format_data(exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs)

  print_and_log("Training model...", log_fn)
  print_and_log("Splitting data into train and test...", log_fn)
  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  save_train_test_names(NAMES_train, NAMES_test, out_dir)

  print_and_log(" Iter\t| Train Loss\t| Train Rsq\t| Test Loss\t| Test Rsq\t|", log_fn)
  nn1_data = INP_train, INP_test, OBS_train, OBS_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test
  global trained_nn1
  global use_max
  use_max = False
  old_params = out_dir_params
  out_dir_params = out_dir + 'split/'
  ensure_dir_exists(out_dir_params)
  trained_nn1 = train_parameter(nn1_data, seed, nn_layer_sizes, exec_id, is_nn1=True)
  jrq.save_statistics(out_dir_params+'loss_nn1/', pd.DataFrame(execution_statistics))

  execution_statistics = []
  nn2_data = INP_train, INP_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test
  trained_nn2 = train_parameter(nn2_data, seed, nn2_layer_sizes, exec_id, is_nn1=False)
  jrq.save_statistics(out_dir_params+'loss_nn2/', pd.DataFrame(execution_statistics))
  main_objective(trained_nn1, trained_nn2, INP, OBS, OBS2, DEL_LENS, store=True)

  # Training parameters using the max instead of sum
  use_max = True
  print_and_log("Training model - max...", log_fn)
  print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
  execution_statistics = []
  out_dir_params = out_dir + 'max/'
  ensure_dir_exists(out_dir_params)
  trained_params_max = train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, exec_id)
  jrq.save_statistics(out_dir_params, pd.DataFrame(execution_statistics))
  main_objective(trained_params_max[0], trained_params_max[1], INP, OBS, OBS2, DEL_LENS, store=True)
  out_dir_params = old_params

  # Training parameters using the fixed sum
  print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
  use_max = False
  print_and_log("Training model - fix...", log_fn)
  execution_statistics = []
  out_dir_params = out_dir + 'fix/'
  ensure_dir_exists(out_dir_params)
  trained_params_fix = train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, exec_id)
  jrq.save_statistics(out_dir_params, pd.DataFrame(execution_statistics))
  main_objective(trained_params_fix[0], trained_params_fix[1], INP, OBS, OBS2, DEL_LENS, store=True)
  # out_dir_params = old_params

  return [trained_nn1, trained_nn2], trained_params_max, trained_params_fix

