import helper, datetime
import pandas as pd
import util

from autograd import grad
from autograd.misc import flatten
import autograd.numpy as np
import autograd.numpy.random as npr

from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import helper_jon as jrq


def initialize_model():
  """
  Neural Network settings creation
  @return: seed, layering for nn1, layering for nn2
  """
  # Model Settings & params
  seed = npr.RandomState(1)

  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]
  return seed, nn_layer_sizes, nn2_layer_sizes


def parse_data(all_data):
  """
  Transform the data provided to us (U2OS.pkl or dataset.pkl)
  into their expected format keeping the conditions they specified in their code & literature
  @param all_data: all the data (del_features and counts) provided in the file
  @return:
    exps     : Sample_Name
    mh_lens  : homologyLength as int32
    gc_fracs : homologyGCContent
    del_lens : Size as int 32
    freqs    : countEvents
    dl_freqs : DL frequencies for all del len (1:28)
  """
  deletions = all_data[all_data['Type'] == 'DELETION']
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
    mh_exp_data = microhomologies[microhomologies['Sample_Name'] == exp][['countEvents', 'homologyLength', 'homologyGCContent', 'Size']]

    # Normalize Counts
    total_count = sum(mh_exp_data['countEvents'])
    if total_count == 0:
      continue
    mh_exp_data['countEvents'] = mh_exp_data['countEvents'].div(total_count)

    freqs.append(mh_exp_data['countEvents'])
    mh_lens.append(mh_exp_data['homologyLength'].astype('int32'))
    gc_fracs.append(mh_exp_data['homologyGCContent'])
    del_lens.append(mh_exp_data['Size'].astype('int32'))

    curr_dl_freqs = []
    all_dels = deletions[deletions['Sample_Name'] == exp][['countEvents', 'Size']]
    total_count = sum(all_dels['countEvents'])
    all_dels['countEvents'] = all_dels['countEvents'].div(total_count)
    dl_freq_df = all_dels[all_dels['Size'] <= 28]
    # dl_freq_df = mh_exp_data[mh_exp_data['Size'] <= 28]
    for del_len in range(1, 28 + 1):
      dl_freq = sum(dl_freq_df[dl_freq_df['Size'] == del_len]['countEvents'])
      curr_dl_freqs.append(dl_freq)
    dl_freqs.append(curr_dl_freqs)

    # # Microhomology-less computation
    # mh_less_exp_data = mh_less[mh_less['Sample_Name'] == exp][['countEvents', 'Size']]
    # del_lens.append(mh_exp_data['Size'].astype('int32'))

  return exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs


def main_objective(nn_params, nn2_params, inp, obs, obs2, del_lens, rs, store=False):
  LOSS = 0
  nn_loss = 0
  nn2_loss = 0
  total_phi_del_freq = []  # 1961 x 1
  for idx in range(len(inp)):  # for each gRNA's zipped [MH lengths, MH GC fracs] in the training set
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = helper.nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]                         # vector of all MH lengths for each MH
    mhfull_contribution = np.zeros(mh_vector.shape)   # vector of 0s, one for each MH
    for jdx in range(len(mh_vector)):                 # for each MH
      if del_lens[idx][jdx] == mh_vector[jdx]:        #     if the deletion length for the indexed MH of that gRNA = MH length (i.e. full MH)
        dl = del_lens[idx][jdx]                       #           store deletion length
        mhless_score = helper.nn_match_score_function(nn2_params, np.array(dl))    # predict MH-less psi
        mhless_score = np.exp(mhless_score - 0.25 * dl)                     # conver to MH-less phi
        mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
        #                      element 1        element 2                     element 3
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
    nn_loss += neg_rsq

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
    nn2_scores = helper.nn_match_score_function(nn2_params, dls)
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
    nn2_loss += neg_rsq

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

  # if iter == num_epochs - 1:
  if store:
    column_names = ["exp", "total_phi", "norm_entropy"]
    df = pd.DataFrame(total_phi_del_freq, columns=column_names)
    df.to_pickle(out_dir_params + 'total_phi_delfreq.pkl')

  # TODO - Why are we returning a single loss value, if we're comuting the loss for both NNs?
  #  Shouldn't this return 2 items - Loss of NN1 and Loss of NN2?
  current_statistics['nn_loss'] = nn_loss
  current_statistics['nn2_loss'] = nn2_loss
  return LOSS # / num_samples #, nn_loss/len(inp), nn2_loss/len(inp)


def train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, exec_id):
  param_scale = 0.1
  # num_epochs = 7*200 + 1
  global num_epochs
  num_epochs = 51

  step_size = 0.10
  init_nn_params = helper.init_random_params(param_scale, nn_layer_sizes, rs=seed)
  init_nn2_params = helper.init_random_params(param_scale, nn2_layer_sizes, rs=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans

  # batch_size = 200
  # num_batches = int(np.ceil(len(INP_train) / batch_size))

  # def batch_indices(iter):
  #   idx = iter % num_batches
  #   return slice(idx * batch_size, (idx + 1) * batch_size)

  def objective(nn_params, nn2_params, iter):
    # idx = batch_indices(iter)
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, seed)

  both_objective_grad = grad(objective, argnum=[0, 1])

  def print_perf(nn_params, nn2_params, iter):
    # if iter % 5 != 0:
    #   return None

    # TODO - Check with team, why for one we use batch size - set to 200 (constant) and the other we use input length
    #  Why not use the len(INP_Train)?
    # train_size = batch_size
    train_size = len(INP_train)
    test_size = len(INP_test)

    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, seed) / train_size
    # Jon - Research Question - Start
    loss_statistics = {
      'nn_train_loss': current_statistics['nn_loss']/train_size,
      'nn2_train_loss': current_statistics['nn2_loss']/train_size
    }
    # Jon - Research Question - End
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, seed) / test_size
    # Jon - Research Question - Start
    loss_statistics['nn_test_loss'] = current_statistics['nn_loss']/test_size
    loss_statistics['nn2_test_loss'] = current_statistics['nn2_loss']/test_size
    # Jon - Research Question - End

    tr1_rsq, tr2_rsq = helper.rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train)
    te1_rsq, te2_rsq = helper.rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test)

    tr1_rsq_mean = np.mean(tr1_rsq)
    tr2_rsq_mean = np.mean(tr2_rsq)
    te1_rsq_mean = np.mean(te1_rsq)
    te2_rsq_mean = np.mean(te2_rsq)

    out_line = ' %s\t\t| %.3f\t\t| %.3f\t\t\t| %.3f\t\t\t| %.3f\t| %.3f\t\t| %.3f\t\t|' % (iter, train_loss, tr1_rsq_mean, tr2_rsq_mean, test_loss, te1_rsq_mean, te2_rsq_mean)
    helper.print_and_log(out_line, log_fn)

    # Jon - Research Question - Start
    statistics = {'iteration': iter, 'train_loss': train_loss, 'train_rsq1': tr1_rsq_mean,
                  'train_rsq2': tr2_rsq_mean, 'train_sample_size': train_size, 'test_loss': test_loss,
                  'test_rsq1': te1_rsq_mean, 'test_rsq2': te2_rsq_mean, 'test_sample_size': test_size}
    statistics.update(loss_statistics)
    execution_statistics.append(statistics)
    # Jon - Research Question - End

    if iter % 10 == 0:
      letters = helper.alphabetize(int(iter / 10))
      # helper.print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
      helper.print_and_log('...Saving parameters... | Timestamp: %s | Execution ID: %s | Parameter ID: %s' % (datetime.datetime.now(), exec_id, letters), log_fn)
      helper.save_parameters(nn_params, nn2_params, out_dir_params, letters)
      if iter == num_epochs - 1:
        pass
        # print('check rsq now')
        # jrq.plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train)
        # jrq.plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test)
        # jrq.plot_mh_score_function(nn_params)
        # jrq.plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train)
        # jrq.plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test)

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
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    g_nn, _ = flatten(g_nn_uf)
    g_nn2, _ = flatten(g_nn2_uf)

    if callback:
      callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)

    step_size = helper.exponential_decay(step_size)

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


def create_neural_networks(merged, log, out_directory, exec_id):
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
  util.ensure_dir_exists(out_dir_params)

  # Jon - Research Question - Start
  global current_statistics
  current_statistics = {}
  global execution_statistics
  execution_statistics = []
  # Jon - Research Question - End

  seed, nn_layer_sizes, nn2_layer_sizes = initialize_model()
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(merged)

  helper.print_and_log("Parsing data...", log_fn)
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

  helper.print_and_log("Training model...", log_fn)
  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  helper.save_train_test_names(NAMES_train, NAMES_test, out_dir)
  helper.print_and_log(" Iter\t| Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2\t|", log_fn)
  trained_params = train_parameters(ans, seed, nn_layer_sizes, nn2_layer_sizes, exec_id)
  # get total phi scores and del frequencies for all of the data (not 85/15 split)
  main_objective(trained_params[0], trained_params[1], INP, OBS, OBS2, DEL_LENS, seed, store=True)

  # Jon - Research Question - Start
  jrq.save_statistics(out_dir, pd.DataFrame(execution_statistics))
  # Jon - Research Question - End
  return trained_params

