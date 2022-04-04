import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_folder)

import argparse
import pandas as pd
import autograd.numpy as np
from collections import Counter

import warnings
from pandas.core.common import SettingWithCopyWarning

from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve

from functionality.author_helper import print_and_log
from functionality.RQs.Jon.helper import load_nn_statistics
# from functionality.neural_networks import mh_del_subset, normalize_count, del_subset
from functionality.prediction import featurize

import functionality.RQs.Jon.plots as plt
import functionality.helper as helper
import functionality.RQs.Jon.nn as nn
import functionality.ins_network as knn

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)


def learning_curves(all_data, total_values):
  rate_stats, bp_stats = helper.load_statistics(all_data, total_values, model_folder + helper.FOLDER_STAT_KEY)
  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  X, y, Normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
  knn = KNeighborsRegressor()
  train_sizes, train_scores, test_scores = learning_curve(knn, X, y, cv=100,
                                                          scoring='neg_mean_squared_error',
                                                          n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))

  knn.fit(X, y)
  # y_pred = knn.predict(X)
  # print('R-Squared: ' + str(r2_score(y, y_pred)))
  # print('KNN Score: ' + str())
  train_mean = -1 * np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  test_mean = -1 * np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  plt.plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, knn.score(X, y))
  # train_scores_mean = -train_scores.mean(axis=1)
  # validation_scores_mean = -test_scores.mean(axis=1)
  # plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean)


def get_predictions_and_observations_adaptation(all_data):
  data = mh_del_subset(all_data)[1]
  data = normalize_count(data)

  libA = helper.load_lib_data(helper.INPUT_DIRECTORY + 'libX/', 'libA')
  targets = helper.get_targets(libA, data)

  results = {}
  for grna in targets.keys():
    current_result = {}
    seq = targets[grna]
    cutsite = 30
    pred_del_df, pred_df, total_phi_score, rate_1bpins = predict_all(seq, cutsite, nn_params, nn2_params, rate_model, bp_model, normalizer)

    join_cols = ['Category', 'Genotype Position', 'Length']
    # Fails here - columns do not exist in original df - not sure what they should be mapped to
    mdf = data.merge(pred_df, how='outer', on=join_cols)
    mdf['Frequency'].fillna(value=0, inplace=True)
    mdf['Predicted_Frequency'].fillna(value=0, inplace=True)
    obs = mdf['Frequency']
    pred = mdf['Predicted_Frequency']
    current_result['gt_r'] = pearsonr(obs, pred)[0]

    df = del_subset(all_data)
    df = df[df['Size'] <= 28]
    df = normalize_count(df)
    obs_dl = []
    for del_len in range(1, 28 + 1):
      freq = sum(df[df['Size'] == del_len]['countEvents'])
      obs_dl.append(freq)
    pred_dl = deletion_length_distribution(seq, cutsite) # = get_indel_len_pred
    current_result['dl_r'] = pearsonr(obs_dl, pred_dl)[0]

    results['_Experiment'] = current_result
  return


def deletion_length_distribution(seq, cutsite):
  raise NotImplementedError()


def get_pred_obs(prediction, observation):
  results = {}
  preds = []
  obs = []
  for i in range(1, -31, -1):
    if i == 0:
      continue
    current_pred = 0
    current_obs = 0
    for pred in prediction:
      current_pred += pred[1][i]
    preds.append(current_pred/len(prediction))
    for idx, key in enumerate(observation.keys()):
      current_obs += observation[key][i]
    obs.append(current_obs/len(observation.keys()))

    results[i] = {'prediction': current_pred/len(prediction),
                  'observed': current_obs/len(observation.keys())}
  results = pd.DataFrame(results).T
  return results


def get_indel_len_pred(pred_all_df, del_len_limit):
  indel_len_pred = {}

  # 1 bp insertions
  crit = (pred_all_df['Category'] == 'ins')
  indel_len_pred[1] = float(sum(pred_all_df[crit]['Predicted_Frequency']))

  for del_len in range(1, del_len_limit):
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Length'] == del_len)
    freq = float(sum(pred_all_df[crit]['Predicted_Frequency']))
    dl_key = -1 * del_len
    indel_len_pred[dl_key] = freq
  return indel_len_pred


def get_predictions_and_observations(data, nn_params, nn2_params, rate_model, bp_model, normalizer):
  libA = helper.load_lib_data(helper.INPUT_DIRECTORY + 'libX/', 'libA')
  targets = helper.get_targets(libA, data, with_grna=True)
  # for grna in targets.keys():
  unique_samples = data['Sample_Name'].unique()
  indel_len_obs = {}
  indel_len_prd = {}
  ins_only = {}
  del_limit = 61
  for sample_name in unique_samples:
    # Calculate observations
    res = {1: sum(data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'INSERTION') &
                       (data['Indel'].str.startswith('1+'))]['countEvents'])}
    total = res[1]

    sample_del = data[(data['Sample_Name'] == sample_name) & (data['Type'] == 'DELETION')]
    for i in range(1, del_limit):
      res[-i] = sum(sample_del[sample_del['Size'] == i]['countEvents'].tolist())
      total += res[-i]

    # Normalize
    for length, count in res.items():
      res[length] = count / total
    indel_len_obs[sample_name] = res

    # Calculate predictions
    grna = sample_name.split('_')
    grna = grna[len(grna) - 1]
    seqs = list(set(targets[grna]))
    cutsite = 30
    current_indel_pred = {}
    for id, seq in enumerate(seqs):
      ans = predict_all(seq, cutsite, nn_params, nn2_params, rate_model, bp_model, normalizer)  # trained k-nn, bp summary dict, normalizer
      pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans
      indel_pred = get_indel_len_pred(pred_all_df, del_limit)

      if id != 0:
        current_indel_pred = dict(Counter(current_indel_pred) + Counter(indel_pred))
      else:
        current_indel_pred = indel_pred

    # Normalize
    total_seq = len(seqs)
    if total_seq == 1:
      indel_pred = current_indel_pred
    else:
      # Normalize
      indel_pred = {k: v / total_seq for k, v in current_indel_pred.items()}
    ins_only[sample_name] = {'prediction': indel_pred[1], 'observation': res[1]}
    indel_len_prd[sample_name] = indel_pred

  return indel_len_prd, indel_len_obs, ins_only


def get_pearson_pred_obs(prediction, observation):
  r_values = []
  t_values = []
  n = len(prediction)
  pred_normalized_fq = []
  for pred in prediction.keys():                                   # for each held-out gRNA
    current_pred_normalized_fq = []
    for i in range(1, -61, -1):                             #   for indel length +1, -1, -2, ...,-30 (keys)
      if i != 0:
        current_pred_normalized_fq.append(prediction[pred][i])       #       get freq for key i
    pred_normalized_fq.append(current_pred_normalized_fq)   #   return array of predicted frequencies

  for idx, key in enumerate(observation.keys()):
    # Get prediction of GRNA
    # normalized_fq = prediction[prediction['Sample_Name']]
    normalized_fq = []
    for i in range(1, -61, -1):
      if i != 0:
        normalized_fq.append(observation[key][i])

    # For dictionary, get items from 1 to -30 into an array

    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(pred_normalized_fq[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean) * (pred_normalized_fq[idx] - y_mean))
    pearson_denom = np.sqrt(np.sum((normalized_fq - x_mean) ** 2) * np.sum((pred_normalized_fq[idx] - y_mean) ** 2))
    r_value = pearson_numerator / pearson_denom
    r_values.append(r_value)

    t_value = (r_value/(np.sqrt(1-(r_value ** 2)))) * np.sqrt(n-2)
    t_values.append(t_value)
  return r_values, t_values


def load_and_plot_model_loss(model_folder):
  loss_values = load_nn_statistics(model_folder)
  plt.plot_nn_loss_epoch(loss_values)
  return loss_values['seed'][0]


def model_creation(data, model_type):
  '''
  Neural Network (MH)
  Model Creation, Training & Optimization
  '''
  out_folder = out_dir + model_type
  print_and_log("Training Neural Networks...", log_fn)
  nn_params, nn2_params = nn.create_neural_networks(data, log_fn, out_folder, exec_id, seed_value)
  '''
  KNN - 1 bp insertions
  Model Creation, Training & Optimization
  '''
  print_and_log("Training KNN...", log_fn)
  total_values = helper.load_total_phi_delfreq(out_folder)
  insertion_model = knn.InsertionModel(out_folder, out_folder + helper.FOLDER_STAT_KEY)
  rate_model, bp_model, normalizer = insertion_model.train_knn(data, total_values)
  model = {'nn': nn_params, 'nn_2': nn2_params, 'rate': rate_model, 'bp': bp_model, 'norm': normalizer}
  return model



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execution Details')

  parser.add_argument('--model_folder', dest='model_folder', type=str, help='Variable indicating the execution id of the trained neural network and knn')
  args = parser.parse_args()
  if args.model_folder:
    user_exec_id = args.model_folder
  else:
    raise Exception("Please specify --model_folder")

  out_dir, log_fn, exec_id = helper.initialize_files_and_folders(user_exec_id)
  if user_exec_id != exec_id:
    raise Exception("Please specify a valid pre-trained model")

  print_and_log("Loading pre-trained networks...", log_fn)
  model_folder = out_dir + 'fig_3/'
  nn_path = model_folder + helper.FOLDER_PARAM_KEY
  models = helper.load_models(model_folder)
  nn_params = models['nn']
  nn2_params = models['nn_2']
  rate_model = models['rate']
  bp_model = models['bp']
  normalizer = models['norm']

  # Loading and plotting the current model loss values
  print_and_log("Learning Curve for Current Neural Networks...", log_fn)
  seed_value = load_and_plot_model_loss(model_folder)

  # Training a new model with alterations to the NN
  print_and_log("Learning new Neural Network - Split...", log_fn)

  all_data_mesc = pd.concat(helper.read_data(helper.INPUT_DIRECTORY + 'dataset.pkl'), axis=1).reset_index()
  split_nns, max_nns = model_creation(all_data_mesc, 'fig_3_opt/')
  # Loading and plotting the current model loss values
  print_and_log("Learning Curve for Current Neural Networks...", log_fn)
  model_folder = out_dir + 'fig_3opt/max'
  load_and_plot_model_loss(model_folder)
  model_folder = out_dir + 'fig_3opt/max'
  load_and_plot_model_loss(model_folder)



















  # mesc_file = ''
  # libA = helper.load_lib_data(helper.INPUT_DIRECTORY + 'libX/', 'libA')
  # train_mesc_file = f'{out_dir + helper.FOLDER_PRED_KEY + helper.FOLDER_PARAM_KEY}train_mesc.pkl'
  # test_mesc_file = f'{out_dir + helper.FOLDER_PRED_KEY + helper.FOLDER_PARAM_KEY}test_mesc.pkl'
  # prediction_files = os.listdir(out_dir + helper.FOLDER_PRED_KEY)
  #
  # for prediction_file in prediction_files:
  #   if "mesc" in prediction_file:
  #     mesc_file = prediction_file
  #     break
  #
  # if mesc_file != '':
  #   predictions = helper.load_predictions(out_dir + helper.FOLDER_PRED_KEY + mesc_file)
  #   test_mesc = helper.load_pickle(test_mesc_file)
  #   # Get actual observations
  #   observations = get_observed_values(test_mesc)
  # else:
  #   raise Exception("Please retrain the model")
  #
  # pred, obs, ins_only = get_predictions_and_observations(test_mesc, nn_params, nn2_params, rate_model, bp_model, normalizer)
  # insertion_pred_obs = pd.DataFrame(ins_only).T
  #
  # pearson_co, t_values = get_pearson_pred_obs(pred, obs)
  # plt.plot_prediction_observation(insertion_pred_obs)
  # plt.plot_student_t_distribution(t_values)
  #
  # # [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(test_mesc)
  # # INP, OBS, OBS2, NAMES, DEL_LENS = format_data(exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs)
  # # Predict unseen values
  # # rsq1, rsq2 = helper.rsq(nn_params, nn2_params, INP, OBS, OBS2, DEL_LENS, NAMES)
  # #
  # # plot_rsqs(rsq1, rsq2)
  #
  # helper.print_and_log("Original Learning Curve for Insertion Model...", log_fn)
  # total_values = helper.load_pickle(model_folder + helper.FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  # all_data_mesc = pd.concat(helper.read_data(helper.INPUT_DIRECTORY + 'dataset.pkl'), axis=1).reset_index()
  # learning_curves(all_data_mesc, total_values)


