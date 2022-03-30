import os
import argparse
import pandas as pd
import autograd.numpy as np
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve

import helper
from helper_jon import load_nn_statistics
from ins_network import load_statistics, featurize
from functionality.neural_networks import mh_del_subset, normalize_count, del_subset
from all_func import initialize_files_and_folders, load_models, load_predictions, get_observed_values, load_lib_data, get_targets
from prediction import predict_all

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_PRED_KEY = 'predictions/'
FOLDER_INPUT_KEY = '/in/'
EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))


def plot_nn_loss_epoch(loss_values):
  # Plot Global Loss
  plt.plot(loss_values['iteration'], loss_values['train_loss'], label="train NNs", color='#ff0000')
  plt.plot(loss_values['iteration'], loss_values['test_loss'], label="test NNs", color='#0000ff')
  ylim_min = min(min(loss_values['train_loss']), min(loss_values['test_loss']), min(loss_values['nn_train_loss']), min(loss_values['nn_test_loss']), min(loss_values['nn2_train_loss']), min(loss_values['nn2_test_loss'])) - 0.1
  ylim_max = max(max(loss_values['train_loss']), max(loss_values['test_loss']), max(loss_values['nn_train_loss']), max(loss_values['nn_test_loss']), max(loss_values['nn2_train_loss']), max(loss_values['nn2_test_loss'])) + 0.1
  xlim_max = max(loss_values['iteration'])
  plt.title('Train Test Loss\nNegative R squared Summed', fontsize=14)
  plt.xlim(0, xlim_max)
  plt.ylim(ylim_min, ylim_max)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Loss', fontsize=14)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.tight_layout()
  # plt.grid(True)
  plt.gca().spines[['top', 'right']].set_visible(False)

  plt.show()

  plt.plot(loss_values['iteration'], loss_values['nn_train_loss'], label="train NN1", color='#0000ff', linestyle='--')
  plt.plot(loss_values['iteration'], loss_values['nn_test_loss'], label="test NN1", color='#0000ff')

  plt.plot(loss_values['iteration'], loss_values['nn2_train_loss'], label="train NN2", color='#ff0000', linestyle='--')
  plt.plot(loss_values['iteration'], loss_values['nn2_test_loss'], label="test NN2", color='#ff0000')

  plt.title('Train Test Loss\nNegative R squared Per Network', fontsize=14)
  plt.xlim(0, xlim_max)
  plt.ylim(ylim_min, ylim_max)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Loss', fontsize=14)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # info = 'NN1 - MH deletion\nNN2 - MH-less deletions'
  # plt.text(.5, .05, info, ha='center')
  # plt.figtext(info, wrap=True, horizontalalignment='center', fontsize=12)
  plt.tight_layout()
  plt.gca().spines[['top', 'right']].set_visible(False)
  # plt.grid(True)
  plt.show()

  ylim_min = min(min(loss_values['train_rsq1']), min(loss_values['train_rsq2']), min(loss_values['test_rsq1']),
                 min(loss_values['test_rsq2'])) - 0.1
  ylim_max = max(max(loss_values['train_rsq1']), max(loss_values['train_rsq2']), max(loss_values['test_rsq1']),
                 max(loss_values['test_rsq2'])) + 0.1
  plt.plot(loss_values['iteration'], loss_values['train_rsq1'], label="train NN1", color='#0000ff', linestyle='--')
  plt.plot(loss_values['iteration'], loss_values['test_rsq1'], label="test NN1", color='#0000ff')

  plt.plot(loss_values['iteration'], loss_values['train_rsq2'], label="train NN2", color='#ff0000', linestyle='--')
  plt.plot(loss_values['iteration'], loss_values['test_rsq2'], label="test NN2", color='#ff0000')

  plt.title('Average RSQ values', fontsize=14)
  plt.xlim(0, xlim_max)
  plt.ylim(ylim_min, ylim_max)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('R Squared', fontsize=14)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # info = 'NN1 - MH deletion\nNN2 - MH-less deletions'
  # plt.text(.5, .05, info, ha='center')
  # plt.figtext(info, wrap=True, horizontalalignment='center', fontsize=12)
  plt.tight_layout()
  plt.gca().spines[['top', 'right']].set_visible(False)
  # plt.grid(True)
  plt.show()
  return


def learning_curves(all_data, total_values):
  rate_stats, bp_stats = load_statistics(all_data, total_values, model_folder + FOLDER_STAT_KEY)
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
  plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, knn.score(X, y))
  # train_scores_mean = -train_scores.mean(axis=1)
  # validation_scores_mean = -test_scores.mean(axis=1)
  # plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean)


def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, score):
  plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
  plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
  plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
  plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

  plt.title("Learning Curve - Score = {:.3f}".format(score))
  plt.xlabel("Training Set Size"), plt.ylabel("Mean Squared Error"), plt.legend(loc="best")
  # plt.annotate("R-Squared = {:.3f}".format(score), (0, 1))

  plt.tight_layout()
  plt.show()
  return


def plot_learning_curve_old(train_sizes, mean_training, mean_testing):
  plt.plot(train_sizes, mean_training, label='Training error')
  plt.plot(train_sizes, mean_testing, label='Validation error')
  plt.ylabel('MSE', fontsize=14)
  plt.xlabel('Training set size', fontsize=14)
  plt.title(f'Learning curves for KNN regression', fontsize=14)
  plt.legend()
  plt.xlim(0, max(train_sizes)+50)
  plt.ylim(min(min(mean_training), min(mean_testing))-0.0005, max(max(mean_training), max(mean_testing))+0.0005)
  plt.gca().spines[['top', 'right']].set_visible(False)
  plt.show()


def plot_mh_score_function(nn_params):
  data = defaultdict(list)
  col_names = ['MH Length', 'GC', 'MH Score']
  # Add normal MH
  for ns in range(5000):
    length = np.random.choice(range(1, 28+1))
    gc = np.random.uniform()
    features = np.array([length, gc])
    ms = helper.nn_match_score_function(nn_params, features)[0]
    data['Length'].append(length)
    data['GC'].append(gc)
    data['MH Score'].append(ms)
  df = pd.DataFrame(data)

  # Plot length vs. match score
  sns.violinplot(x='Length', y='MH Score', data=df, scale='width')
  plt.title('Learned Match Function: MH Length vs. MH Score')
  plt.tight_layout()
  plt.show()

  # Plot GC vs match score, color by length
  palette = sns.color_palette('hls', max(df['Length']) + 1)
  for length in range(1, max(df['Length'])+1):
    ax = sns.regplot(x='GC', y='MH Score', data=df.loc[df['Length'] == length], color=palette[length-1], label='Length: %s' % (length))
  plt.legend(loc='best')
  plt.xlim([0, 1])
  plt.title('GC vs. MH Score, colored by MH Length')
  plt.show()

  return


def plot_prediction_observation(data):
  plt.scatter(data['observation'], data['prediction'], c='crimson')
  p1 = max(max(data['prediction']), max(data['observation']))
  p2 = min(min(data['prediction']), min(data['observation']))
  # plt.plot([p1, p2], [p1, p2], 'b-')
  plt.plot(np.unique(data['observation']), np.poly1d(np.polyfit(data['observation'], data['prediction'], 1))(np.unique(data['observation'])))
  linreg = linregress(data['observation'], data['prediction'])
  plt.text(0.6, 0.5, 'R-squared = %0.2f' % linreg.rvalue)
  # plt.plot(data['observation'], linreg.intercept + linreg.slope * data['observation'], 'r')

  plt.title('Predictions vs Accuracy')
  plt.xlabel('Observations')
  plt.ylabel('Prediction')
  plt.ylim(0, p1)
  plt.xlim(0, p1+0.05)
  plt.tight_layout()
  plt.show()
  return


def get_predictions_and_observations_adaptation(all_data):
  data = mh_del_subset(all_data)[1]
  data = normalize_count(data)

  libA = load_lib_data(input_dir + 'libX/', 'libA')
  targets = get_targets(libA, data, with_grna=True)

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
  libA = load_lib_data(input_dir + 'libX/', 'libA')
  targets = get_targets(libA, data, with_grna=True)
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
    # Get prediction of GRNA - TODO Change based on grna item
    # normalized_fq = prediction[prediction['Sample_Name']]
    normalized_fq = []
    for i in range(1, -61, -1):
      if i != 0:
        normalized_fq.append(observation[key][i])

    # For dictionary, get items from 1 to -30 into an array

    # TODO - not sure if we should use the in-built pearson function? pearsonr
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(pred_normalized_fq[idx]) # TODO - check item to pick
    pearson_numerator = np.sum((normalized_fq - x_mean) * (pred_normalized_fq[idx] - y_mean))
    pearson_denom = np.sqrt(np.sum((normalized_fq - x_mean) ** 2) * np.sum((pred_normalized_fq[idx] - y_mean) ** 2))
    r_value = pearson_numerator / pearson_denom
    r_values.append(r_value)

    t_value = (r_value/(np.sqrt(1-(r_value ** 2)))) * np.sqrt(n-2)
    t_values.append(t_value)
  return r_values, t_values


def plot_student_t_distribution(t_values):
  plt.hist(t_values, density=True, edgecolor='black', bins=20)
  plt.show()
  return


def plot_pearson_correlation(pearson_co):
  # ax = sns.scatterplot(x="FlyAsh", y="Strength", data=pearson_co)
  # sns.lmplot(x="FlyAsh", y="Strength", data=pearson_co)
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Execution Details')

  parser.add_argument('--model_folder', dest='model_folder', type=str, help='Variable indicating the execution id of the trained neural network and knn')
  args = parser.parse_args()
  if args.model_folder:
    user_exec_id = args.model_folder
  else:
    raise Exception("Please specify --model_folder")

  out_dir, log_file, execution_id = initialize_files_and_folders(user_exec_id)
  if user_exec_id != execution_id:
    raise Exception("Please specify a valid pre-trained model")

  global log_fn
  log_fn = log_file

  global exec_id
  exec_id = execution_id
  global input_dir
  input_dir = EXECUTION_PATH + FOLDER_INPUT_KEY

  helper.print_and_log("Loading pre-trained networks...", log_fn)
  model_folder = out_dir + 'fig_4mesc/'
  nn_path = model_folder + FOLDER_PARAM_KEY
  models = load_models(model_folder)
  nn_params = models['nn']
  nn2_params = models['nn_2']
  rate_model = models['rate']
  bp_model = models['bp']
  normalizer = models['norm']

  loss_values = load_nn_statistics(model_folder)
  helper.print_and_log("Learning Curve for Neural Networks...", log_fn)
  # plot_nn_loss_epoch(loss_values)

  mesc_file = ''
  libA = load_lib_data(input_dir + 'libX/', 'libA')
  train_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}train_mesc.pkl'
  test_mesc_file = f'{out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY}test_mesc.pkl'
  prediction_files = os.listdir(out_dir + FOLDER_PRED_KEY)

  for prediction_file in prediction_files:
    if "mesc" in prediction_file:
      mesc_file = prediction_file
      break

  if mesc_file != '':
    predictions = load_predictions(out_dir + FOLDER_PRED_KEY + mesc_file)
    test_mesc = helper.load_pickle(test_mesc_file)
    # Get actual observations
    observations = get_observed_values(test_mesc)
  else:
    raise Exception("Please retrain the model")

  pred, obs, ins_only = get_predictions_and_observations(test_mesc, nn_params, nn2_params, rate_model, bp_model, normalizer)
  insertion_pred_obs = pd.DataFrame(ins_only).T

  pearson_co, t_values = get_pearson_pred_obs(pred, obs)
  plot_prediction_observation(insertion_pred_obs)
  plot_student_t_distribution(t_values)

  # [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(test_mesc)
  # INP, OBS, OBS2, NAMES, DEL_LENS = format_data(exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs)
  # Predict unseen values
  # rsq1, rsq2 = helper.rsq(nn_params, nn2_params, INP, OBS, OBS2, DEL_LENS, NAMES)
  #
  # plot_rsqs(rsq1, rsq2)

  helper.print_and_log("Original Learning Curve for Insertion Model...", log_fn)
  total_values = helper.load_pickle(model_folder + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
  learning_curves(all_data_mesc, total_values)


