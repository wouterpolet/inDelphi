import os
import argparse
import pandas as pd
import autograd.numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import helper
from helper_jon import load_nn_statistics, save_statistics
from ins_network import load_statistics, featurize
from all_func import initialize_files_and_folders, load_neural_networks

FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_INPUT_KEY = '/in/'
EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))


def plot_nn_loss(loss_values):
  plt.plot(loss_values['iteration'], loss_values['train_loss'], label="train", color='red', marker='o')
  plt.plot(loss_values['iteration'], loss_values['test_loss'], label="test", color='blue', marker='o')
  plt.title('Train Test Loss', fontsize=14)
  plt.xlabel('Epoch', fontsize=14)
  plt.ylabel('Loss', fontsize=14)
  plt.grid(True)
  plt.show()
  return


def learning_curves(all_data, total_values):
  rate_stats, bp_stats = load_statistics(all_data, total_values, model_folder + FOLDER_STAT_KEY)
  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  X, y, Normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')

  # Train rate model
  model = KNeighborsRegressor()

  cross_val = [5, 10, 50, 100]
  for val in cross_val:
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=val, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -test_scores.mean(axis=1)
    plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, val, model)


def plot_learning_curve(train_sizes, mean_training, mean_testing, cross_val, model_name):
  plt.style.use('seaborn')
  plt.plot(train_sizes, mean_training, label='Training error')
  plt.plot(train_sizes, mean_testing, label='Validation error')
  plt.ylabel('MSE', fontsize=14)
  plt.xlabel('Training set size', fontsize=14)
  plt.title(f'Learning curves for KNN regression - CV: {cross_val} - Model: {model_name}', fontsize=14)
  plt.legend()
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


def plot_pred_obs(nn_params, nn2_params, inp, obs, del_lens, nms):
  num_samples = len(inp)
  ctr = 0
  pred = []
  obs_dls = []
  for idx in range(len(inp)):
    mh_scores = helper.nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))
    curr_pred = np.zeros(30)
    curr_obs = np.zeros(30)
    for jdx in range(len(del_lens[idx])):
      dl_idx = int(del_lens[idx][jdx]) - 1
      curr_pred[dl_idx] += normalized_fq[jdx]
      curr_obs[dl_idx] += obs[idx][jdx]
    pred.append(curr_pred.flatten())
    obs_dls.append(curr_obs.flatten())


  for idx in range(len(inp)):
    ymax = max(max(pred[idx]), max(obs_dls[idx])) + 0.05
    rsq = pearsonr(obs_dls[idx], pred[idx])[0] ** 2

    fig, ax = plt.subplots()
    ax.scatter(obs_dls[idx], pred[idx])
    ax.plot([0, ymax], [0, ymax], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('R2: ' + str(r2_score(obs_dls[idx], pred[idx])) + 'Len: ' + str(idx))
    # regression line
    y_test, y_predicted = obs_dls[idx].reshape(-1, 1), pred[idx].reshape(-1, 1)
    ax.plot(y_test, LinearRegression().fit(y_test, y_predicted).predict(y_test))
    plt.show()
    #
    # plt.subplot(211)
    # plt.title('Designed Oligo %s, Rsq=%s' % (nms[idx], rsq))
    # plt.bar(range(1, 30+1), obs_dls[idx], align='center', color='#D00000')
    # plt.xlim([0, 30+1])
    # plt.ylim([0, ymax])
    # plt.ylabel('Observed')
    #
    # plt.subplot(212)
    # plt.bar(range(1, 30+1), pred[idx], align='center', color='#FFBA08')
    # plt.xlim([0, 30+1])
    # plt.ylim([0, ymax])
    # plt.ylabel('Predicted')
    # plt.show()

    ctr += 1
    if ctr >= 50:
      break
  return


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
  model_folder = out_dir + 'fig_3/'
  nn_path = model_folder + FOLDER_PARAM_KEY
  nn_params, nn2_params = load_neural_networks(nn_path)
  loss_values = load_nn_statistics(model_folder)

  helper.print_and_log("Learning Curve for Neural Networks...", log_fn)
  plot_nn_loss(loss_values)
  #
  # helper.print_and_log("Learning Curve for Insertion Model...", log_fn)
  # total_values = helper.load_pickle(model_folder + FOLDER_PARAM_KEY + 'total_phi_delfreq.pkl')
  # all_data_mesc = pd.concat(helper.read_data(input_dir + 'dataset.pkl'), axis=1).reset_index()
  # learning_curves(all_data_mesc, total_values)
