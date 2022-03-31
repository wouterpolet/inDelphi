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


def plot_student_t_distribution(t_values):
  plt.hist(t_values, density=True, edgecolor='black', bins=20)
  plt.show()
  return


def plot_pearson_correlation(pearson_co):
  # ax = sns.scatterplot(x="FlyAsh", y="Strength", data=pearson_co)
  # sns.lmplot(x="FlyAsh", y="Strength", data=pearson_co)
  pass

