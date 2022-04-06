import matplotlib.pyplot as plt


def plot_nn_loss_epoch(loss_values, save_file):
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

  if save_file != '':
    plt.savefig(save_file + '_group_loss.png')
  else:
    plt.show()
  plt.clf()

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
  if save_file != '':
    plt.savefig(save_file + '_split_loss.png')
  else:
    plt.show()
  plt.clf()

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
  if save_file != '':
    plt.savefig(save_file + '_average_rsq.png')
  else:
    plt.show()
  plt.clf()
  plt.clf()
  return


def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, score, plot_type, save_dir):
  plt.plot(train_sizes, train_mean, '--', color="#B1003F", label="Training score")
  plt.plot(train_sizes, test_mean, color="#2596be", label="Cross-validation score")
  plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#B1003F", alpha=0.5)
  plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#2596be", alpha=0.5)

  plt.title(plot_type + " Learning Curve - Score = {:.3f}".format(score))
  plt.xlabel("Training Set Size"), plt.ylabel("Mean Squared Error"), plt.legend(loc="best")
  # plt.annotate("R-Squared = {:.3f}".format(score), (0, 1))

  plt.tight_layout()
  if save_dir != '':
    plt.savefig(save_dir + plot_type + '_knn_curve.png')
  else:
    plt.show()
  plt.clf()
  return
