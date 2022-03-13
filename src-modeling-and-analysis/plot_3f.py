# Figure 3f plotting
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def hist(predictions, save_file=''):
  predictions['Highest Del Rate'] = predictions['Highest Del Rate'].apply(lambda x: x*100)
  predictions['Highest Ins Rate'] = predictions['Highest Ins Rate'].apply(lambda x: x*100)
  #num_bins = 100
  bins_range = np.asarray(range(1,101))
  print(bins_range)

  fix, ax = plt.subplots()
  fig_3f_data_del = np.asarray(predictions['Highest Del Rate'])
  ax.hist(fig_3f_data_del, range=(0, 100), bins=bins_range, orientation='horizontal')
  ax.yaxis.tick_right()
  ax.set_xlim(ax.get_xlim()[::-1])
  ax.set_title('Predicted frequency among major editing products using mESC-trained inDelphi (%)', loc='center', wrap=True)
  ax.set_xlabel('Number of Cas9 gRNAs from libB')
  plt.show()

  if save_file != '':
    plt.savefig(save_file)



  # fig2 = plt.subplot()
  # fig_3f_data_ins = np.asarray(predictions['Highest Ins Rate'])
  # plt.hist(fig_3f_data_ins, range=(0, 100), bins=bins_range, orientation='horizontal')
  # plt.show()


if __name__ == '__main__':
  franz_path = "G:/My Drive/3) Work/Study/TU_Delft_MSc_Nanobiology/3-courses/Machine-Learning-in-Bioinformatics/inDelphi/shared/inDelphi/out/aaa/statistics/prediction_output.csv"
  predictions = pd.read_csv(franz_path)
  hist(predictions)
