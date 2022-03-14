# Figure 3f plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def hist(predictions, save_file=''):
  predictions['Highest Del Rate'] = predictions['Highest Del Rate'].apply(lambda x: x * 100)
  predictions['Highest Ins Rate'] = predictions['Highest Ins Rate'].apply(lambda x: x * 100)

  bins_range = np.asarray(range(1, 101))
  print(bins_range)

  fix, (ax1, ax2) = plt.subplots(1, 2)
  fix.suptitle('Predicted frequency among major editing products using mESC-trained inDelphi (%)', wrap=True)
  fig_3f_data_del = np.asarray(predictions['Highest Del Rate'])

  count, bins, patches = ax1.hist(fig_3f_data_del, range=(0, 100), bins=bins_range, orientation='horizontal', edgecolor=None)
  #data = np.random.uniform(0, 1, 1000)  # You are generating 1000 points between 0 and 1.
  #count, bins, patches = ax1.hist(data, 100, orientation='horizontal')
  ax1.spines['left'].set_visible(False)
  ax1.spines['top'].set_visible(False)

  for i in range(0, 30):
    patches[i].set_facecolor('lightcoral')
  for i in range(30, 50):
    patches[i].set_facecolor('indianred')
  for i in range(50, len(patches)):
    patches[i].set_facecolor('brown')

  ax1.yaxis.tick_right()
  ax1.set_xlim(ax1.get_xlim()[::-1])

  ax1.set_xlabel('Number of Cas9 gRNAs from libB')

  fig_3f_data_ins = np.asarray(predictions['Highest Ins Rate'])
  count, bins, patches = ax2.hist(fig_3f_data_ins, range=(0, 100), bins=bins_range, orientation='horizontal')
  #data = np.random.uniform(0, 1, 1000)  # You are generating 1000 points between 0 and 1.
  # count, bins, patches = ax2.hist(data, 100, orientation='horizontal')
  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)

  for i in range(0, 30):
    patches[i].set_facecolor('lightskyblue')
  for i in range(30, 50):
    patches[i].set_facecolor('skyblue')
  for i in range(50, len(patches)):
    patches[i].set_facecolor('deepskyblue')

  ax2.yaxis.tick_left()
  ax2.set_yticklabels([])

  ax2.set_xlabel('Number of Cas9 gRNAs from libB')

  plt.show()

  if save_file != '':
    plt.savefig(save_file)


# https://stackoverflow.com/questions/43345599/process-pandas-dataframe-into-violinplot
def box(predictions, save_file=''):
  data1 = np.random.poisson(lam=1, size=100)
  data2 = np.random.choice(2, len(data1))
  data = pd.DataFrame({'Length': data1, 'MH Score': data2})
  sns.violinplot(x='Length', y='MH Score', data=data, scale='width')
  plt.ylabel('Counts')
  #
  # # Plot GC vs match score, color by length
  # palette = sns.color_palette('hls', max(data['Length']) + 1)
  # # for length in range(1, max(data['Length']) + 1):
  # #   ax = sns.regplot(x='GC', y='MH Score', data=data.loc[data['Length'] == length], color=palette[length - 1], label='Length: %s' % (length))
  # plt.legend(loc='best')
  # plt.xlim([0, 1])
  # plt.title('GC vs. MH Score, colored by MH Length')

  if save_file != '':
    plt.savefig(save_file)

  plt.close()

if __name__ == '__main__':
  # franz_path = "G:/My Drive/3) Work/Study/TU_Delft_MSc_Nanobiology/3-courses/Machine-Learning-in-Bioinformatics/inDelphi/shared/inDelphi/out/aaa/statistics/prediction_output.csv"
  # predictions = pd.read_csv(franz_path)
  # hist(predictions)
  box([], 'C:\\Development\\CS4260\\inDelphi\\out\\aaa\\statistics\\test.pdf')
