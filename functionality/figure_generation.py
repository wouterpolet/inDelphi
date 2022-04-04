import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stat
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats


def figure_3(predictions):
  """
  Generate Figure 3f - Deletion and Length Distributions
  @rtype: object
  """
  predictions['Highest Del Rate'] = predictions['Highest Del Rate'].apply(lambda x: x * 100)
  predictions['Highest Ins Rate'] = predictions['Highest Ins Rate'].apply(lambda x: x * 100)

  bins_range = np.asarray(range(1, 101))
  print(bins_range)

  fix, (ax1, ax2) = plt.subplots(1, 2)
  fix.suptitle('Predicted frequency among major editing products\nusing mESC-trained inDelphi (%)', wrap=True)
  fig_3f_data_del = np.asarray(predictions['Highest Del Rate'])

  count, bins, patches = ax1.hist(fig_3f_data_del, range=(0, 100), bins=bins_range, orientation='horizontal', edgecolor=None)
  ax1.axhline(30, color='black')
  ax1.annotate("{:.1f}%".format(sum(count[31:])/sum(count) * 100), xy=(40000, 30), xycoords='data',
              xytext=(-10, 10), textcoords='offset points',
              horizontalalignment='left', verticalalignment='top')
  ax1.axhline(50, color='black')
  ax1.annotate("{:.1f}%".format(sum(count[51:])/sum(count) * 100), xy=(40000, 50), xycoords='data',
              xytext=(-10, 10), textcoords='offset points',
              horizontalalignment='left', verticalalignment='top')
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

  # ax1.set_xlabel('Number of Cas9 gRNAs from libB')

  fig_3f_data_ins = np.asarray(predictions['Highest Ins Rate'])
  count, bins, patches = ax2.hist(fig_3f_data_ins, range=(0, 100), bins=bins_range, orientation='horizontal')
  #data = np.random.uniform(0, 1, 1000)  # You are generating 1000 points between 0 and 1.
  # count, bins, patches = ax2.hist(data, 100, orientation='horizontal')
  ax2.axhline(30, color='black')
  ax2.annotate("{:.1f}%".format(sum(count[31:])/sum(count) * 100), xy=(40000, 30), xycoords='data',
              xytext=(100, 10), textcoords='offset points',
              horizontalalignment='right', verticalalignment='top')
  ax2.axhline(50, color='black')
  ax2.annotate("{:.1f}%".format(sum(count[51:])/sum(count) * 100), xy=(40000, 50), xycoords='data',
              xytext=(100, 10), textcoords='offset points',
              horizontalalignment='right', verticalalignment='top')

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

  # ax2.set_xlabel('Number of Cas9 gRNAs from libB')

  ax1.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
  ax1.xaxis.get_offset_text().set_visible(False)

  ax2.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
  ax2.xaxis.get_offset_text().set_visible(False)
  fix.text(0.5, 0.04, 'Number of Cas9 gRNAs from human exons and introns ($10^{4}$)', ha='center')
  plt.show()

  # sns.distplot(fig_3f_data_del, kde=True, label='Population')
  # plt.title('Population Distribution', fontsize=18)
  # plt.ylabel('Frequency', fontsize=16)
  #
  # print(f'Population Mean: {np.mean(fig_3f_data_del):.3}')
  # print(f'Population Std: {np.std(fig_3f_data_del):.3}')
  # plt.show()
  #

  density = stats.gaussian_kde(fig_3f_data_del, bw_method='silverman')
  count, x, patches = ax1.hist(fig_3f_data_del, range=(0, 100), bins=bins_range, orientation='horizontal', edgecolor=None)
  plt.plot(x, density(x))
  plt.show()

  density = stats.gaussian_kde(fig_3f_data_ins, bw_method='silverman')
  count, x, patches = ax1.hist(fig_3f_data_ins, range=(0, 100), bins=bins_range, orientation='horizontal', edgecolor=None)
  plt.plot(x, density(x))
  plt.show()



#   We resampled each predicted value from a Gaussian centered at the predicted value with a specified standard deviation.
#   Set the standard deviation as the predicted value divided by 4, up to a maximum of 3% for insertions
#   while for deletions we used the predicted value divided by 4 with a minimum of 6%


# Code source
# JohanC, “How to display boxplot in front of violinplot in seaborn -
# seaborn zorder?” Aug 2021, last accessed 18 March 2022. [Online].
# Available: https://stackoverflow.com/questions/68614447/
# how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder
def figure_4(corr_gentyp_mESC, corr_gentyp_U2OS):
    """
    Generate Figure 4b - Indel length predictions box plots
    For mESCs and U2OS
    @rtype: object
    """
    wht_clr = 'white'
    gry_clr = '#889494'
    pri_clr_mesc = '#2596be'
    sec_clr_mesc = '#c8dcec'
    r_median_mesc = stat.median(corr_gentyp_mESC)
    r_median_u2os = stat.median(corr_gentyp_U2OS)
    labels = [f'mESCs\nN={len(corr_gentyp_mESC)}\nmedian\nr={format(r_median_mesc, ".2f")}',
              f'U2OS cells\nN={len(corr_gentyp_U2OS)}\nmedian\nr={format(r_median_u2os, ".2f")}']
    # 4bi
    corr_df1 = pd.DataFrame({"mESC": corr_gentyp_mESC})
    df = corr_df1.melt(value_vars=['mESC'], var_name='group')

    # 4bii
    corr_df2 = pd.DataFrame({"U2OS": corr_gentyp_U2OS})
    df2 = pd.concat([df, corr_df2.melt(value_vars=['U2OS'], var_name='group')])

    pri_clr_u2os = '#90ac4c'
    sec_clr_u2os = '#e0e4cc'

    ax = sns.violinplot(data=df2, x='group', y='value', palette={'mESC': sec_clr_mesc, 'U2OS': sec_clr_u2os}, inner=None
                        , linewidth=0, saturation=1)
    bx = sns.boxplot(data=df2, x='group', y='value', palette={'mESC': pri_clr_mesc, 'U2OS': pri_clr_u2os}, saturation=1
                     , width=0.4, ax=ax, boxprops=dict(zorder=2, linewidth=0), capprops=dict(color=gry_clr)
                     , whiskerprops=dict(color=gry_clr), medianprops=dict(color=wht_clr), showfliers=False)
    ax.set_xlabel('')
    bx.set_xticklabels(labels, rotation=0, fontsize=8)
    bx.set(title='', xlabel='',
           ylabel='Indel length prediction\nPearson correlation with observations\nin held-out Lib-A target sites')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_ylim([0, 1])
    bx.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

