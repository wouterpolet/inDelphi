import numpy as np
import pandas as pd
import seaborn as sns
import statistics as stat
import matplotlib.pyplot as plt


def hist(predictions):
  """
  Generate Figure 3f - Deletion and Length Distributions
  @rtype: object
  """
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


# Code source
# JohanC, “How to display boxplot in front of violinplot in seaborn -
# seaborn zorder?” Aug 2021, last accessed 18 March 2022. [Online].
# Available: https://stackoverflow.com/questions/68614447/
# how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder
def box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS):
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
                     , whiskerprops=dict(color=gry_clr), medianprops=dict(color=wht_clr))
    ax.set_xlabel('')
    bx.set_xticklabels(labels, rotation=0, fontsize=8)
    bx.set(title='', xlabel='',
           ylabel='Indel length prediction\nPearson correlation with observations\nin held-out Lib-A target sites')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ax.set_ylim([0, 1])
    # bx.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

