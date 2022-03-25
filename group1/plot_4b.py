import statistics as stat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Code source
# JohanC, “How to display boxplot in front of violinplot in seaborn -
# seaborn zorder?” Aug 2021, last accessed 18 March 2022. [Online].
# Available: https://stackoverflow.com/questions/68614447/
# how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder
def box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS, save_file=''):
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


if __name__ == '__main__':
  # Genotype pearson correlation data
  corr_gentyp_mESC = np.array([99, 97, 86, 93, 85])
  corr_gentyp_U2OS = np.array([70, 80, 73, 65, 79])

  # Preparing dataframe
  box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS)

