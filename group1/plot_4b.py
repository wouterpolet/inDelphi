import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS, save_file=''):
    wht_clr = 'white'
    gry_clr = '#889494'
    pri_clr = '#2596be'
    sec_clr = '#c8dcec'
    # 4bi
    corr_df1 = pd.DataFrame({"mESC": corr_gentyp_mESC})
    df = corr_df1.melt(value_vars=['mESC'], var_name='group')

    ax = sns.violinplot(data=df, x='group', y='value', color=sec_clr, inner=None, linewidth=0, saturation=0.5)
    sns.boxplot(data=df, x='group', y='value', saturation=0.5, width=0.4, palette='rocket', ax=ax,
                boxprops=dict(facecolor=pri_clr, color=pri_clr, zorder=2),
                flierprops=dict(color=pri_clr, markeredgecolor=pri_clr),
                capprops=dict(color=gry_clr),
                whiskerprops=dict(color=gry_clr),
                medianprops=dict(color=wht_clr)
                )
    plt.show()

    # 4bii
    corr_df2 = pd.DataFrame({"U2OS": corr_gentyp_U2OS})
    df2 = corr_df2.melt(value_vars=['U2OS'], var_name='group')

    pri_clr = '#90ac4c'
    sec_clr = '#e0e4cc'
    ax = sns.violinplot(data=df2, x='group', y='value', color=sec_clr, inner=None, linewidth=0, saturation=0.5)
    sns.boxplot(data=df2, x='group', y='value', saturation=0.5, width=0.4, palette='rocket', ax=ax,
               boxprops = dict(facecolor=pri_clr, color=pri_clr, zorder=2),
               flierprops = dict(color=pri_clr, markeredgecolor=pri_clr),
               capprops = dict(color=gry_clr),
               whiskerprops = dict(color=gry_clr),
               medianprops = dict(color=wht_clr)
              )
    plt.show()

if __name__ == '__main__':
  # Source
  # https://stackoverflow.com/questions/68614447/how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder
  # Genotype pearson correlation data
  corr_gentyp_mESC = np.array([99, 97, 86, 93, 85])
  corr_gentyp_U2OS = np.array([70, 80, 73, 65, 79])

  # Preparing dataframe
  box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS)

