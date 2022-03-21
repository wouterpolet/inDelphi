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
    # 4bi
    corr_df1 = pd.DataFrame({"mESC": corr_gentyp_mESC})
    df = corr_df1.melt(value_vars=['mESC'], var_name='group')

    # 4bii
    corr_df2 = pd.DataFrame({"U2OS": corr_gentyp_U2OS})
    df2 = pd.concat([df, corr_df2.melt(value_vars=['U2OS'], var_name='group')])

    pri_clr_u2os = '#90ac4c'
    sec_clr_u2os = '#e0e4cc'

    ax = sns.violinplot(data=df2, x='group', y='value', palette={'mESC': sec_clr_mesc, 'U2OS': sec_clr_u2os}, inner=None, linewidth=0, saturation=0.5)
    sns.boxplot(data=df2, x='group', y='value', saturation=0.5, width=0.4, palette={'mESC': pri_clr_mesc, 'U2OS': pri_clr_u2os},
                ax=ax,
                boxprops = dict(zorder=2, linewidth=0),
                capprops = dict(color=gry_clr),
                whiskerprops = dict(color=gry_clr),
                medianprops = dict(color=wht_clr),
                )
    ax.set_xlabel('')
    plt.show()

if __name__ == '__main__':
  # Genotype pearson correlation data
  corr_gentyp_mESC = np.array([99, 97, 86, 93, 85])
  corr_gentyp_U2OS = np.array([70, 80, 73, 65, 79])

  # Preparing dataframe
  box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS)

