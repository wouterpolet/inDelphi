import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS, save_file=''):

    # 4bi
    corr_df1 = pd.DataFrame({"mESC": corr_gentyp_mESC})
    df = corr_df1.melt(value_vars=['mESC'], var_name='group')

    ax = sns.violinplot(data=df, x='group', y='value', color="#af52f4", inner=None, linewidth=0, saturation=0.5)
    sns.boxplot(data=df,
          x='group', y='value',
          saturation=0.5, width=0.4,
          palette='rocket', boxprops={'zorder': 2}, ax=ax)
    plt.show()

    # 4bii
    corr_df2 = pd.DataFrame({"U2OS": corr_gentyp_U2OS})
    df2 = corr_df2.melt(value_vars=['U2OS'], var_name='group')

    ax = sns.violinplot(data=df2, x='group', y='value', color="#af52f4", inner=None, linewidth=0, saturation=0.5)
    sns.boxplot(data=df2,
          x='group', y='value',
          saturation=0.5, width=0.4,
          palette='rocket', boxprops={'zorder': 2}, ax=ax)
    plt.show()

if __name__ == '__main__':
  # Source
  # https://stackoverflow.com/questions/68614447/how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder
  # Genotype pearson correlation data
  corr_gentyp_mESC = np.array([99, 97, 86, 93, 85])
  corr_gentyp_U2OS = np.array([70, 80, 73, 65, 79])

  # Preparing dataframe
  box_voilin(corr_gentyp_mESC, corr_gentyp_U2OS)

