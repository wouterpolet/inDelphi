import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Source
# https://stackoverflow.com/questions/68614447/how-to-display-boxplot-in-front-of-violinplot-in-seaborn-seaborn-zorder

# Genotype pearson correlation data
corr_gentyp_mESC = np.array([99, 97, 86, 93, 85])
corr_gentyp_U2OS = np.array([70, 80, 73, 65, 79])

# Preparing dataframe
corr_df = pd.DataFrame({"mESC": corr_gentyp_mESC, "U2OS": corr_gentyp_U2OS})
df = corr_df.melt(value_vars=['mESC', 'U2OS'], var_name='group')

# Plotting violins and boxes
ax = sns.violinplot(data=df,
                    x='group', y='value',
                    color="#af52f4", inner=None,
                    linewidth=0, saturation=0.5)
sns.boxplot(data=df,
            x='group', y='value',
            saturation=0.5, width=0.4,
            palette='rocket', boxprops={'zorder': 2}, ax=ax)

plt.show()