import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/43345599/process-pandas-dataframe-into-violinplot

# Genotype pearson correlation data
corr_gentyp_mESC = np.transpose(np.random.uniform(90, 100, 5))
corr_gentyp_U2OS = np.transpose(np.random.uniform(80, 90, 5))


numpy_data = np.array([corr_gentyp_mESC, corr_gentyp_U2OS])
df = pd.DataFrame(data=numpy_data, columns=['mESC', 'U2OS'])
print(df)

# ax = sns.violinplot(data=df,
#                     x='group', y='value',
#                     color="#af52f4", inner=None,
#                     linewidth=0, saturation=0.5)
# sns.boxplot(data=df,
#             x='group', y='value',
#             saturation=0.5, width=0.4,
#             palette='rocket', boxprops={'zorder': 2}, ax=ax)
# plt.show()