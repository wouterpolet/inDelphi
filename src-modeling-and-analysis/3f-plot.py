# Figure 3f plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

franz_path = "G:/My Drive/3) Work/Study/TU_Delft_MSc_Nanobiology/3-courses/Machine-Learning-in-Bioinformatics/inDelphi/shared/inDelphi/out/aaa/statistics/prediction_output.csv"
predictions = pd.read_csv(franz_path)

predictions['Highest Del Rate'] = predictions['Highest Del Rate'].apply(lambda x: x*100)
predictions['Highest Ins Rate'] = predictions['Highest Ins Rate'].apply(lambda x: x*100)
#num_bins = 100
bins_range = np.asarray(range(1,101))
print(bins_range)

fig1 = plt.subplot()
fig_3f_data_del = np.asarray(predictions['Highest Del Rate'])
plt.hist(fig_3f_data_del, range=(0, 100), bins=bins_range)
plt.show()

fig2 = plt.subplot()
fig_3f_data_ins = np.asarray(predictions['Highest Ins Rate'])
plt.hist(fig_3f_data_ins, range=(0, 100), bins=bins_range)
plt.show()


