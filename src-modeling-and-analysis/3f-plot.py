# Figure 3f plotting
import pandas as pd
import matplotlib.pyplot as plt

path = "G:/My Drive/3) Work/Study/TU_Delft_MSc_Nanobiology/3-courses/Machine-Learning-in-Bioinformatics/inDelphi/shared/inDelphi/out/aaa/statistics/prediction_output.csv"
predictions = pd.read_csv(path)

fig_3f_data_del = predictions['Highest Del Rate'].apply(lambda x: x*100)
fig_3f_data_ins = predictions['Highest Ins Rate'].apply(lambda x: x*100)
#num_bins = 100

predictions.hist(column=['Highest Del Rate', 'Highest Ins Rate'])
plt.show()

#plt.hist(fig_3f_data_del, num_bins)
# plt.hist(fig_3f_data_del)
# plt.show()
# print(max(fig_3f_data_del))
# print(min(fig_3f_data_del))


