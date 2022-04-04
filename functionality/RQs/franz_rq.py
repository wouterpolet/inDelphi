import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Functions
############

# Open pickle files
def _pickle_load(file):
    data = pickle.load(open(file, 'rb'))
    return data


# Read from pickle files
def read_data(file):
    master_data = _pickle_load(file)
    return master_data['counts'], master_data['del_features']


# Get deletion distribution for each gRNA
def get_del_len_distribs(deletions_df):
    all_gRNAs = deletions_df['Sample_Name'].unique()
    all_gRNAs_del_len_distribs = {}
    gRNA_count = 1
    for single_gRNA in all_gRNAs:
        print(gRNA_count)
        gRNA_del_len_distrib = {}
        dels_of_single_gRNA = deletions_df[(deletions_df['Sample_Name'] == single_gRNA)]
        # total = 0
        for del_len in range(1, 31):
            gRNA_del_len_distrib[del_len] = \
                int(sum(dels_of_single_gRNA[dels_of_single_gRNA['Size'] == del_len]['countEvents'].tolist()))
            # total += gRNA_del_len_distrib[del_len]

        # Normalize
        # for length, count in gRNA_del_len_distrib.items():
        #     try:
        #         gRNA_del_len_distrib[length] = count / total
        #     except ZeroDivisionError:
        #         gRNA_del_len_distrib[length] = 0
        all_gRNAs_del_len_distribs[single_gRNA] = gRNA_del_len_distrib
        gRNA_count += 1
    return all_gRNAs_del_len_distribs


def get_del_len_distrib_per_type(deletions_df):

    # gRNA names
    exps = deletions_df['Sample_Name'].unique()

    # MH-based deletions
    MH_deletions = deletions_df[deletions_df['homologyLength'] != 0]
    MH_deletions = MH_deletions.reset_index()

    # MH-less deletions
    MHless_deletions = deletions_df[deletions_df['homologyLength'] == 0]
    MHless_deletions = MHless_deletions.reset_index()

    MH_freq = get_del_len_distribs(MH_deletions)
    MH_less_freq = get_del_len_distribs(MHless_deletions)

    return exps, MH_freq, MH_less_freq


# Data Prep
########


# Load data
input_dir = os.path.dirname(os.path.dirname(__file__)) + '/in/'
counts, del_features = read_data(input_dir + 'dataset.pkl')  # mESC
# counts, del_features = read_data(input_dir + 'U2OS.pkl')        # U2OS

# mESC df
mESC_merged_data = pd.concat([counts, del_features], axis=1)
mESC_merged_data = mESC_merged_data.reset_index()

# U2OS df


# mESC distributions
#####################

# Isolate MH-based and MH-less deletions
mESC_deletions = mESC_merged_data[mESC_merged_data['Type'] == 'DELETION']
# only one mESC_deletions
mESC_deletions_test = mESC_deletions[mESC_deletions['Sample_Name'] == '0_0_0_0_CTTTCACTTTATAGATTTAT']
mESC_exps, mESC_MH_freq, mESC_MH_less_freq = get_del_len_distrib_per_type(mESC_deletions_test)

# Dict -> Df
#mESC_MH_freq_df = pd.DataFrame.from_dict(mESC_MH_freq)
#del_len = np.asarray(list(range(1, 30+1)))
#mESC_MH_freq_df['Deletion length'] = del_len
#mESC_MH_less_freq_df = pd.DataFrame.from_dict(mESC_MH_less_freq)

# Plotting
###########
plt.hist(mESC_MH_freq['0_0_0_0_CTTTCACTTTATAGATTTAT'].keys(),
        weights=mESC_MH_freq['0_0_0_0_CTTTCACTTTATAGATTTAT'].values(),
        bins=range(30))
plt.show()

plt.hist(mESC_MH_less_freq['0_0_0_0_CTTTCACTTTATAGATTTAT'].keys(),
        weights=mESC_MH_less_freq['0_0_0_0_CTTTCACTTTATAGATTTAT'].values(),
        bins=range(30))
plt.show()
