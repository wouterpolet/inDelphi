import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import collections                # for summing up dictionaries
import functionality.helper as helper
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_folder)



'''
Functions
'''


# Reading files: Open pickle files
def _pickle_load(file):
    data = pickle.load(open(file, 'rb'))
    return data


# Add fraction of full MH distribution to non full MH and MH less distributions
def append_fraction_of_full_MH_distrib(full_MH_distrib, non_full_MH_distrib, MH_less_distrib, frac=0.5):
    # frac = % going to non full MH

    # Getting half of the full MH distribution
    full_MH_for_non_full_MH = {}
    full_MH_for_MH_less = {}
    for del_len in range(1, 30 + 1):
        full_MH_for_non_full_MH[del_len] = float(frac * full_MH_distrib[del_len])
        full_MH_for_MH_less[del_len] = float((1 - frac) * full_MH_distrib[del_len])

    # Non full MH distribution + 50% of full MH distribution
    non_full_MH_and_full_MH_distribs = list([])
    non_full_MH_and_full_MH_distribs.append(full_MH_for_non_full_MH)
    non_full_MH_and_full_MH_distribs.append(non_full_MH_distrib)

    non_full_MH_counter = collections.Counter()
    for distrib in non_full_MH_and_full_MH_distribs:
        non_full_MH_counter.update(distrib)
    non_full_MH_and_half_of_full_MH_distrib = dict(non_full_MH_counter)

    # MH less distribution + 50% of full MH distribution
    MH_less_and_full_MH_distribs = list([])
    MH_less_and_full_MH_distribs.append(full_MH_for_MH_less)
    MH_less_and_full_MH_distribs.append(MH_less_distrib)

    MH_less_counter = collections.Counter()
    for distrib in MH_less_and_full_MH_distribs:
        MH_less_counter.update(distrib)
    MH_less_and_half_of_full_MH_distrib = dict(MH_less_counter)

    return non_full_MH_and_half_of_full_MH_distrib, MH_less_and_half_of_full_MH_distrib


# Plot the MH-dep and MH-indep distributions for both wt and NHEJ-KO cases
def four_histograms(wt_MH, wt_MH_less, KO_MH, KO_MH_less):
    del_bins = np.arange(1, 30 + 2)  # 30 bins, 31 edges needed
    fig, axs = plt.subplots(2, 2)
    # wt MH dep
    axs[0, 0].hist(list(wt_MH.keys()),
                   weights=list(wt_MH.values()),
                   bins=del_bins, align='left', edgecolor='black', linewidth=1)
    axs[0, 0].set_title('MH-dependent \n wt')
    # wt MH indep
    axs[0, 1].hist(list(wt_MH_less.keys()),
                   weights=list(wt_MH_less.values()),
                   bins=del_bins, align='left', edgecolor='black', linewidth=1)
    axs[0, 1].set_title('MH-independent \n wt')

    # KO MH dep
    axs[1, 0].hist(list(KO_MH.keys()),
                   weights=list(KO_MH.values()),
                   bins=del_bins, align='left', edgecolor='black', linewidth=1)
    axs[1, 0].set_title('NHEJ-KO')
    # KO MH indep
    axs[1, 1].hist(list(KO_MH_less.keys()),
                   weights=list(KO_MH_less.values()),
                   bins=del_bins, align='left', edgecolor='black', linewidth=1)
    axs[1, 1].set_title('NHEJ-KO')

    for ax in axs.flat:
        ax.set(xlabel='Deletion length', ylabel='Relative frequency')
        ax.set_xlim([0, 31])
        ax.set_ylim([0, 0.073])

    fig.tight_layout()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    return None


# Reading files: Read from pickle files
def read_data(file):
    master_data = _pickle_load(file)
    return master_data['counts'], master_data['del_features']


# Generating 1D del len distributions for a type of deletion for a set of gRNAs
def get_distribs(deletions_of_gRNAs_in_set, distrib_number, KO_type):
    all_gRNAs = deletions_of_gRNAs_in_set['Sample_Name'].unique()
    all_gRNAs_del_len_distribs = {}
    gRNA_count = 1

    for gRNA in all_gRNAs:
        if gRNA_count % 100 == 0:
            print("gRNA #", str(gRNA_count), ", distrib #", str(distrib_number), "KO type:", KO_type)
        gRNA_del_len_distrib = {}
        dels_of_current_gRNA = deletions_of_gRNAs_in_set[(deletions_of_gRNAs_in_set['Sample_Name'] == gRNA)]

        for del_len in range(1, 30 + 1):
            gRNA_del_len_distrib[del_len] = \
                int(sum(dels_of_current_gRNA[dels_of_current_gRNA['Size'] == del_len]['countEvents'].tolist()))

        all_gRNAs_del_len_distribs[gRNA] = gRNA_del_len_distrib
        gRNA_count += 1

    return all_gRNAs_del_len_distribs


# Generating 1D del len distributions from a deletions of a set of gRNAs
def get_del_len_distribs(deletions_of_gRNA_set, KO_type='wt'):
    # Collecting gRNA names
    gRNA_names = deletions_of_gRNA_set['Sample_Name'].unique()

    # Collecting all full and non-full MH-based deletions
    MH_deletions_of_gRNA_set = deletions_of_gRNA_set[deletions_of_gRNA_set['homologyLength'] != 0]
    full_MH_deletions_of_gRNA_set = MH_deletions_of_gRNA_set[
        MH_deletions_of_gRNA_set['homologyLength'] == MH_deletions_of_gRNA_set['Size']]
    full_MH_deletions_of_gRNA_set = full_MH_deletions_of_gRNA_set.reset_index()
    non_full_MH_deletions_of_gRNA_set = MH_deletions_of_gRNA_set[
        MH_deletions_of_gRNA_set['homologyLength'] != MH_deletions_of_gRNA_set['Size']]
    non_full_MH_deletions_of_gRNA_set = non_full_MH_deletions_of_gRNA_set.reset_index()

    # Collecting MH-less deletions
    MHless_deletions_of_gRNA_set = deletions_of_gRNA_set[deletions_of_gRNA_set['homologyLength'] == 0]
    MHless_deletions_of_gRNA_set = MHless_deletions_of_gRNA_set.reset_index()

    # Getting del len distributions of all gRNAs
    print("Getting del len distribs of all dels...")
    del_len_distribs_of_gRNA_set = get_distribs(deletions_of_gRNA_set, 1, KO_type)

    print("Getting del len distribs of full MH dels")
    full_MH_distribs_of_gRNA_set = get_distribs(full_MH_deletions_of_gRNA_set, 2, KO_type)

    print("Getting del len distribs of non-full MH dels...")
    non_full_MH_distribs_of_gRNA_set = get_distribs(non_full_MH_deletions_of_gRNA_set, 3, KO_type)

    print("Getting del len distribs of MH-less dels...")
    MH_less_distribs_of_gRNA_set = get_distribs(MHless_deletions_of_gRNA_set, 4, KO_type)

    return gRNA_names, del_len_distribs_of_gRNA_set, full_MH_distribs_of_gRNA_set, non_full_MH_distribs_of_gRNA_set, MH_less_distribs_of_gRNA_set


# From MH/MH-less del len distribs for all gRNAs, return avg'd and norm'd MH/MH-less del len distrib
def accumulate_distribs_across_gRNAs(del_len_distribs_for_gRNA_set):
    # Get the dict-distribution of each gRNA in the set into a list
    all_gRNA_del_len_distribs = list([])
    gRNA_names = list(del_len_distribs_for_gRNA_set.keys())
    gRNA_count = len(gRNA_names)
    for gRNA in gRNA_names:
        all_gRNA_del_len_distribs.append(del_len_distribs_for_gRNA_set[gRNA])

    # Sum up the counts of deletions for each length across all gRNAs
    counter = collections.Counter()
    for gRNA_del_len_distrib in all_gRNA_del_len_distribs:
        counter.update(gRNA_del_len_distrib)
    cumul_del_len_distrib = dict(counter)

    # Average the del length distribution
    for del_len in cumul_del_len_distrib:
        cumul_del_len_distrib[del_len] = float(cumul_del_len_distrib[del_len] / gRNA_count)

    return cumul_del_len_distrib


# Plot histogram of given distribution
def histogram(del_distrib, ylabel, ylim=[None, None], xscale=[0, 0, 2, 1], title=''):
    del_lengths = list(del_distrib.keys())
    del_length_freqs = list(del_distrib.values())

    ax = plt.axes(xscale)
    del_bins = np.arange(1, 30 + 2)  # 30 bins, 31 edges needed
    ax.hist(del_lengths, weights=del_length_freqs, bins=del_bins, align='left', edgecolor='black', linewidth=1)
    ax.set_xlim([0, 31])
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_xlabel('Deletion length')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, 31))
    plt.show()

    return None


# Normalise the del length distributions relative to the distrib of all dels
def normalise(all_dels_distrib, full_MH_distrib, non_full_MH_distrib, MH_less_distrib):
    total_del_count_of_avg_gRNA = sum(all_dels_distrib.values())
    for del_len in range(1, 30 + 1):
        all_dels_distrib[del_len] = float(all_dels_distrib[del_len] / total_del_count_of_avg_gRNA)
        full_MH_distrib[del_len] = float(full_MH_distrib[del_len] / total_del_count_of_avg_gRNA)
        non_full_MH_distrib[del_len] = float(non_full_MH_distrib[del_len] / total_del_count_of_avg_gRNA)
        MH_less_distrib[del_len] = float(MH_less_distrib[del_len] / total_del_count_of_avg_gRNA)

    return all_dels_distrib, full_MH_distrib, non_full_MH_distrib, MH_less_distrib


'''
Data sources
'''


#franz_data_pertubed_train.pkl'
#franz_data_pertubed_test.pkl'



'''
Data preparation
'''

# Load data
input_dir = helper.INPUT_DIRECTORY
mESC_counts, mESC_del_features = helper.read_data(input_dir + 'dataset.pkl')  # mESC

# mESC df
mESC_merged_data = pd.concat([mESC_counts, mESC_del_features], axis=1)
mESC_merged_data = mESC_merged_data.reset_index()

# Pertubed dataset (mESC)
rq_franz = input_dir + 'franz/'
pertubed_train_counts, pertubed_train_del_features = helper.read_data(rq_franz + 'franz_data_pertubed_train.pkl.pkl')
pertubed_test_counts, pertubed_test_del_features = helper.read_data(rq_franz + 'franz_data_pertubed_test.pkl.pkl')

# mESC pertubed dfs
mESC_perturbed_train_merged_data = pd.concat([pertubed_train_counts, pertubed_train_del_features], axis=1)
mESC_perturbed_train_merged_data = mESC_perturbed_train_merged_data.reset_index()

mESC_perturbed_test_merged_data = pd.concat([pertubed_test_counts, pertubed_test_del_features], axis=1)
mESC_perturbed_test_merged_data = mESC_perturbed_test_merged_data.reset_index()

mESC_perturbed_merged_data = pd.concat([mESC_perturbed_train_merged_data, mESC_perturbed_test_merged_data])

'''
Calculating distributions
'''

# Get all deletions (MH-dep and MH-indep) for all gRNAs
mESC_deletions_for_all_gRNAs = mESC_merged_data[mESC_merged_data['Type'] == 'DELETION']
KO_mESC_deletions_for_all_gRNAs = mESC_perturbed_merged_data[mESC_perturbed_merged_data['Type'] == 'DELETION']

# Get a mESC test set for running and building the code
gRNA_names = mESC_deletions_for_all_gRNAs['Sample_Name'].unique()
test_gRNAs_desired_count = 5
test_gRNAs_desired = list([])
for gRNA in range(test_gRNAs_desired_count):
  test_gRNAs_desired.append(gRNA_names[gRNA])
mESC_deletions_for_test_gRNAs = mESC_deletions_for_all_gRNAs[ mESC_deletions_for_all_gRNAs['Sample_Name'].isin(test_gRNAs_desired) ]

# Get a KO mESC test set for running and building the code
KO_gRNA_names = KO_mESC_deletions_for_all_gRNAs['Sample_Name'].unique()
KO_test_gRNAs_desired = list([])
for gRNA in range(test_gRNAs_desired_count):
  KO_test_gRNAs_desired.append(KO_gRNA_names[gRNA])
KO_mESC_deletions_for_test_gRNAs = KO_mESC_deletions_for_all_gRNAs[ KO_mESC_deletions_for_all_gRNAs['Sample_Name'].isin(KO_test_gRNAs_desired) ]

#--- CHANGE THIS FOR DESIRED OUTPUT!
model = 'test' # 'test' or 'full'

if model == 'test': # Only the first (test number) of gRNAs considered
  # wt mESC
  mESC_gRNA_names, mESC_dels_distribs_for_all_gRNAs, mESC_full_MH_distribs_for_all_gRNAs, mESC_non_full_MH_distribs_for_all_gRNAs, mESC_MH_less_distribs_for_all_gRNAs = get_del_len_distribs(mESC_deletions_for_test_gRNAs, KO_type='wt')

  #pertubed mESC
  KO_mESC_gRNA_names, KO_mESC_dels_distribs_for_all_gRNAs, KO_mESC_full_MH_distribs_for_all_gRNAs, KO_mESC_non_full_MH_distribs_for_all_gRNAs, KO_mESC_MH_less_distribs_for_all_gRNAs = get_del_len_distribs(KO_mESC_deletions_for_test_gRNAs, KO_type='NHEJ-KO')

else:               # All gRNAs considered
  # wt mESC
  mESC_gRNA_names, mESC_dels_distribs_for_all_gRNAs, mESC_full_MH_distribs_for_all_gRNAs, mESC_non_full_MH_distribs_for_all_gRNAs, mESC_MH_less_distribs_for_all_gRNAs = get_del_len_distribs(mESC_deletions_for_all_gRNAs, KO_type='wt')

  # pertubed mESC
  KO_mESC_gRNA_names, KO_mESC_dels_distribs_for_all_gRNAs, KO_mESC_full_MH_distribs_for_all_gRNAs, KO_mESC_non_full_MH_distribs_for_all_gRNAs, KO_mESC_MH_less_distribs_for_all_gRNAs = get_del_len_distribs(KO_mESC_deletions_for_all_gRNAs, KO_type='NHEJ-KO')

'''
Visualising mESC distributions
'''
# Accumulated (unnormalised) distributions for the average gRNA
mESC_dels_distrib_counts = accumulate_distribs_across_gRNAs(mESC_dels_distribs_for_all_gRNAs)
mESC_MH_full_distrib_counts = accumulate_distribs_across_gRNAs(mESC_full_MH_distribs_for_all_gRNAs)
mESC_non_full_MH_distrib_counts = accumulate_distribs_across_gRNAs(mESC_non_full_MH_distribs_for_all_gRNAs)
mESC_MH_less_distrib_counts = accumulate_distribs_across_gRNAs(mESC_MH_less_distribs_for_all_gRNAs)

# histogram(mESC_dels_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='wt mESC avg gRNA all dels')
# histogram(mESC_MH_full_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='wt mESC avg gRNA full MH dels')
# histogram(mESC_non_full_MH_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='wt mESC avg gRNA non full MH dels')
# histogram(mESC_MH_less_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='wt mESC avg gRNA MH-less dels')

# Normalised Del len distribution for the average gRNA
mESC_dels_distrib_rel_freqs, mESC_full_MH_distrib_rel_freqs, mESC_non_full_MH_distrib_rel_freqs, mESC_MH_less_distrib_rel_freqs = normalise(mESC_dels_distrib_counts, mESC_MH_full_distrib_counts, mESC_non_full_MH_distrib_counts, mESC_MH_less_distrib_counts)

# Appending a fraction of full MH distribution to nonfull MH and MH-less distributions
mESC_non_full_MH_and_half_of_full_MH_distrib, mESC_MH_less_and_half_of_full_MH_distrib = append_fraction_of_full_MH_distrib(mESC_full_MH_distrib_rel_freqs, mESC_non_full_MH_distrib_rel_freqs, mESC_MH_less_distrib_rel_freqs, 0.5)

# histogram(mESC_dels_distrib_rel_freqs, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA all dels')
# histogram(mESC_full_MH_distrib_rel_freqs, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA full MH dels')
# histogram(mESC_non_full_MH_distrib_rel_freqs, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA non full MH dels')
# histogram(mESC_non_full_MH_and_half_of_full_MH_distrib, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA non full MH + 50% full MH dels')
# histogram(mESC_MH_less_distrib_rel_freqs, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA MH-less dels')
# histogram(mESC_MH_less_and_half_of_full_MH_distrib, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='wt mESC avg gRNA MH-less + 50% full MH dels')

'''
Visualising KO mESC distributions
'''
# Accumulated (unnormalised) distributions for the average gRNA
KO_mESC_dels_distrib_counts = accumulate_distribs_across_gRNAs(KO_mESC_dels_distribs_for_all_gRNAs)
KO_mESC_MH_full_distrib_counts = accumulate_distribs_across_gRNAs(KO_mESC_full_MH_distribs_for_all_gRNAs)
KO_mESC_non_full_MH_distrib_counts = accumulate_distribs_across_gRNAs(KO_mESC_non_full_MH_distribs_for_all_gRNAs)
KO_mESC_MH_less_distrib_counts = accumulate_distribs_across_gRNAs(KO_mESC_MH_less_distribs_for_all_gRNAs)

# histogram(KO_mESC_dels_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='KO mESC avg gRNA MH-less dels')
# histogram(KO_mESC_MH_full_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='KO mESC avg gRNA full MH dels')
# histogram(KO_mESC_non_full_MH_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='KO mESC avg gRNA non full MH dels')
# histogram(KO_mESC_MH_less_distrib_counts, ylabel='Counts', xscale=[0,0,1,1], ylim=[0, 16500], title='KO mESC avg gRNA MH-less dels')

# Normalised Del length distribution for the average gRNA
KO_mESC_dels_distrib_rel_freqs, KO_mESC_full_MH_distrib_rel_freqs, KO_mESC_non_full_MH_distrib_rel_freqs, KO_mESC_MH_less_distrib_rel_freqs = normalise(KO_mESC_dels_distrib_counts, KO_mESC_MH_full_distrib_counts, KO_mESC_non_full_MH_distrib_counts, KO_mESC_MH_less_distrib_counts)

# Appending a fraction of full MH distribution to nonfull MH and MH-less distributions
KO_mESC_non_full_MH_and_half_of_full_MH_distrib, KO_mESC_MH_less_and_half_of_full_MH_distrib = append_fraction_of_full_MH_distrib(KO_mESC_full_MH_distrib_rel_freqs, KO_mESC_non_full_MH_distrib_rel_freqs, KO_mESC_MH_less_distrib_rel_freqs, 0.5)

# histogram(KO_mESC_dels_distrib_rel_freqs, ylabel='Relative frequency', xscale=[0,0,1,1], ylim=[0,0.11], title='KO mESC avg gRNA MH-less dels')
# histogram(KO_mESC_full_MH_distrib_rel_freqs, ylabel='Relative frequency', xscale=[0,0,1,1], ylim=[0,0.11], title='KO mESC avg gRNA full MH dels')
# histogram(KO_mESC_non_full_MH_distrib_rel_freqs, ylabel='Relative frequency', xscale=[0,0,1,1], ylim=[0,0.11], title='KO mESC avg gRNA non full MH dels')
# histogram(KO_mESC_non_full_MH_and_half_of_full_MH_distrib, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='KO mESC avg gRNA non full MH + 50% full MH dels')
# histogram(KO_mESC_MH_less_distrib_rel_freqs, ylabel='Relative frequency', xscale=[0,0,1,1], ylim=[0,0.11], title='KO mESC avg gRNA MH-less dels')
# histogram(KO_mESC_MH_less_and_half_of_full_MH_distrib, ylabel='Relative frequency', ylim=[0,0.11], xscale=[0,0,1,1], title='KO mESC avg gRNA MH-less + 50% full MH dels')

four_histograms(mESC_non_full_MH_and_half_of_full_MH_distrib, mESC_MH_less_and_half_of_full_MH_distrib, KO_mESC_non_full_MH_and_half_of_full_MH_distrib, KO_mESC_MH_less_and_half_of_full_MH_distrib)