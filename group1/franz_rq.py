import math
import os
import pickle
import pandas as pd

def _pickle_load(file):
  data = pickle.load(open(file, 'rb'))
  return data

def read_data(file):
  master_data = _pickle_load(file)
  return master_data['counts'], master_data['del_features']

def get_del_type_freqs(deletions_df):     # input: MH-based deletions

  MH_deletions = deletions_df[deletions_df['homologyLength'] != 0]
  MH_deletions = MH_deletions.reset_index()

  MHless_deletions = deletions_df[deletions_df['homologyLength'] == 0]
  MHless_deletions = MHless_deletions.reset_index()

  exps = deletions_df['Sample_Name'].unique()

  freqs = []
  MH_freqs = {}
  MH_less_freq = {}
  i = 0
  for id, exp in enumerate(exps):
    print(i)
    # Microhomology computation
    exp_MH_dels = MH_deletions[MH_deletions['Sample_Name'] == exp][
      ['Sample_Name', 'Indel', 'countEvents', 'Size', 'homologyLength', 'homologyGCContent']]

    exp_MH_less_dels = MHless_deletions[MHless_deletions['Sample_Name'] == exp][
      ['Sample_Name', 'Indel', 'countEvents', 'Size']]


    grouped = data.groupby('Sample_Name')['Size'].apply(list).to_dict()
    grouped_res = {}
    # create deletion dicts
    for k, v in grouped.items():
      res = {}
      for i in range(1, 31):
        res[i] = v.count(i)
      grouped_res[k] = res

    # Normalize Counts
    total_count = sum(exp_data['countEvents'])
    exp_data['countEvents'] = exp_data['countEvents'].div(total_count)

    freqs.append(exp_data['countEvents'])
    i += 1
  return exps #, MH_freq, MH_less_freq


# Load data
input_dir = os.path.dirname(os.path.dirname(__file__)) + '/in/'
counts, del_features = read_data(input_dir + 'dataset.pkl')       # mESC
mESC_merged_data = pd.concat([counts, del_features], axis=1)
mESC_merged_data = mESC_merged_data.reset_index()

# counts, del_features = read_data(input_dir + 'U2OS.pkl')        # U2OS

# Isolate MH-based and MH-less deletions
mESC_deletions = mESC_merged_data[mESC_merged_data['Type'] == 'DELETION']
#mESC_exps, mESC_MH_freq, mESC_MH_less_freq = get_del_type_freqs(mESC_deletions)


# genotype frequency distribution



#MHless_deletions = MHless_deletions.drop(columns=['index', 'homologyLength', 'homologyGCContent'])
#[exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_data(MH_deletions)
# unique del characterised by:
#     delta, del length/Size
#     homologyGCContent
#     homologyLength

# MH-less deletions characterised by:
#     delta, del length/Size

# frequency given by:
#     countEvents
#     fraction

