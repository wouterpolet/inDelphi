import os
import pickle
import pandas as pd

def _pickle_load(file):
  data = pickle.load(open(file, 'rb'))
  return data

def read_data(file):
  master_data = _pickle_load(file)
  return master_data['counts'], master_data['del_features']

def parse_data(gRNA_df):     # input: MH-based deletions

  # A single value GRNA -Train until 1871
  exps = gRNA_df['Sample_Name'].unique()

  #microhomologies = gRNA_df[gRNA_df['homologyLength'] != 0 && gRNA_df['Indel'].str.startswith('1+')]]
  # mh_less = deletions[deletions['homologyLength'] == 0]
  freqs = []
  i = 0
  for id, exp in enumerate(exps):
    print(i)
    # Microhomology computation
    exp_data = gRNA_df[gRNA_df['Sample_Name'] == exp][
      ['Indel', 'Type', 'countEvents', 'fraction', 'Size', 'homologyLength', 'homologyGCContent']]

    # Normalize Counts
    total_count = sum(exp_data['countEvents'])
    exp_data['countEvents'] = exp_data['countEvents'].div(total_count)

    freqs.append(exp_data['countEvents'])
    i += 1
  return exps, freqs



input_dir = os.path.dirname(os.path.dirname(__file__)) + '/in/'
counts, del_features = read_data(input_dir + 'dataset.pkl')

# counts, del_features = read_data(input_dir + 'U2OS.pkl')
merged_data = pd.concat([counts, del_features], axis=1)
merged_data = merged_data.reset_index()

deletions = merged_data[merged_data['Type'] == 'DELETION']
insertions = merged_data[merged_data['Type'] == 'INSERTION']
MHs = deletions[deletions['homologyLength'] != 0]
insertions = insertions.reset_index()
insertions1_bp = insertions[insertions['Indel'].str.startswith('1+')]
MHs_n_Ins = pd.concat([MHs, insertions1_bp], axis=0)
MHs_n_Ins = MHs_n_Ins.reset_index()
MHs_n_Ins = MHs_n_Ins.drop(columns=['index'])

MH_deletions = deletions[deletions['homologyLength'] != 0]
MH_deletions = MH_deletions.reset_index()
#MH_deletions = MH_deletions.drop(columns=['index'])

MHless_deletions = deletions[deletions['homologyLength'] == 0]
MHless_deletions = MHless_deletions.reset_index()
#MHless_deletions = MHless_deletions.drop(columns=['index', 'homologyLength', 'homologyGCContent'])

# genotype frequency distribution
exps, freqs = parse_data(merged_data)

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

