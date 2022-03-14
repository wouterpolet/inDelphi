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
  exps = deletions['Sample_Name'].unique()

#  microhomologies = deletions[deletions['homologyLength'] != 0]
  # mh_less = deletions[deletions['homologyLength'] == 0]
  mh_lens, gc_fracs, del_lens, freqs, dl_freqs = [], [], [], [], []
  for id, exp in enumerate(exps):
    # Microhomology computation
    mh_exp_data = gRNA_df[gRNA_df['Sample_Name'] == exp][
      ['Indel', 'countEvents', 'fraction', 'Size', 'homologyLength', 'homologyGCContent']]

    # Normalize Counts
    total_count = sum(mh_exp_data['countEvents'])
    mh_exp_data['countEvents'] = mh_exp_data['countEvents'].div(total_count)

    freqs.append(mh_exp_data['countEvents'])
    mh_lens.append(mh_exp_data['homologyLength'].astype('int32'))
    gc_fracs.append(mh_exp_data['homologyGCContent'])
    del_lens.append(mh_exp_data['Size'].astype('int32'))

    curr_dl_freqs = []
    dl_freq_df = mh_exp_data[mh_exp_data['Size'] <= 28]
    for del_len in range(1, 28 + 1):
      dl_freq = sum(dl_freq_df[dl_freq_df['Size'] == del_len]['countEvents'])
      curr_dl_freqs.append(dl_freq)
    dl_freqs.append(curr_dl_freqs)

  return exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs



input_dir = os.path.dirname(os.path.dirname(__file__)) + '/in/'
counts, del_features = read_data(input_dir + 'dataset.pkl')

# counts, del_features = read_data(input_dir + 'U2OS.pkl')
merged_data = pd.concat([counts, del_features], axis=1)
merged_data = merged_data.reset_index()

deletions = merged_data[merged_data['Type'] == 'DELETION']
insertions = merged_data[merged_data['Type'] == 'INSERTION']
insertions = insertions.reset_index()
insertions1_bp = insertions[insertions['Indel'].str.startswith('1+')]
MH_deletions = deletions[deletions['homologyLength'] != 0]
MH_deletions = MH_deletions.reset_index()
#MH_deletions = MH_deletions.drop(columns=['index'])

MHless_deletions = deletions[deletions['homologyLength'] == 0]
MHless_deletions = MHless_deletions.reset_index()
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

