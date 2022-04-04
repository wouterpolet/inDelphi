# 
from __future__ import division
import _config, _lib
import sys, pickle

sys.path.append('/cluster/mshen/')
from author_code.mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
in_dir = _config.IN_PLACE + '/'


##
# Sequence featurization
##
def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)


def find_microhomologies(left, right):
  start_idx = max(len(right) - len(left), 0)
  mhs = []
  mh = [start_idx]
  for idx in range(min(len(right), len(left))):
    if left[idx] == right[start_idx + idx]:
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx + 1]
  mhs.append(mh)
  return mhs


##
# Main featurizer
##
def featurize(orig_df):
  # seq, cutsite = _lib.get_sequence_cutsite(orig_df)
  # exps = [x.split('_')[-1] for x in orig_df['Sample_Name'].values]

  deletions = orig_df[orig_df['Type'] == 'DELETION']
  deletions = deletions.reset_index()

  seq = []
  cutsite = []
  for id, x in enumerate(deletions['Sample_Name'].values):
    data_split = x.split('_')
    seq.append(data_split[-1])
    cutsite.append(deletions['Indel'][id])

  mh_lens, gc_fracs, del_lens, freqs = [], [], [], []
  dl_freqs = []

  DELLEN_LIMIT = 60

  df = _lib.mh_del_subset_new(orig_df)
  # df = _lib.indels_without_mismatches_subset(df)
  df = df[df['Size'] <= DELLEN_LIMIT]

  if sum(df['Size']) < 1000:
    return None

  criteria = (orig_df['Type'] == 'DELETION') & (orig_df['Size'] <= 28)
  s = orig_df[criteria]
  s['countEvents'] = _lib.normalize_frequency_new(s)
  for del_len in range(1, 28 + 1):
    dl_freq = sum(s[s['Size'] == del_len]['countEvents'])
    dl_freqs.append(dl_freq)

  df['countEvents'] = _lib.normalize_frequency_new(df)

  for del_len in range(1, DELLEN_LIMIT + 1):
    left = seq[cutsite - del_len: cutsite]
    right = seq[cutsite: cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s: e]
        gc_frac = get_gc_frac(mh_seq)

        criteria = (df['Size'] == del_len) & (df['Genotype Position'] == gtpos)
        freq = sum(df[criteria]['countEvents'])

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)
        freqs.append(freq)
  insertions = orig_df[orig_df['Type'] == 'INSERTION']
  insertions = insertions.reset_index()

  return mh_lens, gc_fracs, del_lens, freqs, dl_freqs

def featurize_new(del_features):

  # seq, cutsite = _lib.get_sequence_cutsite(orig_df)
  # exps = [x.split('_')[-1] for x in orig_df['Sample_Name'].values]

  # deletions = del_features[del_features['Type'] == 'DELETION']
  deletions = del_features.reset_index()

  # seq = []
  # cutsite = []
  # for id, x in enumerate(deletions['Sample_Name'].values):
  #   data_split = x.split('_')
  #   seq.append(data_split[-1])
  #   cutsite.append(deletions['Indel'][id])

  mh_lens, gc_fracs, del_lens, freqs = [], [], [], []
  dl_freqs = []

  DELLEN_LIMIT = 60

  df = _lib.mh_del_subset_new(orig_df)
  # df = _lib.indels_without_mismatches_subset(df)
  df = df[df['Size'] <= DELLEN_LIMIT]

  if sum(df['Size']) < 1000:
    return None

  criteria = (orig_df['Type'] == 'DELETION') & (orig_df['Size'] <= 28)
  s = orig_df[criteria]
  s['countEvents'] = _lib.normalize_frequency_new(s)
  for del_len in range(1, 28 + 1):
    dl_freq = sum(s[s['Size'] == del_len]['countEvents'])
    dl_freqs.append(dl_freq)

  df['countEvents'] = _lib.normalize_frequency_new(df)

  for del_len in range(1, DELLEN_LIMIT + 1):
    left = seq[cutsite - del_len: cutsite]
    right = seq[cutsite: cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s: e]
        gc_frac = get_gc_frac(mh_seq)

        criteria = (df['Size'] == del_len) & (df['Genotype Position'] == gtpos)
        freq = sum(df[criteria]['countEvents'])

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)
        freqs.append(freq)
  insertions = orig_df[orig_df['Type'] == 'INSERTION']
  insertions = insertions.reset_index()

  return mh_lens, gc_fracs, del_lens, freqs, dl_freqs


##
# main 
##
def prepare_library_dataset(dataset, featurized_data):
  print('Assumes library data')

  good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs = featurized_data
  timer = util.Timer(total=len(dataset))

  ans = featurize(dataset)

  for exp in dataset.keys():
    df = dataset[exp]
    ans = featurize(df)
    if ans is None:
      continue
    mh_len, gc_frac, del_len, freq, dl_freq = ans
    good_exps.append(exp)
    mh_lengths.append(mh_len)
    gc_fracs.append(gc_frac)
    del_lens.append(del_len)
    freqs.append(freq)
    dl_freqs.append(dl_freq)
    timer.update()

  print('Found %s good exps' % (len(good_exps)))
  return


def init_featurized_data():
  good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs = [], [], [], [], [], []
  all_data = [good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs]
  return all_data


def pickle_featurized_data(featurized_data, nm):
  print('Pickling..')
  with open(out_dir + '%s.pkl' % (nm), 'w') as f:
    pickle.dump(featurized_data, f)
  print('Done')
  return


##
# Dataset
##
def prepare_dataset_libA():
  dataset_nm = 'dataset'
  print('Preparing %s' % (dataset_nm))

  featurized_data = init_featurized_data()

  dataset = pickle.load(open(in_dir + dataset_nm + '.pkl', 'rb'))
  counts = dataset['counts']
  del_features = dataset['del_features']

  merged = pd.concat([counts, del_features], axis=1)
  prepare_library_dataset(merged, featurized_data)

  # Load Lib1 data
  dataset = _data.load_dataset('Lib1-mES-controladj')

  # Remove VO spacers from lib 1
  for vo_spacer_idx in range(1872, 1961 + 1):
    vo_spacer_exp = str(vo_spacer_idx)
    del dataset[vo_spacer_exp]
  # Remove low rep spacers from lib1
  for exp in dataset.keys():
    if int(exp) not in _config.d.HIGHREP_LIB1_EXPS:
      del dataset[exp]

  print(len(dataset))
  prepare_library_dataset(dataset, featurized_data)

  pickle_featurized_data(featurized_data, dataset_nm)
  return


##
# Main
##
@util.time_dec
def main(data_nm=''):
  print(NAME)
  global out_dir
  global in_dir
  util.ensure_dir_exists(out_dir)
  util.ensure_dir_exists(in_dir)
  prepare_dataset_libA()
  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(data_nm=sys.argv[1])
  else:
    main()
