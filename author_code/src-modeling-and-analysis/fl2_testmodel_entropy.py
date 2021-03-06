from __future__ import division
import _config, _lib, _data, _predict2
import sys, os

sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from author_code.mylib import util
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
from scipy.stats import entropy

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Going wide: experiments to analyze
##
exps = ['Lib1-mES-controladj', 
        'Lib1-HEK293T-controladj', 
        'Lib1-HCT116-controladj', 
        'DisLib-mES-controladj', 
        'DisLib-HEK293T', 
        'DisLib-U2OS-controladj', 
        '0226-PRLmESC-Lib1-Cas9',
        '0226-PRLmESC-Dislib-Cas9',
        'VO-spacers-HEK293-48h-controladj', 
        'VO-spacers-HCT116-48h-controladj', 
        'VO-spacers-K562-48h-controladj',
        '0105-mESC-Lib1-Cas9-Tol2-BioRep2-r1-controladj',
        '0105-mESC-Lib1-Cas9-Tol2-BioRep3-r1-controladj',
        ]

##
# Run statistics
##
def calc_statistics(orig_df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df = _lib.mh_del_subset(orig_df)
  df = _lib.indels_without_mismatches_subset(df)
  if sum(df['Count']) <= 1000:
    return
  df['Frequency'] = _lib.normalize_frequency(df)

  _predict2.init_model()

  seq, cutsite = _lib.get_sequence_cutsite(df)
  pred_df = _predict2.predict_mhdel(seq, cutsite)

  join_cols = ['Category', 'Genotype Position', 'Length']
  mdf = df.merge(pred_df, how = 'outer', on = join_cols)
  mdf['Frequency'].fillna(value = 0, inplace = True)
  mdf['Predicted_Frequency'].fillna(value = 0, inplace = True)
  obs = mdf['Frequency']
  pred = mdf['Predicted_Frequency']

  obs_entropy = entropy(obs) / np.log(len(obs))
  pred_entropy = entropy(pred) / np.log(len(pred))
  alldf_dict['obs gt entropy'].append(obs_entropy)
  alldf_dict['pred gt entropy'].append(pred_entropy)

  df = orig_df[orig_df['Category'] == 'del']
  df = df[df['Length'] <= 28]
  df['Frequency'] = _lib.normalize_frequency(df)
  obs_dl = []
  for del_len in range(1, 28+1):
    freq = sum(df[df['Length'] == del_len]['Frequency'])
    obs_dl.append(freq)
  pred_dl = _predict2.deletion_length_distribution(seq, cutsite)

  obs_entropy = entropy(obs_dl) / np.log(len(obs_dl))
  pred_entropy = entropy(pred_dl) / np.log(len(pred_dl))
  alldf_dict['obs dl entropy'].append(obs_entropy)
  alldf_dict['pred dl entropy'].append(pred_entropy)

  alldf_dict['_Experiment'].append(exp)

  return alldf_dict

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  if 'Lib1' in data_nm or 'VO' in data_nm:
    dataset = _data.load_dataset(data_nm, exp_subset ='vo_spacers', exp_subset_col ='Designed Name')
  if 'DisLib' in data_nm:
    dataset = _data.load_dataset(data_nm, exp_subset ='clin', exp_subset_col ='Designed Name')
    # Remove data with iterated editing
    dlwt = _config.d.DISLIB_WT
    for idx, row in dlwt.iterrows():
      if row['wt_repairable'] == 'iterwt':
        del dataset[row['name']]
  if dataset is None:
    return

  timer = util.Timer(total = len(dataset))
  # for exp in dataset.keys()[:100]:
  for exp in dataset.keys():
    df = dataset[exp]
    calc_statistics(df, exp, alldf_dict)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
  return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm):
  print data_nm
  stats_csv_fn = out_dir + '%s.csv' % (data_nm)
  if not os.path.isfile(stats_csv_fn) or redo:
    print 'Running statistics from scratch...'
    stats_csv = prepare_statistics(data_nm)
    stats_csv.to_csv(stats_csv_fn)
  else:
    print 'Getting statistics from file...'
    stats_csv = pd.read_csv(stats_csv_fn, index_col = 0)
  print 'Done'
  return stats_csv

##
# Plotters
##
def plot():
  # Frequency of deletions by length and MH basis.

  return


##
# qsubs
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating qsub scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for exp in exps:
    command = '/cluster/mshen/env/anaconda2/bin/python %s.py %s redo' % (NAME, exp)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, exp)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -m e -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir)
  return

##
# nohups
##
def gen_nohups():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating nohup scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  nh_commands = []

  num_scripts = 0
  for exp in exps:
    script_id = NAME.split('_')[0]
    command = 'nohup python -u %s.py %s redo > nh_%s_%s.out &' % (NAME, exp, script_id, exp)
    nh_commands.append(command)

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(nh_commands))

  return


##
# Main
##
@util.time_dec
def main(data_nm = '', redo_flag = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  if redo_flag == 'redo':
    global redo
    redo = True

  if data_nm == '':
    gen_nohups()
    return

  if data_nm == 'plot':
    plot()

  else:
    load_statistics(data_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  elif len(sys.argv) == 3:
    main(data_nm = sys.argv[1], redo_flag = sys.argv[2])
  else:
    main()
