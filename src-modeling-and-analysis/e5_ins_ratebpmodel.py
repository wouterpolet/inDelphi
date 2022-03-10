from __future__ import division
import _config, _lib, _data, _predict, _predict2
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import random
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Default params
DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'

##
# Functions
##
def convert_oh_string_to_nparray(input):
    input = input.replace('[', '').replace(']', '')
    nums = input.split(' ')
    return np.array([int(s) for s in nums])

def featurize(rate_stats, Y_nm):  # rate_stats for all 5 experiments and Y_nm = 'Ins1bp/Del Ratio'
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']]) # "[0 0 0 1]" -> [0 0 0 1]
    threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)  # vector of precision scores for each gRNA in 5 experiments
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)  # vector of deletion scores for each gRNA in 5 experiments
    print ent.shape, fivebases.shape, del_scores.shape

    Y = np.array(rate_stats[Y_nm])  # Y_nm = 'Ins1bp/Del Ratio'
    print Y_nm
    # fivebase / -4 base most common ins are G and T (fig 2c)
    # threebase / -3 base most common ins aare G and C ? (fig 2c)
    # why do they collet G, T, A, G?
    Normalizer = [(np.mean(fivebases.T[2]),  # 3rd column is G; fraction of G
                      np.std(fivebases.T[2])), # 3rd column; std of G
                  (np.mean(fivebases.T[3]), # 4th column is T; fraction of T
                      np.std(fivebases.T[3])),  # std of T
                  (np.mean(threebases.T[0]), # 1st column is A; fraction of A
                      np.std(threebases.T[0])), # std of A
                  (np.mean(threebases.T[2]), # 3rd column, fraction of G
                      np.std(threebases.T[2])), # std of G
                  (np.mean(ent),
                      np.std(ent)),
                  (np.mean(del_scores),
                      np.std(del_scores)),
                 ]
    # Normalised 5G 5T 3A 3G frequencies for all gRNAs across 5 experiments?
    fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
    fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
    threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
    threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
    gtag = np.array([fiveG, fiveT, threeA, threeG]).T

    ent = (ent - np.mean(ent)) / np.std(ent)  # normalised precision score across 5 experiments?
    del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)  # normalised deletion score across 5 experiments?

    X = np.concatenate(( gtag, ent, del_scores), axis = 1) # concatenate as columns
    X = np.concatenate(( gtag, ent, del_scores), axis = 1)
    feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    print 'Num. samples: %s, num. features: %s' % X.shape  # ? samples, 6 features

    return X, Y, Normalizer

def generate_models(X, Y, bp_stats, Normalizer):
  # Train rate model
  model = KNeighborsRegressor()
  model.fit(X, Y)  # (norm freq of -5G -5T -3A -3G + norm precision score + norm del score) VS freq of 1 bp ins over all indels
  with open(out_dir + 'rate_model_v2.pkl', 'w') as f:
    pickle.dump(model, f)

  # Obtain bp stats
  bp_model = dict()
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']

  # bp_stats:
  # 'Frequency'     the freq of 1bp ins over all Cas9 products /
  # 'A frac'        the freq of A ins over all 1bp ins
  # 'C frac'
  # 'G frac'
  # 'T frac'
  # 'Base'          the -4 base
  # '_Experiment'   the gRNA experiment

  t_melt = pd.melt(bp_stats,                # for each gRNA, its -4 base and the % of nucleotide N as the 1-bp ins
                   id_vars = ['Base'],      # t_melt drawn out (example): https://bit.ly/t_melt
                   value_vars = ins_bases,
                   var_name = 'Ins Base', 
                   value_name = 'Fraction')
  for base in list('ACGT'):             # for each base N
    bp_model[base] = dict()             # create a dict 'N': {..}
    mean_vals = []
    for ins_base in ins_bases:          # for each 'N frac'
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)    # e.g. for all gRNAs w/ -4 b as Ts
      mean_vals.append(float(np.mean(t_melt[crit])))                        #      append avg. freq of ins's as N={A,C,G,T}
    for bp, freq in zip(list('ACGT'), mean_vals):                           #      into [ , , , ]
      bp_model[base][bp] = freq / sum(mean_vals)                            #            A C G T
                                                                            # line 104 normalises the avg. freqs
  with open(out_dir + 'bp_model_v2.pkl', 'w') as f:
    pickle.dump(bp_model, f)                        # -4 base
                                                    #
                                                    # {'A': {'A': .., 'C': .., 'G': .., 'T': ..},
                                                    #  'C': {'A': .., 'C': .., 'G': .., 'T': ..},
                                                    #  'G': {'A': .., 'C': .., 'G': .., 'T': ..},
                                                    #  'T': {'A': .., 'C': .., 'G': .., 'T': ..} }
                                                    # each .. is the norm.'d avg. freq. of that N when -4 base is 'N'
                                                    # this dict represents all gRNAs

  with open(out_dir + 'Normalizer_v2.pkl', 'w') as f:
    pickle.dump(Normalizer, f)
      
  return

##
# Main
##
@util.time_dec
def main(data_nm = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  import fi2_ins_ratio
  import fk_1bpins

  exps = ['VO-spacers-HEK293-48h-controladj', 
          'VO-spacers-K562-48h-controladj',
          'DisLib-mES-controladj',
          'DisLib-U2OS-controladj',
          'Lib1-mES-controladj'
         ]

  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()  
  for exp in exps:
    rate_stats = fi2_ins_ratio.load_statistics(exp) #for all gRNAs in each exp ^: ' Editing Rate' % of Cas9 induced products /
                                                                                # *'Ins1bp/Del Ratio' 1bp in frequency over all indels /
                                                                                # 'Ins1bp/MHDel Ratio' 1bp frequency over ins + MH-based dels /
                                                                                # 'Ins1bp Ratio' 1bp in frequency over all Cas9 induced products /
                                                                                # *'Fivebase' the -4 base (yes, the -4 base) /
                                                                                # *'Del Score' phi total deletion score /
                                                                                # *'Entropy' normalised precision score /
                                                                                # 'GC' % GC in local sequence (4 bases L and R of cutsite) /
                                                                                # *'Fivebase_OH' -4 base one-hot encoded /
                                                                                # *'Threebase' -3 base /
                                                                                # *'Threebase_OH' -3 base one-hot encoded /
                                                                                # '_Experiment' the gRNA
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]       # get gRNAs with precision of del length distribution > 0.01

    bp_stats = fk_1bpins.load_statistics(exp)   # for all gRNAs in each exp ^: 'Frequency' the freq of 1bp ins over all Cas9 products /
                                                                            #  *'A frac' the freq of A ins over all 1bp ins
                                                                            #  *'C frac'
                                                                            #  *'G frac'
                                                                            #  *'T frac'
                                                                            #  *'Base' the -4 base
                                                                            #  '_Experiment' the gRNA experiment

    exps = rate_stats['_Experiment']

    if 'DisLib' in exp:
      crit = (rate_stats['_Experiment'] >= 73) & (rate_stats['_Experiment'] <= 300)     # for gRNAs 73-300
      rs = rate_stats[crit]
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)                   # get their rate stats

      crit = (rate_stats['_Experiment'] >= 16) & (rate_stats['_Experiment'] <= 72)      # for gRNAs 16-72
      rs = rate_stats[crit]
      rs = rs[rs['Ins1bp Ratio'] < 0.3] # remove outliers                               # get only those meeting this condition
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)                   # also get their rate stats

      crit = (bp_stats['_Experiment'] >= 73) & (bp_stats['_Experiment'] <= 300)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)

      crit = (bp_stats['_Experiment'] >= 16) & (bp_stats['_Experiment'] <= 72)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)                       # and get bp stats too

    elif 'VO' in exp or 'Lib1' in exp:
      all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
      all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

    print exp, len(all_rate_stats)

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')      # for all 5 experiments, pass their rate stats
  generate_models(X, Y, all_bp_stats, Normalizer)                       # for all 5 experiments, pass their bp stats

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  else:
    main()
