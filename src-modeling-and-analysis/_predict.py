from __future__ import division
import pickle, imp
import copy
import numpy as np
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy


# global vars
model = None
nn_params = None
nn2_params = None
test_exps = None

##
# Sequence featurization
##
def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)

def find_microhomologies(left, right):            # for a given pair of resected left and right strands, equally long
  start_idx = max(len(right) - len(left), 0)      # TAGATT - TATAGG = 0
  mhs = []
  mh = [start_idx]                                # [0]
  for idx in range(min(len(right), len(left))):   # for each base in the overhangs
    # {--left[idx] == right[idx]--} = {--left[idx] complementary to reverse_right[star_idx+idx]--}
    if left[idx] == right[start_idx + idx]:
                                                  # TAGATT    2 MHs
                                                  # || |
                                                  # ATATCC
                                                  # 012345
                                        # gt pos    123456
                                        # MH 1 del outcome: GTGCTCTTAACTTTCACTTTATATAGGGTTAATAAATGGGAATTTATAT
                                        # MH 2 del outcome: GTGCTCTTAACTTTCACTTTATAGAGGGTTAATAAATGGGAATTTATAT
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx +1]
  mhs.append(mh)
  return mhs                                                #                              1234567890123456789012345678901234567890123456789012345
                                                            #                              0123456789012345678901234567890123456789012345678901234
                                                            # MH 1 outcome:                GTGCTCTTAACTTTCACTTTATA------TAGGGTTAATAAATGGGAATTTATAT, gt pos 2, del len 6
                                                            # MH 2 outcome:                GTGCTCTTAACTTTCACTTTATAGA------GGGTTAATAAATGGGAATTTATAT, gt pos 4, del len 6
def featurize(seq, cutsite, DELLEN_LIMIT = 60):             # for each gRNA sequence, e.g. GTGCTCTTAACTTTCACTTTATAGATTTATAGGGTTAATAAATGGGAATTTATAT
                                                            # cutsite 27:                  GTGCTCTTAACTTTCACTTTATAGATT
                                                            #                                                         TATAGGGTTAATAAATGGGAATTTATAT (this is not reverse strand)
  # print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT)                                                           TATCTAAATATCCCAATTATTTACCCTTAAATATA
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(1, DELLEN_LIMIT):                    # for each deletion length 1:60, e.g. del length 6:
    left = seq[cutsite - del_len : cutsite]                 # get 3' overhang nucleotides on the left            TAGATT
    right = seq[cutsite : cutsite + del_len]                # and 5' overhang on the right of cutsite                  TATAGG (used to model the 3' overhang)
                                                            #                           complementary 3' overhang:     ATATCC

    mhs = find_microhomologies(left, right)                 # e.g. del lengh = 6, mhs = [[0, 1, 2], [3, 4], [5], [6]]
    for mh in mhs:                                          #                       len      3        2      1    1
      mh_len = len(mh) - 1                                  #                       len-1    2        1      0     0
      if mh_len > 0:                                        # i.e. if true MH
        gtpos = max(mh)                                     # for MH1, genotype position = 2, for MH2, genotype position = 4
        gt_poss.append(gtpos)

        s = cutsite - del_len + gtpos - mh_len              # 27 - 6 + 2 - 2 = 21, cutsite is b/w 27 and 28 (python 26 and 27), cutsite labelled at 27 on python
        e = s + mh_len                                      # 21 + 2 = 23
        mh_seq = seq[s : e]                                 # seq[21:23] = TA
        gc_frac = get_gc_frac(mh_seq)

        mh_lens.append(mh_len)                              # 2
        gc_fracs.append(gc_frac)                            # 0%
        del_lens.append(del_len)                            # 6

  return mh_lens, gc_fracs, gt_poss, del_lens               # all MHs for each resection length, their gc fractions, deltas and deletion lengths
  #      90x1     90x1      90x1     90x1 lists

##
# Prediction
##
def predict_all(seq, cutsite, rate_model, bp_model, normalizer):
  # Predict 1 bp insertions and all deletions (MH and MH-less)
  # Most complete "version" of inDelphi
  # Requires rate_model (k-NN) to predict 1 bp insertion rate compared to deletion rate
  # Also requires bp_model to predict 1 bp insertion genotype given -4 nucleotide

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH deletions

  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)
  # for this sequence context and cutsite: return all MHs for each resection length, their gc fractions, deltas and deletion lengths

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T        # input to MH-NN
  del_lens = np.array(del_len).T                    # input to MH-less NN
  
  # Predict
  mh_scores = model.nn_match_score_function(nn_params, pred_input)  # nn_params are the trained MH-NN params
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unfq = np.exp(mh_scores - 0.25*Js)                                # unnormalised MH-NN phi for each MH (each of which corresponds to a unique genotype)

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))    # trained nn2_params
      mhless_score = np.exp(mhless_score - 0.25*dl)
      mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
      mhfull_contribution = mhfull_contribution + mask
  mhfull_contribution = mhfull_contribution.reshape(-1, 1)
  unfq = unfq + mhfull_contribution                                   # unnormalised MH deletion genotype freq distribution

  # Store predictions to combine with mh-less deletion predictions
  pred_del_len = copy.copy(del_len)     # prediction deletion lenghts
  pred_gt_pos = copy.copy(gt_pos)       # prediction deltas             these 2 together correspond to a unique genotype

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH-less deletions
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)      # same results as previously

  unfq = list(unfq)                                               # unnormalised freq distribution of MH deletion genotype

  pred_mhless_d = defaultdict(list)
  # Include MH-less contributions at non-full MH deletion lengths
  nonfull_dls = []
  for dl in range(1, 60):
    if dl not in del_len:         # for a deletion length that a MH-based deletion doesn't correspond to
      nonfull_dls.append(dl)
    elif del_len.count(dl) == 1:  # for a deletion length that occurs once for a MH-based deletion...
      idx = del_len.index(dl)
      if mh_len[idx] != dl:       #     and is not a full-MH (MH-length = deletion length)
        nonfull_dls.append(dl)
    else:                         # e.g. if deletion length occurs but occurs more than once?
        nonfull_dls.append(dl)

  mh_vector = np.array(mh_len)
  for dl in nonfull_dls:          # for each deletion length 1- 60 unaccounted for by MH-NN predictions
    mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))  # nn2_params are the trained MH-less NN parameters
    mhless_score = np.exp(mhless_score - 0.25*dl) # get its the MH-less phi

    unfq.append(mhless_score)                     # unnormalised scores for MH-based deletion genotypes
                                                  #     + unnormalised scores for each unacccounted for MH-less based genotype
    pred_gt_pos.append('e')                       # gtpos = delta, but delta position = e?
    pred_del_len.append(dl)                       # deletion length

  unfq = np.array(unfq)
  total_phi_score = float(sum(unfq))

  nfq = np.divide(unfq, np.sum(unfq))             # normalised scores for MH-based and MH-less based deletion genotypes
  pred_freq = list(nfq.flatten())                 # convert into 1D: number of all deletion genotypes x 1 list

  d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
  pred_del_df = pd.DataFrame(d)
  pred_del_df['Category'] = 'del'                 # dataframe of all predicted deletion products:
                                                  # 'Length'                predicted deletion length
                                                  # 'Genotype Position'     predicted delta
                                                  # 'Predicted_Frequency'   predicted normalised frequency
                                                  # 'Category'              deletion

  ################################################################
  #####
  ##### Predict Insertions
  #####
  # Predict 1 bp insertions
  del_score = total_phi_score                                       # <- input to k-nn
  dlpred = []
  for dl in range(1, 28+1):                                         # for each deletion length 1:28
    crit = (pred_del_df['Length'] == dl)                            #   select the predicted dels with that del length
    dlpred.append(sum(pred_del_df[crit]['Predicted_Frequency']))    #   store the predicted freq of all dels with that length
  dlpred = np.array(dlpred) / sum(dlpred)                           # normalised frequency distribution of deletion lengths
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))              # precision score of ^ <- input to k-nn

  # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
  fiveohmapper = {'A': [0, 0], 
                  'C': [0, 0],      # no difference between A and C
                  'G': [1, 0], 
                  'T': [0, 1]}
  threeohmapper = {'A': [1, 0], 
                   'C': [0, 0],     # no difference between C and T
                   'G': [0, 1], 
                   'T': [0, 0]}
  fivebase = seq[cutsite - 1]       # the -4 base, e.g. T
  threebase = seq[cutsite]          # the -3 base
  onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [norm_entropy] + [del_score]   # all inputs to knn
  for idx in range(len(onebp_features)):        # for each G, T, A, G, norm-entropy, del-scoer
    val = onebp_features[idx]
    onebp_features[idx] = (val - normalizer[idx][0]) / normalizer[idx][1]   # normalise acc. to set normaliser
  onebp_features = np.array(onebp_features).reshape(1, -1)
  rate_1bpins = float(rate_model.predict(onebp_features))     # based on 1bp features of this sequence context, predict
                                                              #   the fraction frequency of 1bp ins over all ins and dels
  # Predict 1 bp genotype frequencies
  pred_1bpins_d = defaultdict(list)                                               # empty dict
  for ins_base in bp_model[fivebase]:   # structure of bp_model in e5 line 107     e.g. -4 base = T, bp_model[T] retuns e5 line 112
                                                                                  # for each base {A,C,G,T,} when -4 base is T:
    freq = bp_model[fivebase][ins_base]                                           #   e.g. freq = avg. freq of A when -4 base is T
    freq *= rate_1bpins / (1 - rate_1bpins)                                       #   e.g. freq of ins_base A =  ratio between fraction frequency of A as 1bp ins when -4 base is T
                                                                                  #                                          and the fraction frequency of all deletions
                                                                                  # the division by denominator is required to normalise properly at the last line before return
    pred_1bpins_d['Category'].append('ins')
    pred_1bpins_d['Length'].append(1)
    pred_1bpins_d['Inserted Bases'].append(ins_base)                              #   store 'A'
    pred_1bpins_d['Predicted_Frequency'].append(freq)                             #   and freq of 'A' when -4 base is T

  pred_1bpins_df = pd.DataFrame(pred_1bpins_d)                                    # dict -> df
  pred_all_df = pred_del_df.append(pred_1bpins_df, ignore_index = True)           # to dataframe of all unique predicted deletion products, append unique insertion products and rename
  pred_all_df['Predicted_Frequency'] /= sum(pred_all_df['Predicted_Frequency'])   # normalised frequency of all unique indel products for given sequence and cutsite

  return pred_del_df, pred_all_df, total_phi_score, rate_1bpins   # predicted: df of uniq pred'd del products, df of all uniq pred in+del products, total NN1+2 phi score, fraction freq of 1bp ins over all indels



##
# Init
##
def init_model(run_iter = 'aax', param_iter = 'aag'):
  # run_iter = 'aav'
  # param_iter = 'aag'
  # run_iter = 'aaw'
  # param_iter = 'aae'
  # run_iter = 'aax'
  # param_iter = 'aag'
  # run_iter = 'aay'
  # param_iter = 'aae'
  global model
  if model != None:
    return

  print('Initializing model %s/%s...' % (run_iter, param_iter))
  
  model_out_dir = './out/d2_model/'

  param_fold = model_out_dir + '%s/parameters/' % (run_iter)
  global nn_params
  global nn2_params
  nn_params = pickle.load(open(param_fold + '%s_nn.pkl' % (param_iter)))    # '/out/d2_model/aax/parameters/aag_nn.pkl'
  nn2_params = pickle.load(open(param_fold + '%s_nn2.pkl' % (param_iter)))  # '/out/d2_model/aax/parameters/aag_nn2.pkl'

  model = imp.load_source('model', model_out_dir + '%s/d2_model-noautograd.py' % (run_iter))
  #                       '       ./out/d2_model/aax/d2_model-noautograd.py'
  
  print('Done')
  return