import copy
import helper
import pandas as pd
import autograd.numpy as np
from scipy.stats import entropy
from collections import defaultdict

DELETION_LEN_LIMIT = 28


# TODO fix / optimize
def get_indel_len_pred(pred_all_df, del_len_limit=DELETION_LEN_LIMIT):
  indel_len_pred = dict()

  # 1 bp insertions
  crit = (pred_all_df['Category'] == 'ins')                                 # for all insertions
  indel_len_pred[1] = float(sum(pred_all_df[crit]['Predicted_Frequency']))  # predicted frequency of 1bp ins over all indel products
                                                                            # store for +1 key in dictionary
  # Deletions
  # Deletion length set to 28 instead of 60 i.e. original-> 55/2
  for del_len in range(1, del_len_limit):
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Length'] == del_len)    # for each deletion length
    freq = float(sum(pred_all_df[crit]['Predicted_Frequency']))                       #   get pred freq of del with that len over all indel products
    dl_key = -1 * del_len                                                             #   give -dl key in dict
    indel_len_pred[dl_key] = freq                                                     #   store as -dl key in dict

                                                                            # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}

  # Frameshifts, insertion-orientation
  fs = {'+0': 0, '+1': 0, '+2': 0}
  for indel_len in indel_len_pred:              # for each predicted frequency of +1, -1, ..., -60
    fs_key = '+%s' % (indel_len % 3)            #   calculate the resulting frameshift +0, +1 or +2 by remainder division
    fs[fs_key] += indel_len_pred[indel_len]     #   and accumulate the predicted frequency of frame shifts
  return indel_len_pred, fs                     # return dict: {+1 = [..], -1 = [..], ..., -60 = [..]} and fs = {'+0': [..], '+1': [..], '+2': [..]}


#                              0123456789012345678901234567890123456789012345678901234
# MH 1 outcome:                GTGCTCTTAACTTTCACTTTATA------TAGGGTTAATAAATGGGAATTTATAT, gt pos 2, del len 6
# MH 2 outcome:                GTGCTCTTAACTTTCACTTTATAGA------GGGTTAATAAATGGGAATTTATAT, gt pos 4, del len 6
# cutsite 27:                  GTGCTCTTAACTTTCACTTTATAGATT
#                                                         TATAGGGTTAATAAATGGGAATTTATAT (this is not reverse strand)
# print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT) TATCTAAATATCCCAATTATTTACCCTTAAATATA
# Deletion limit set to x, since sequence length is 55 -> 55/2 = 27
# for each gRNA sequence, e.g. GTGCTCTTAACTTTCACTTTATAGATTTATAGGGTTAATAAATGGGAATTTATAT
def featurize(seq, cutsite, del_len_limit=DELETION_LEN_LIMIT):
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(1, del_len_limit):
    left = seq[cutsite - del_len: cutsite]  # get 3' overhang nucleotides on the left            TAGATT
    right = seq[
            cutsite: cutsite + del_len]  # and 5' overhang on the right of cutsite                  TATAGG (used to model the 3' overhang)
    # complementary 3' overhang:     ATATCC

    # e.g. del lengh = 6, mhs = [[0, 1, 2], [3, 4], [5], [6]]
    mhs = helper.find_microhomologies(left, right)
    for mh in mhs:  # len      3        2      1     1
      mh_len = len(mh) - 1  # len-1    2        1      0     0
      if mh_len > 0:  # i.e. if true MH
        gtpos = max(mh)  # for MH1, genotype position = 2, for MH2, genotype position = 4
        gt_poss.append(gtpos)

        s = cutsite - del_len + gtpos - mh_len  # 27 - 6 + 2 - 2 = 21, cutsite is b/w 27 and 28 (python 26 and 27), cutsite labelled at 27 on python
        e = s + mh_len  # 21 + 2 = 23
        mh_seq = seq[s: e]  # seq[21:23] = TA
        gc_frac = helper.get_gc_frac(mh_seq)

        mh_lens.append(mh_len)  # 2
        gc_fracs.append(gc_frac)  # 0%
        del_lens.append(del_len)  # 6

  # all MHs for each resection length, their gc fractions, deltas and deletion lengths
  #      90x1     90x1      90x1     90x1 lists
  return mh_lens, gc_fracs, gt_poss, del_lens


# TODO fix / optimize
def predict_all(seq, cutsite):
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
  pred_input = np.array([mh_len, gc_frac]).T  # input to MH-NN
  del_lens = np.array(del_len).T  # input to MH-less NN

  # Predict
  mh_scores = helper.nn_match_score_function(nn_params, pred_input)  # nn_params are the trained MH-NN params
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  # unnormalised MH-NN phi for each MH (each of which corresponds to a unique genotype)
  unfq = np.exp(mh_scores - 0.25 * Js)

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = helper.nn_match_score_function(nn2_params, np.array(dl))  # trained nn2_params
      mhless_score = np.exp(mhless_score - 0.25 * dl)
      mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
      mhfull_contribution = mhfull_contribution + mask
  mhfull_contribution = mhfull_contribution.reshape(-1, 1)
  unfq = unfq + mhfull_contribution  # unnormalised MH deletion genotype freq distribution

  # Store predictions to combine with mh-less deletion predictions
  pred_del_len = copy.copy(del_len)  # prediction deletion lenghts
  pred_gt_pos = copy.copy(gt_pos)  # prediction deltas these 2 together correspond to a unique genotype

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH-less deletions
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)  # same results as previously

  unfq = list(unfq)  # unnormalised MH deletion genotype freq distribution

  pred_mhless_d = defaultdict(list)
  # Include MH-less contributions at non-full MH deletion lengths
  nonfull_dls = []
  for dl in range(1, 60):
    if dl not in del_len:  # for a deletion length that a MH-based deletion doesn't correspond to
      nonfull_dls.append(dl)
    elif del_len.count(dl) == 1:  # for a deletion length that occurs once for a MH-based deletion...
      idx = del_len.index(dl)
      if mh_len[idx] != dl:  # and is not a full-MH (MH-length = deletion length)
        nonfull_dls.append(dl)
    else:  # e.g. if delebution length occurs but occurs more than once?
      nonfull_dls.append(dl)

  mh_vector = np.array(mh_len)
  for dl in nonfull_dls:  # for each deletion length 1- 60 unaccounted for by MH-NN predictions
    mhless_score = helper.nn_match_score_function(nn2_params,
                                                  np.array(dl))  # nn2_params are the trained MH-less NN parameters
    mhless_score = np.exp(mhless_score - 0.25 * dl)  # get its the MH-less phi

    # unnormalised scores for MH-based deletion genotypes + unnormalised scores for each unacccounted for MH-less based genotype
    unfq.append(mhless_score)
    pred_gt_pos.append('e')  # gtpos = delta, but delta position = e?
    pred_del_len.append(dl)  # deletion length

  unfq = np.array(unfq)
  total_phi_score = float(sum(unfq))

  nfq = np.divide(unfq, np.sum(unfq))  # normalised scores for MH-based and MH-less based deletion genotypes
  pred_freq = list(nfq.flatten())  # convert into 1D: number of all deletion genotypes x 1 list

  d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
  pred_del_df = pd.DataFrame(d)
  pred_del_df['Category'] = 'del'  # dataframe of all predicted deletion products:
  # 'Length'                predicted deletion length
  # 'Genotype Position'     predicted delta
  # 'Predicted_Frequency'   predicted normalised frequency
  # 'Category'              deletion

  ################################################################
  #####
  ##### Predict Insertions
  #####
  # Predict 1 bp insertions
  del_score = total_phi_score  # <- input to k-nn
  dlpred = []
  for dl in range(1, 28 + 1):  # for each deletion length 1:28
    crit = (pred_del_df['Length'] == dl)  # select the predicted dels with that del length
    dlpred.append(
      sum(pred_del_df[crit]['Predicted_Frequency']))  # store the predicted freq of all dels with that length
  dlpred = np.array(dlpred) / sum(dlpred)  # normalised frequency distribution of deletion lengths
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))  # precision score of ^ <- input to k-nn

  # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
  fiveohmapper = {'A': [0, 0], 'C': [0, 0],  # no difference between A and C
                  'G': [1, 0], 'T': [0, 1]}
  threeohmapper = {'A': [1, 0], 'C': [0, 0],  # no difference between C and T
                   'G': [0, 1], 'T': [0, 0]}
  fivebase = seq[cutsite - 1]  # the -4 base, e.g. T
  threebase = seq[cutsite]  # the -3 base
  onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [norm_entropy] + [del_score]  # all inputs to knn
  for idx in range(len(onebp_features)):  # for each G, T, A, G, norm-entropy, del-scoer
    val = onebp_features[idx]
    onebp_features[idx] = (val - normalizer[idx][0]) / normalizer[idx][1]  # normalise acc. to set normaliser
  onebp_features = np.array(onebp_features).reshape(1, -1)
  rate_1bpins = float(rate_model.predict(onebp_features))  # based on 1bp features of this sequence context, predict
  #   the fraction frequency of 1bp ins over all ins and dels
  # Predict 1 bp genotype frequencies
  pred_1bpins_d = defaultdict(list)  # empty dict
  # structure of bp_model in e5 line 107     e.g. -4 base = T, bp_model[T] retuns e5 line 112
  for ins_base in bp_model[fivebase]:
    # for each base {A,C,G,T,} when -4 base is T:
    freq = bp_model[fivebase][ins_base]  # e.g. freq = avg. freq of A when -4 base is T
    freq *= rate_1bpins / (
          1 - rate_1bpins)  # e.g. freq of ins_base A =  ratio between fraction frequency of A as 1bp ins when -4 base is T and the fraction frequency of all deletions
    # the division by denominator is required to normalise properly at the last line before return
    pred_1bpins_d['Category'].append('ins')
    pred_1bpins_d['Length'].append(1)
    pred_1bpins_d['Inserted Bases'].append(ins_base)  # store 'A'
    pred_1bpins_d['Predicted_Frequency'].append(freq)  # and freq of 'A' when -4 base is T

  pred_1bpins_df = pd.DataFrame(pred_1bpins_d)  # dict -> df
  pred_all_df = pred_del_df.append(pred_1bpins_df,
                                   ignore_index=True)  # to dataframe of all unique predicted deletion products, append unique insertion products and rename
  pred_all_df['Predicted_Frequency'] /= sum(pred_all_df[
                                              'Predicted_Frequency'])  # normalised frequency of all unique indel products for given sequence and cutsite

  return pred_del_df, pred_all_df, total_phi_score, rate_1bpins  # predicted: df of uniq pred'd del products, df of all uniq pred in+del products, total NN1+2 phi score, fraction freq of 1bp ins over all indels


def predict_sequence_outcome(gene_data):
  d = defaultdict(list)
  size = len(gene_data)
  for index, row in gene_data.iterrows():
    if index % 100000 == 0:
      print(f'Predicted {index} out of {size} sequences')

    seq = row['target']
    chromosome = row['Chromosome']
    location = row['Location']
    orientation = row['Orientation']

    local_cutsite = 30
    grna = seq[13:33]
    # cutsite_coord = start + idx
    # unique_id = '%s_%s_hg38_%s_%s_%s' % (gene_kgid, grna, chrom, cutsite_coord, orientation)

    # the SpCas9 gRNAs targeting exons and introns
    d['Sequence Context'].append(seq)
    d['Local Cutsite'].append(local_cutsite)
    # d['Chromosome'].append(chrom)
    # d['Cutsite Location'].append(cutsite_coord)
    d['Orientation'].append(orientation)
    d['Cas9 gRNA'].append(grna)
    # d['Gene kgID'].append(gene_kgid)
    # d['Unique ID'].append(unique_id)

    # Make predictions for each SpCas9 gRNA targeting exons and introns
    ans = predict_all(seq, local_cutsite)  # trained k-nn, bp summary dict, normalizer
    pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans  #
    # predict all receives seq_context = the gRNA sequence and local_cutsite = the -3 base index
    # pred_del_df = df of predicted unique del products             for sequence context and cutsite
    # pred_all_df = df of all predicted unique in+del products          ^
    # total_phi_score = total NN1+2 phi score                           ^
    # ins_del_ratio = fraction frequency of 1bp ins over all indels     ^

    # pred_all_df ( pred_del_df only has the first 4 columns, and only with info for dels):
    #   'Length'                predicted in/del length
    #   'Genotype Position'     predicted delta (useful only for dels)
    #   'Predicted_Frequency'   predicted normalised in/del frequency
    #   'Category'              deletion/insertion
    #   'Inserted Bases'        predicted inserted base (useful only for ins)

    # Save predictions
    # del_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'dels', shuffled_nm)
    # pred_del_df.to_csv(del_df_out_fn)
    # all_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'all', shuffled_nm)
    # pred_all_df.to_csv(all_df_out_fn)

    ## Translate predictions to indel length frequencies
    indel_len_pred, fs = get_indel_len_pred(pred_all_df, 60)  # normalised frequency distributon on indel lengths
    # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}
    #   and normalised frequency distribution of frameshifts
    #   fs = {'+0': [..], '+1': [..], '+2': [..]}
    # d = zip[dd, dd_shuffled] of dictionary of lists for sequence and shuffled sequence
    # Keys:
    # 'Sequence Context'
    # 'Local Cutsite'
    # 'Chromosome'
    # 'Cutsite Location'
    # 'Orientation'
    # 'Cas9 gRNA'
    # 'Gene kgID'
    # 'Unique ID'
    # 'Total Phi Score'
    # '1ins/del Ratio'
    # '1ins Rate Model'
    # '1ins bp Model'
    # '1ins normalizer'
    # 'Frameshift +0'     normalised frequency distribution of frameshift +0 (i.e. not a fs)
    # 'Frameshift +1'     normalised frequency distribution of frameshift +1
    # 'Frameshift +2'     normalised frequency distribution of frameshift +2
    # 'Frameshift'        normalised frequency distribution of frameshifts (due to +1 and +2)
    # 'Precision - Del Genotype'  precision of freq distrib for MH-based deletion genotypes
    # 'Precision - Del Length'    precision of freq distrib for del lengths 1:60
    # 'Precision - All Genotype'  precision of freq distrib for MH-based del and 1-bp ins genotypes
    # '-4 nt'
    # '-3 nt'
    # 'Highest Ins Rate'    pred freq for the most freq 1bp ins genotype
    # 'Highest Del Rate'    pred freq for most freq MH-based del genotype

    #
    # Store prediction statistics
    #
    d['Total Phi Score'].append(total_phi_score)
    d['1ins/del Ratio'].append(ins_del_ratio)

    d['1ins Rate Model'].append(rate_model)
    d['1ins bp Model'].append(bp_model)
    d['1ins normalizer'].append(normalizer)

    d['Frameshift +0'].append(fs['+0'])
    d['Frameshift +1'].append(fs['+1'])
    d['Frameshift +2'].append(fs['+2'])
    d['Frameshift'].append(fs['+1'] + fs['+2'])

    crit = (pred_del_df['Genotype Position'] != 'e')  # get only MH-based deletion genotypes
    s = pred_del_df[crit]['Predicted_Frequency']
    s = np.array(s) / sum(s)  # renormalised freq distrib of only MH-based deletion genotypes
    del_gt_precision = 1 - entropy(s) / np.log(len(s))
    d['Precision - Del Genotype'].append(del_gt_precision)

    dls = []
    for del_len in range(1, 60):
      dlkey = -1 * del_len
      dls.append(indel_len_pred[dlkey])
    dls = np.array(dls) / sum(dls)  # renormalised freq distrib of del lengths
    del_len_precision = 1 - entropy(dls) / np.log(len(dls))
    d['Precision - Del Length'].append(del_len_precision)

    crit = (pred_all_df['Genotype Position'] != 'e')  # i.e. get only MH-based deletion and 1-bp ins genotypes
    s = pred_all_df[crit]['Predicted_Frequency']
    s = np.array(s) / sum(s)  # renormalised freq distrib of MH dels and 1-bp ins
    all_gt_precision = 1 - entropy(s) / np.log(len(s))
    d['Precision - All Genotype'].append(all_gt_precision)

    negthree_nt = seq[local_cutsite]  # local_cutsite = 30. I think -1 gives the -4 nt....?
    negfour_nt = seq[local_cutsite - 1]
    d['-4 nt'].append(negfour_nt)
    d['-3 nt'].append(negthree_nt)

    crit = (pred_all_df['Category'] == 'ins')
    highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for the most freq 1bp ins genotype
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
    highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for most freq MH-based del genotype
    d['Highest Ins Rate'].append(highest_ins_rate)
    d['Highest Del Rate'].append(highest_del_rate)
  return pd.DataFrame(d)


def bulk_predict(seq, d):
  # Input: A specific sequence
  # Find all Cas9 cutsites, gather metadata, and run inDelphi
  total_len = len(seq)
  all_cutsites = []
  '''
  # for idx in range(len(sequence)):  # for each base in the sequence
    # this loop finishes only each of 5% of all found cutsites with 60-bp long sequences containing only ACGT
  seq = ''
  if sequence[idx: idx + 2] == 'CC':  # if on top strand find CC
    cutsite = idx + 6  # cut site of complementary GG is +6 away
    seq = sequence[cutsite - 30: cutsite + 30]  # get sequence 30bp L and R of cutsite
    seq = reverse_complement(seq)  # compute reverse strand (complimentary) to target with gRNA
    orientation = '-'
  if sequence[idx: idx + 2] == 'GG':  # if GG on top strand
    cutsite = idx - 4  # cut site is -4 away
    seq = sequence[cutsite - 30: cutsite + 30]  # get seq 30bp L and R of cutsite
    orientation = '+'
  if seq == '':
    continue
  if len(seq) != 60:
    continue

  # Sanitize input
  seq = seq.upper()
  if 'N' in seq:  # if N in collected sequence, return to start of for loop / skip rest
    continue
  if not re.match('^[ACGT]*$', seq):  # if there not only ACGT in seq, ^
    continue

  # Store found cutsite
  # if True:
  all_cutsites.append(seq)

  # Randomly query subset for broad shallow coverage
  r = random.random()
  if (idx - 1) % 1000000 == 0 and idx > 1:
    print('%s index out of %s: %s' % (idx - 1, total_len, datetime.datetime.now()))

  if r > (1/350):
    continue  # randomly decide if will predict on the found cutsite or not. 5% of time will
  '''

  # for one set of sequence context and its shuffled counterpart
  #
  # Store metadata statistics
  #
  local_cutsite = 30
  grna = seq[13:33]
  # cutsite_coord = start + idx
  # unique_id = '%s_%s_hg38_%s_%s_%s' % (gene_kgid, grna, chrom, cutsite_coord, orientation)

  # the SpCas9 gRNAs targeting exons and introns
  # d['Sequence Context'].append(seq)
  # d['Local Cutsite'].append(local_cutsite)
  # d['Chromosome'].append(chrom)
  # d['Cutsite Location'].append(cutsite_coord)
  # d['Orientation'].append(orientation)
  # d['Cas9 gRNA'].append(grna)
  # d['Gene kgID'].append(gene_kgid)
  # d['Unique ID'].append(unique_id)

  # Make predictions for each SpCas9 gRNA targeting exons and introns
  ans = predict_all(seq, local_cutsite)  # trained k-nn, bp summary dict, normalizer
  pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans  #
  # predict all receives seq_context = the gRNA sequence and local_cutsite = the -3 base index
  # pred_del_df = df of predicted unique del products             for sequence context and cutsite
  # pred_all_df = df of all predicted unique in+del products          ^
  # total_phi_score = total NN1+2 phi score                           ^
  # ins_del_ratio = fraction frequency of 1bp ins over all indels     ^

  # pred_all_df ( pred_del_df only has the first 4 columns, and only with info for dels):
  #   'Length'                predicted in/del length
  #   'Genotype Position'     predicted delta (useful only for dels)
  #   'Predicted_Frequency'   predicted normalised in/del frequency
  #   'Category'              deletion/insertion
  #   'Inserted Bases'        predicted inserted base (useful only for ins)

  # Save predictions
  # del_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'dels', shuffled_nm)
  # pred_del_df.to_csv(del_df_out_fn)
  # all_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'all', shuffled_nm)
  # pred_all_df.to_csv(all_df_out_fn)

  ## Translate predictions to indel length frequencies
  indel_len_pred, fs = get_indel_len_pred(pred_all_df, 30 + 1)  # normalised frequency distributon on indel lengths TODO: extract
  # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}
  #   and normalised frequency distribution of frameshifts
  #   fs = {'+0': [..], '+1': [..], '+2': [..]}
  # d = zip[dd, dd_shuffled] of dictionary of lists for sequence and shuffled sequence
  # Keys:
  # 'Sequence Context'
  # 'Local Cutsite'
  # 'Chromosome'
  # 'Cutsite Location'
  # 'Orientation'
  # 'Cas9 gRNA'
  # 'Gene kgID'
  # 'Unique ID'
  # 'Total Phi Score'
  # '1ins/del Ratio'
  # '1ins Rate Model'
  # '1ins bp Model'
  # '1ins normalizer'
  # 'Frameshift +0'     normalised frequency distribution of frameshift +0 (i.e. not a fs)
  # 'Frameshift +1'     normalised frequency distribution of frameshift +1
  # 'Frameshift +2'     normalised frequency distribution of frameshift +2
  # 'Frameshift'        normalised frequency distribution of frameshifts (due to +1 and +2)
  # 'Precision - Del Genotype'  precision of freq distrib for MH-based deletion genotypes
  # 'Precision - Del Length'    precision of freq distrib for del lengths 1:60
  # 'Precision - All Genotype'  precision of freq distrib for MH-based del and 1-bp ins genotypes
  # '-4 nt'
  # '-3 nt'
  # 'Highest Ins Rate'    pred freq for the most freq 1bp ins genotype
  # 'Highest Del Rate'    pred freq for most freq MH-based del genotype

  #
  # Store prediction statistics
  #
  # d['Total Phi Score'].append(total_phi_score)
  # d['1ins/del Ratio'].append(ins_del_ratio)

  # d['1ins Rate Model'].append(rate_model)
  # d['1ins bp Model'].append(bp_model)
  # d['1ins normalizer'].append(normalizer)

  # d['Frameshift +0'].append(fs['+0'])
  # d['Frameshift +1'].append(fs['+1'])
  # d['Frameshift +2'].append(fs['+2'])
  # d['Frameshift'].append(fs['+1'] + fs['+2'])

  crit = (pred_del_df['Genotype Position'] != 'e')  # get only MH-based deletion genotypes
  s = pred_del_df[crit]['Predicted_Frequency']
  s = np.array(s) / sum(s)  # renormalised freq distrib of only MH-based deletion genotypes
  del_gt_precision = 1 - entropy(s) / np.log(len(s))
  # d['Precision - Del Genotype'].append(del_gt_precision)

  dls = []
  for del_len in range(1, 30 + 1):
    dlkey = -1 * del_len
    dls.append(indel_len_pred[dlkey])
  dls = np.array(dls) / sum(dls)  # renormalised freq distrib of del lengths
  del_len_precision = 1 - entropy(dls) / np.log(len(dls))
  # d['Precision - Del Length'].append(del_len_precision)

  crit = (pred_all_df['Genotype Position'] != 'e')  # i.e. get only MH-based deletion and 1-bp ins genotypes
  s = pred_all_df[crit]['Predicted_Frequency']
  s = np.array(s) / sum(s)  # renormalised freq distrib of MH dels and 1-bp ins TODO: extract
  all_gt_precision = 1 - entropy(s) / np.log(len(s))
  # d['Precision - All Genotype'].append(all_gt_precision)

  negthree_nt = seq[local_cutsite]  # local_cutsite = 30. I think -1 gives the -4 nt....?
  negfour_nt = seq[local_cutsite - 1]
  # d['-4 nt'].append(negfour_nt)
  # d['-3 nt'].append(negthree_nt)

  crit = (pred_all_df['Category'] == 'ins')
  highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for the most freq 1bp ins genotype
  crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
  highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for most freq MH-based del genotype
  # d['Highest Ins Rate'].append(highest_ins_rate)
  # d['Highest Del Rate'].append(highest_del_rate)
  return (seq, indel_len_pred, s)


def bulk_predict_all(lib_df):
  res = []
  size = len(lib_df)

  for i, seq in enumerate(lib_df):
    if i % 10 == 0:
      print(f'Predicted {i} out of {size} sequences')
    r = np.random.random()
    res.append(bulk_predict(seq, {}))
  return res


def get_pearson_pred_obs(prediction, observation):
  r_values = []
  pred_normalized_fq = []
  for pred in prediction:                                   # for each held-out gRNA
    current_pred_normalized_fq = []
    for i in range(1, -31, -1):                             #   for indel length +1, -1, -2, ...,-30 (keys)
      if i != 0:
        current_pred_normalized_fq.append(pred[1][i])       #       get freq for key i
    pred_normalized_fq.append(current_pred_normalized_fq)   #   return array of predicted frequencies

  for idx, key in enumerate(observation.keys()):
    # Get prediction of GRNA - TODO Change based on grna item
    # normalized_fq = prediction[prediction['Sample_Name']]
    normalized_fq = []
    for i in range(1, -31, -1):
      if i != 0:
        normalized_fq.append(observation[key][i])

    # For dictionary, get items from 1 to -30 into an array

    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(pred_normalized_fq[idx]) # TODO - check item to pick
    pearson_numerator = np.sum((normalized_fq - x_mean) * (pred_normalized_fq[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean) ** 2))
    pearson_denom_y = np.sqrt(np.sum((pred_normalized_fq[idx] - y_mean) ** 2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    r_values.append(pearson_numerator / pearson_denom)
  return r_values


# TODO optimize
def predict_all_items(all_data, df_out_dir, nn_params, nn2_params, rate_model, bp_model, normalizer):
  dd = defaultdict(list)
  dd_shuffled = defaultdict(list)

  num_flushed = 0
  timer = util.Timer(len(all_data))
  # for i, line in enumerate(all_data):
  # d = pd.DataFrame(columns=[
  #   'Sequence Context' , 'Local Cutsite' , 'Cas9 gRNA' , 'Total Phi Score'
  #   , '1ins/del Ratio' , 'Frameshift +0' , 'Frameshift +1' , 'Frameshift +2' , 'Frameshift'
  #   , 'Precision - Del Genotype', 'Precision - Del Length', 'Precision - All Genotype', '-4 nt', '-3 nt', 'Highest Ins Rate', 'Highest Del Rate'
  # ])
  d = defaultdict(list)

  for i, line in all_data.iterrows():
    header = line['name']
    grna = line['grna']
    sequence = line['target']

    # TODO check what to do
    # This might be a problem - all of grna len = 55
    # if len(sequence) < 60 or len(sequence) > 500000:
    #   continue

    # predict for a single exon/intron
    # bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir)
    # dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)

    # local_cutsite = find where grna starts, add the total len of the grna and subtract 3 (PAM)
    # local_cutsite = sequence.index(grna) + len(grna) - 3
    local_cutsite = 27
    # sequence =

    # seq_context is a tuple/pair? of seq and shuffled_seq
    # trained k-nn, bp summary dict, normalizer
    ans = predict_all(sequence, local_cutsite, rate_model, bp_model, normalizer)
    pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans

    # normalised frequency distributon on indel lengths
    indel_len_pred, fs = get_indel_len_pred(pred_all_df)

    #
    # Store prediction statistics
    #
    d['Sequence Context'].append(sequence)
    d['Local Cutsite'].append(local_cutsite)
    d['Cas9 gRNA'].append(grna)

    d['Total Phi Score'].append(total_phi_score)
    d['1ins/del Ratio'].append(ins_del_ratio)

    # d['1ins Rate Model'].append(rate_model)
    # d['1ins bp Model'].append(bp_model)
    # d['1ins normalizer'].append(normalizer)

    d['Frameshift +0'].append(fs['+0'])
    d['Frameshift +1'].append(fs['+1'])
    d['Frameshift +2'].append(fs['+2'])
    d['Frameshift'].append(fs['+1'] + fs['+2'])

    crit = (pred_del_df['Genotype Position'] != 'e')
    s = pred_del_df[crit]['Predicted_Frequency']
    s = np.array(s) / sum(s)
    del_gt_precision = 1 - entropy(s) / np.log(len(s))

    d['Precision - Del Genotype'].append(del_gt_precision)

    dls = []
    for del_len in range(1, DELETION_LEN_LIMIT):
      dlkey = -1 * del_len
      dls.append(indel_len_pred[dlkey])
    dls = np.array(dls) / sum(dls)
    del_len_precision = 1 - entropy(dls) / np.log(len(dls))

    d['Precision - Del Length'].append(del_len_precision)

    crit = (pred_all_df['Genotype Position'] != 'e')
    s = pred_all_df[crit]['Predicted_Frequency']
    s = np.array(s) / sum(s)
    all_gt_precision = 1 - entropy(s) / np.log(len(s))
    d['Precision - All Genotype'].append(all_gt_precision)

    negthree_nt = sequence[local_cutsite - 1]
    negfour_nt = sequence[local_cutsite]
    d['-4 nt'].append(negfour_nt)
    d['-3 nt'].append(negthree_nt)

    crit = (pred_all_df['Category'] == 'ins')
    highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
    highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])
    d['Highest Ins Rate'].append(highest_ins_rate)
    d['Highest Del Rate'].append(highest_del_rate)

    # d.append({
    #   'Sequence Context': sequence
    #   , 'Local Cutsite': local_cutsite, 'Cas9 gRNA': grna
    #   , 'Total Phi Score': total_phi_score, '1ins/del Ratio': ins_del_ratio
    #   , 'Frameshift +0': fs['+0'], 'Frameshift +1': fs['+1']
    #   , 'Frameshift +2': fs['+2'], 'Frameshift': fs['+1'] + fs['+2']
    #   , 'Precision - Del Genotype': del_gt_precision, 'Precision - Del Length': del_len_precision
    #   , 'Precision - All Genotype': all_gt_precision
    #   , '-4 nt': negfour_nt, '-3 nt': negthree_nt,
    #   "Highest Ins Rate": highest_ins_rate, 'Highest Del Rate': highest_del_rate
    # }, ignore_index=True)

    if (i - 1) % 50 == 0 and i > 1:
      print('%s pct, %s' % (i / 500, datetime.datetime.now()))

    timer.update()
  return pd.DataFrame(d)

  # maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force=True)


def predict_data_outcomes(lib_df, models, in_del):
  global nn_params
  nn_params = models[0]
  global nn2_params
  nn2_params = models[1]
  global rate_model
  rate_model = models[2]
  global bp_model
  bp_model = models[3]
  global normalizer
  normalizer = models[4]

  if in_del:
    return bulk_predict_all(lib_df)
  else:
    # Only selecting a smaller number of samples to predict rather than all cutsites
    subset = lib_df.sample(n=1003524)
    # subset = lib_df.sample(n=50)
    return predict_sequence_outcome(subset)



if __name__ == '__main__':
  print('TODO Set main file so we use trained models rather than recompute')
