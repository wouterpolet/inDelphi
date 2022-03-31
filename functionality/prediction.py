import copy
import glob

import pandas as pd
import autograd.numpy as np
from scipy.stats import entropy
from collections import defaultdict

import functionality.helper as helper
from functionality.author_helper import Timer, find_microhomologies, get_gc_frac, nn_match_score_function


def get_indel_len_pred(pred_all_df, del_len_limit=61):
  indel_len_pred = dict()

  # predicted frequency of 1bp ins over all indel products
  indel_len_pred[1] = float(sum(pred_all_df[pred_all_df['Category'] == 'ins']['Predicted_Frequency']))

  # Deletions
  # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}
  # Deletion length set to 28 instead of 60 i.e. original-> 55/2
  for del_len in range(1, del_len_limit):
    # get pred freq of del with that len over all indel products
    freq = float(sum(pred_all_df[(pred_all_df['Category'] == 'del') & (pred_all_df['Length'] == del_len)]['Predicted_Frequency']))
    dl_key = -1 * del_len
    indel_len_pred[dl_key] = freq

  # Frameshifts, insertion-orientation
  fs = {'+0': 0, '+1': 0, '+2': 0}
  # for each predicted frequency of +1, -1, ..., -60
  for indel_len in indel_len_pred:
    # calculate the resulting frameshift +0, +1 or +2 by remainder division
    fs_key = '+%s' % (indel_len % 3)
    fs[fs_key] += indel_len_pred[indel_len]
  # return dict: {+1 = [..], -1 = [..], ..., -60 = [..]}     and       fs = {'+0': [..], '+1': [..], '+2': [..]}
  return indel_len_pred, fs


#                              0123456789012345678901234567890123456789012345678901234
# MH 1 outcome:                GTGCTCTTAACTTTCACTTTATA------TAGGGTTAATAAATGGGAATTTATAT, gt pos 2, del len 6
# MH 2 outcome:                GTGCTCTTAACTTTCACTTTATAGA------GGGTTAATAAATGGGAATTTATAT, gt pos 4, del len 6
# cutsite 27:                  GTGCTCTTAACTTTCACTTTATAGATT
#                                                         TATAGGGTTAATAAATGGGAATTTATAT (this is not reverse strand)

# Deletion limit set to x, since sequence length is 55 -> 55/2 = 27
# for each gRNA sequence, e.g. GTGCTCTTAACTTTCACTTTATAGATTTATAGGGTTAATAAATGGGAATTTATAT
def featurize(seq, cutsite, del_len_limit=61):
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(1, del_len_limit):
    # get 3' overhang nucleotides on the left
    left = seq[cutsite - del_len: cutsite]
    # get 5' overhang on the right of cutsite
    right = seq[cutsite: cutsite + del_len]

    # e.g. del lengh = 6, mhs = [[0, 1, 2], [3, 4], [5], [6]]
    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:  # i.e. if true MH
        gtpos = max(mh)  # for MH1, genotype position = 2, for MH2, genotype position = 4
        gt_poss.append(gtpos)

        # 27 - 6 + 2 - 2 = 21, cutsite is b/w 27 and 28 (python 26 and 27), cutsite labelled at 27 on python
        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len  # 21 + 2 = 23
        mh_seq = seq[s: e]
        gc_frac = get_gc_frac(mh_seq)

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)

  # all MHs for each resection length, their gc fractions, deltas and deletion lengths
  #      90x1     90x1      90x1     90x1 lists
  return mh_lens, gc_fracs, gt_poss, del_lens


def get_pearson_pred_obs(prediction, observation, del_len_limit=61):
  r_values = []
  pred_normalized_fq = []
  for index, pred in prediction.iterrows():
    current_pred_normalized_fq = []
    for i in range(1, -del_len_limit, -1):
      if i != 0:
        current_pred_normalized_fq.append(pred['Indel Length Prediction'][i])
    pred_normalized_fq.append(current_pred_normalized_fq)

  for idx, key in enumerate(observation.keys()):
    normalized_fq = []
    for i in range(1, -del_len_limit, -1):
      if i != 0:
        normalized_fq.append(observation[key][i])

    # TODO - not sure if we should use the in-built pearson function? pearsonr
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(pred_normalized_fq[idx]) # TODO - check item to pick
    pearson_numerator = np.sum((normalized_fq - x_mean) * (pred_normalized_fq[idx] - y_mean))
    pearson_denom = np.sqrt(np.sum((normalized_fq - x_mean) ** 2) * np.sum((pred_normalized_fq[idx] - y_mean) ** 2))
    r_values.append(pearson_numerator / pearson_denom)
  return r_values


class Prediction:
  def __init__(self, del_limit, dl_limit, models):
    # TODO check if we need both or just one?
    self.deletion_len_limit = del_limit + 1
    self.dl_limit = dl_limit + 1

    self.nn_params = models['nn']
    self.nn2_params = models['nn_2']
    self.rate_model = models['rate']
    self.bp_model = models['bp']
    self.normalizer = models['norm']

    # TODO maybe filter out unneeded columns prior to returning?
    # self.predictions = self.predict_all_sequence_outcomes(data)

  def predict_sequence(self, seq, cutsite):
    """
    @param self:
    @param seq: Sequence we are predicting
    @param cutsite: lcoation of the cutsite on the sequence
    @return: predicted: df of uniq pred'd del products; df of all uniq pred in+del products; total NN1+2 phi score; fraction freq of 1bp ins over all indels
    """
    # Predict 1 bp insertions and all deletions (MH and MH-less)
    # Requires rate_model (k-NN) to predict 1 bp insertion rate compared to deletion rate
    # Also requires bp_model to predict 1 bp insertion genotype given -4 nucleotide

    ################################################################
    #####
    ##### Predict MH and MH-less deletions
    #####
    # Predict MH deletions

    # for this sequence context and cutsite: return all MHs for each resection length, their gc fractions, deltas and deletion lengths
    mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

    # Form inputs
    pred_input = np.array([mh_len, gc_frac]).T  # input to MH-NN
    del_lens = np.array(del_len).T  # input to MH-less NN

    # Predict
    mh_scores = nn_match_score_function(self.nn_params, pred_input)  # nn_params are the trained MH-NN params
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
        mhless_score = nn_match_score_function(self.nn2_params, np.array(dl))  # trained nn2_params
        mhless_score = np.exp(mhless_score - 0.25 * dl)
        mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
        mhfull_contribution = mhfull_contribution + mask
    # TODO: Check with team this always returns array of 0s
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
    for dl in range(1, self.deletion_len_limit):
      # for a deletion length that a MH-based deletion doesn't correspond to
      if dl not in del_len:
        nonfull_dls.append(dl)
      # for a deletion length that occurs once for a MH-based deletion...
      elif del_len.count(dl) == 1:
        idx = del_len.index(dl)
        # and is not a full-MH (MH-length = deletion length)
        if mh_len[idx] != dl:
          nonfull_dls.append(dl)
      # e.g. if delebution length occurs but occurs more than once?
      else:
        nonfull_dls.append(dl)

    mh_vector = np.array(mh_len)
    # for each deletion length 1- 60 unaccounted for by MH-NN predictions
    for dl in nonfull_dls:
      # nn2_params are the trained MH-less NN parameters
      mhless_score = nn_match_score_function(self.nn2_params, np.array(dl))
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
    for dl in range(1, self.dl_limit):  # for each deletion length 1:28
      crit = (pred_del_df['Length'] == dl)  # select the predicted dels with that del length
      dlpred.append(
        sum(pred_del_df[crit]['Predicted_Frequency']))  # store the predicted freq of all dels with that length
    dlpred = np.array(dlpred) / sum(dlpred)  # normalised frequency distribution of deletion lengths
    norm_entropy = entropy(dlpred) / np.log(len(dlpred))  # precision score of ^ <- input to k-nn

    # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    # no difference between A and C
    # TODO should these change? st they are unique
    fiveohmapper = {'A': [0, 0], 'C': [0, 0], 'G': [1, 0], 'T': [0, 1]}
    # no difference between C and T
    threeohmapper = {'A': [1, 0], 'C': [0, 0], 'G': [0, 1], 'T': [0, 0]}
    # the -4 base, e.g. T
    fivebase = seq[cutsite - 1]
    # the -3 base
    threebase = seq[cutsite]
    # all inputs to knn
    onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [norm_entropy] + [del_score]
    # for each G, T, A, G, norm-entropy, del-score
    for idx in range(len(onebp_features)):
      val = onebp_features[idx]
      onebp_features[idx] = (val - self.normalizer[idx][0]) / self.normalizer[idx][1]
    onebp_features = np.array(onebp_features).reshape(1, -1)
    rate_1bpins = float(self.rate_model.predict(onebp_features))  # based on 1bp features of this sequence context, predict
    #   the fraction frequency of 1bp ins over all ins and dels

    # Predict 1 bp genotype frequencies
    pred_1bpins_d = defaultdict(list)
    for ins_base in self.bp_model[fivebase]:
      # for each base {A,C,G,T,} when -4 base is T:
      freq = self.bp_model[fivebase][ins_base]
      # e.g. freq of ins_base A =  ratio between fraction frequency of A as 1bp ins when -4 base is T and the fraction frequency of all deletions
      freq *= rate_1bpins / (1 - rate_1bpins)
      # the division by denominator is required to normalise properly at the last line before return
      pred_1bpins_d['Category'].append('ins')
      pred_1bpins_d['Length'].append(1)
      pred_1bpins_d['Inserted Bases'].append(ins_base)  # store 'A'
      pred_1bpins_d['Predicted_Frequency'].append(freq)  # and freq of 'A' when -4 base is T

    pred_1bpins_df = pd.DataFrame(pred_1bpins_d)  # dict -> df
    # to dataframe of all unique predicted deletion products, append unique insertion products and rename
    pred_all_df = pred_del_df.append(pred_1bpins_df, ignore_index=True)
    # normalised frequency of all unique indel products for given sequence and cutsite
    pred_all_df['Predicted_Frequency'] /= sum(pred_all_df['Predicted_Frequency'])

    return pred_del_df, pred_all_df, total_phi_score, rate_1bpins

  def predict_all_sequence_outcomes(self, gene_data):
    all_data = defaultdict(list)
    size = len(gene_data)
    timer = Timer(total=size)
    has_orientation = 'Orientation' in gene_data.columns

    for index, row in gene_data.iterrows():
      seq = row['target']
      local_cutsite = 30
      grna = seq[13:33]

      # the SpCas9 gRNAs targeting exons and introns
      all_data['Sequence Context'].append(seq)
      all_data['Local Cutsite'].append(local_cutsite)
      if has_orientation:
        orientation = row['Orientation']
        all_data['Orientation'].append(orientation)
      all_data['Cas9 gRNA'].append(grna)

      # Make predictions for each SpCas9 gRNA targeting exons and introns
      ans = self.predict_sequence(seq, local_cutsite)  # trained k-nn, bp summary dict, normalizer
      # predict all receives seq_context = the gRNA sequence and local_cutsite = the -3 base index
      # pred_del_df = df of predicted unique del products             for sequence context and cutsite
      # pred_all_df = df of all predicted unique in+del products          ^
      # total_phi_score = total NN1+2 phi score                           ^
      # ins_del_ratio = fraction frequency of 1bp ins over all indels     ^
      pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans  #

      # pred_all_df ( pred_del_df only has the first 4 columns, and only with info for dels):
      #   'Length'                predicted in/del length
      #   'Genotype Position'     predicted delta (useful only for dels)
      #   'Predicted_Frequency'   predicted normalised in/del frequency
      #   'Category'              deletion/insertion
      #   'Inserted Bases'        predicted inserted base (useful only for ins)

      # Translate predictions to indel length frequencies
      indel_len_pred, fs = get_indel_len_pred(pred_all_df, self.deletion_len_limit)  # normalised frequency distributon on indel lengths
      all_data['Indel Length Prediction'].append(indel_len_pred)
      # dict: {+1 = [..], -1 = [..], ..., -60 = [..]}
      #   and normalised frequency distribution of frameshifts
      #   fs = {'+0': [..], '+1': [..], '+2': [..]}

      #
      # Store prediction statistics
      #
      all_data['Total Phi Score'].append(total_phi_score)
      all_data['1ins/del Ratio'].append(ins_del_ratio)

      all_data['1ins Rate Model'].append(self.rate_model)
      all_data['1ins bp Model'].append(self.bp_model)
      all_data['1ins normalizer'].append(self.normalizer)

      all_data['Frameshift +0'].append(fs['+0'])
      all_data['Frameshift +1'].append(fs['+1'])
      all_data['Frameshift +2'].append(fs['+2'])
      all_data['Frameshift'].append(fs['+1'] + fs['+2'])

      # get only MH-based deletion genotypes
      s = pred_del_df[pred_del_df['Genotype Position'] != 'e']['Predicted_Frequency']
      s = np.array(s) / sum(s)  # renormalised freq distrib of only MH-based deletion genotypes
      del_gt_precision = 1 - entropy(s) / np.log(len(s))
      all_data['Precision - Del Genotype'].append(del_gt_precision)

      dls = []
      for del_len in range(1, self.deletion_len_limit):
        dlkey = -1 * del_len
        dls.append(indel_len_pred[dlkey])
      dls = np.array(dls) / sum(dls)  # renormalised freq distrib of del lengths
      del_len_precision = 1 - entropy(dls) / np.log(len(dls))
      all_data['Precision - Del Length'].append(del_len_precision)

      # i.e. get only MH-based deletion and 1-bp ins genotypes
      s = pred_all_df[pred_all_df['Genotype Position'] != 'e']['Predicted_Frequency']
      # renormalised freq distrib of MH dels and 1-bp ins
      s = np.array(s) / sum(s)
      all_gt_precision = 1 - entropy(s) / np.log(len(s))
      all_data['Precision - All Genotype'].append(all_gt_precision)
      all_data['Frequency Distribution'].append(s)

      negthree_nt = seq[local_cutsite]  # local_cutsite = 30. I think -1 gives the -4 nt....?
      negfour_nt = seq[local_cutsite - 1]
      all_data['-4 nt'].append(negfour_nt)
      all_data['-3 nt'].append(negthree_nt)

      crit = (pred_all_df['Category'] == 'ins')
      # pred freq for the most freq 1bp ins genotype
      highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
      # pred freq for most freq MH-based del genotype
      highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      all_data['Highest Ins Rate'].append(highest_ins_rate)
      all_data['Highest Del Rate'].append(highest_del_rate)
      timer.update()
    return pd.DataFrame(all_data)

  def predict_using_batches(self, out_dir):
    batches = glob.glob(helper.INPUT_DIRECTORY + "introns_*") + glob.glob(helper.INPUT_DIRECTORY + "exons_*")
    total_samples = helper.SAMPLE_SIZE
    samples_per_batch = int(total_samples / len(batches))
    extra_samples_at_end = total_samples - (samples_per_batch * len(batches))

    for batch in batches:
      all_cutsites = helper.load_pickle(batch)['Chromosome'].to_numpy()
      if batch == batches[0]:
        cutsites = all_cutsites[
          np.random.choice(len(all_cutsites), size=(samples_per_batch + extra_samples_at_end), replace=False)]
      else:
        cutsites = all_cutsites[np.random.choice(len(all_cutsites), size=samples_per_batch, replace=False)]
      predictions = self.predict_all_sequence_outcomes(cutsites)
      file = f'freq_distribution_{batch.split("/")[-1]}.pkl'
      helper.store_predictions(out_dir, file, predictions)

    # TODO return all predictions to plot
