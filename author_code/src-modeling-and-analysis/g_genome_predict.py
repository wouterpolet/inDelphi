from __future__ import division
import _config, _predict
import sys, datetime, pickle
import re
import random
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from author_code.mylib import util, compbio
import pandas as pd
from scipy.stats import entropy

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_split/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
exon_dfs_out_dir = out_dir + 'exon_dfs/'
intron_dfs_out_dir = out_dir + 'intron_dfs/'

rate_model = None
rate_model_nm = None
bp_model = None
bp_model_nm = None
normalizer = None
normalizer_nm = None

##
# Insertion Modeling Init
##
def init_rate_bp_models():
  global rate_model
  global rate_model_nm
  global bp_model
  global bp_model_nm
  global normalizer
  global normalizer_nm

  model_dir = '/cluster/mshen/prj/mmej_figures/out/e5_ins_ratebpmodel/'

  rate_model_nm = 'rate_model_v2'
  bp_model_nm = 'bp_model_v2'
  normalizer_nm = 'Normalizer_v2'

  print 'Loading %s...\nLoading %s...' % (rate_model_nm, bp_model_nm)
  with open(model_dir + '%s.pkl' % (rate_model_nm)) as f:
    rate_model = pickle.load(f)
  with open(model_dir + '%s.pkl' % (bp_model_nm)) as f:
    bp_model = pickle.load(f)
  with open(model_dir + '%s.pkl' % (normalizer_nm)) as f:
    normalizer = pickle.load(f)
  return

##
# Text parsing
##
def parse_header(header):
  w = header.split('_')
  gene_kgid = w[0].replace('>', '')
  chrom = w[1]
  start = int(w[2]) - 30
  end = int(w[3]) + 30
  data_type = w[4]
  return gene_kgid, chrom, start, end

##
# IO
##
def maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = False):
  if split == '0':
    line_threshold = 500
  else:
    line_threshold = 5000
  norm_condition = bool(bool(len(dd['Unique ID']) > line_threshold) and bool(len(dd_shuffled['Unique ID']) > line_threshold))
  # len ('Unique ID') = 34 chars long + length of gene_kgid, which depends on how long the gene was named in the database
  # I think that most of the time, norm_condition is False

  if norm_condition or force:                   # Likely False if force = False
    print 'Flushing, num. %s' % (num_flushed)
    df_out_fn = out_dir + '%s_%s_%s.csv' % (data_nm, split, num_flushed)  # data_nm = 'exons'/'introns', split = '0', '', or ?, num_flushed = 0
    df = pd.DataFrame(dd) # convert sequence info dict into df
    df.to_csv(df_out_fn)  # turn to csv

    df_out_fn = out_dir + '%s_%s_shuffled_%s.csv' % (data_nm, split, num_flushed)
    df = pd.DataFrame(dd_shuffled)  #  convert flushed sequence info dict into df
    df.to_csv(df_out_fn)            # turn to csv

    num_flushed += 1                # number of saves to csv ('flushing')
    dd = defaultdict(list)          # and flush out content in dictionary (reinitialise)
    dd_shuffled = defaultdict(list) # here too
  else:
    pass                            # don't flush
  return dd, dd_shuffled, num_flushed   # if flushed: empty dicts and count of flushes; if not flushed, same dicts and num_flush const.

##
# Prediction
##
# inp_fn: path + file.fa
def find_cutsites_and_predict(inp_fn, data_nm, split):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  _predict.init_model(run_iter ='aax', param_iter ='aag') # load nn_params, nn2_params and functions from d2_model
  dd = defaultdict(list)            # dictionary with values as list
  dd_shuffled = defaultdict(list)

  if data_nm == 'exons':
    df_out_dir = exon_dfs_out_dir
  elif data_nm == 'introns':
    df_out_dir = intron_dfs_out_dir

  num_flushed = 0
  timer = util.Timer(total = util.line_count(inp_fn))
  with open(inp_fn) as f:         # as long as exon/intron database is open
    for i, line in enumerate(f):  #   for each line index and line text in the database
      if i % 2 == 0:
        header = line.strip()     #     if line is even numbered, get sequence header info
      if i % 2 == 1:
        sequence = line.strip()   #     if line is odd numbered, get the sequence
                                  #     and if got the sequence, go on to prediction
        if len(sequence) < 60:
          continue
        if len(sequence) > 500000:
          continue

        bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir)
        # generate dicts (dd and dd_shuffled) for info on each cutsite seq found in the sequence and for a shuffled cutsite sequence
        dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)  # likely no flush
        #                               maybe flush out the dict contents into csv and return empty dicts
      if (i - 1) % 50 == 0 and i > 1:
        print '%s pct, %s' % (i / 500, datetime.datetime.now())

      timer.update()

  maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = True)   # will flush due to forced flushing
  return

def get_indel_len_pred(pred_all_df):
  indel_len_pred = dict()

  # 1 bp insertions
  crit = (pred_all_df['Category'] == 'ins')                                 # for all insertions
  indel_len_pred[1] = float(sum(pred_all_df[crit]['Predicted_Frequency']))  # predicted frequency of 1bp ins over all indel products
                                                                            # store for +1 key in dictionary
  # Deletions
  for del_len in range(1, 60):
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


##
def bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir):
  # Input: A specific sequence
  # Find all Cas9 cutsites, gather metadata, and run inDelphi
  try:
    ans = parse_header(header)          # header is of FASTA type from NCBI
    gene_kgid, chrom, start, end = ans  # gene id in database / chromosome nunber / start of seq - 30 / end of sequence + 30
  except:
    return

  for idx in range(len(sequence)):        # for each base in the sequence
    # this loop finishes only each of 5% of all found cutsites with 60-bp long sequences containing only ACGT
    seq = ''
    if sequence[idx : idx+2] == 'CC':             # if on top strand find CC
      cutsite = idx + 6                           # cut site of complementary GG is +6 away
      seq = sequence[cutsite - 30 : cutsite + 30] # get sequence 30bp L and R of cutsite
      seq = compbio.reverse_complement(seq)       # compute reverse strand (complimentary) to target with gRNA
      orientation = '-'
    if sequence[idx : idx+2] == 'GG':             # if GG on top strand
      cutsite = idx - 4                           # cut site is -4 away
      seq = sequence[cutsite - 30 : cutsite + 30] # get seq 30bp L and R of cutsite
      orientation = '+'
    if seq == '':
      continue
    if len(seq) != 60:
      continue

    # Sanitize input
    seq = seq.upper()
    if 'N' in seq:                      # if N in collected sequence, return to start of for loop / skip rest
      continue
    if not re.match('^[ACGT]*$', seq):  # if there not only ACGT in seq, ^
      continue

    # Randomly query subset for broad shallow coverage
    r = np.random.random()
    if r > 0.05:
      continue              # randomly decide if will predict on the found cutsite or not. 5% of time will

    # Shuffle everything but GG
    seq_nogg = list(seq[:34] + seq[36:])
    random.shuffle(seq_nogg)
    shuffled_seq = ''.join(seq_nogg[:34]) + 'GG' + ''.join(seq_nogg[36:])       # a sort of -ve control

    # for one set of sequence context and its shuffled counterpart
    for d, seq_context, shuffled_nm in zip([dd, dd_shuffled],     # initially empty dicts (values as list) for each full exon/intron
                                           [seq, shuffled_seq],   # sub-exon/intron cutsite sequence and shuffled sequence
                                           ['wt', 'shuffled']):
      #
      # Store metadata statistics
      #
      local_cutsite = 30
      grna = seq_context[13:33]
      cutsite_coord = start + idx
      unique_id = '%s_%s_hg38_%s_%s_%s' % (gene_kgid, grna, chrom, cutsite_coord, orientation)

      # the SpCas9 gRNAs targeting exons and introns
      d['Sequence Context'].append(seq_context)
      d['Local Cutsite'].append(local_cutsite)
      d['Chromosome'].append(chrom)
      d['Cutsite Location'].append(cutsite_coord)
      d['Orientation'].append(orientation)
      d['Cas9 gRNA'].append(grna)
      d['Gene kgID'].append(gene_kgid)
      d['Unique ID'].append(unique_id)

      # Make predictions for each SpCas9 gRNA targeting exons and introns
      ans = _predict.predict_all(seq_context, local_cutsite,  # seq_context is a tuple/pair? of seq and shuffled_seq
                                 rate_model, bp_model, normalizer)      # trained k-nn, bp summary dict, normalizer
      pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans    #
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
      indel_len_pred, fs = get_indel_len_pred(pred_all_df)    # normalised frequency distributon on indel lengths
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

      d['1ins Rate Model'].append(rate_model_nm)
      d['1ins bp Model'].append(bp_model_nm)
      d['1ins normalizer'].append(normalizer_nm)

      d['Frameshift +0'].append(fs['+0'])
      d['Frameshift +1'].append(fs['+1'])
      d['Frameshift +2'].append(fs['+2'])
      d['Frameshift'].append(fs['+1'] + fs['+2'])

      crit = (pred_del_df['Genotype Position'] != 'e')    # get only MH-based deletion genotypes
      s = pred_del_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)                            # renormalised freq distrib of only MH-based deletion genotypes
      del_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - Del Genotype'].append(del_gt_precision)
      
      dls = []
      for del_len in range(1, 60):
        dlkey = -1 * del_len
        dls.append(indel_len_pred[dlkey])
      dls = np.array(dls) / sum(dls)                      # renormalised freq distrib of del lengths
      del_len_precision = 1 - entropy(dls) / np.log(len(dls))
      d['Precision - Del Length'].append(del_len_precision)
      
      crit = (pred_all_df['Genotype Position'] != 'e')    # i.e. get only MH-based deletion and 1-bp ins genotypes
      s = pred_all_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)                            # renormalised freq distrib of MH dels and 1-bp ins
      all_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - All Genotype'].append(all_gt_precision)

      negthree_nt = seq_context[local_cutsite - 1]    # local_cutsite = 30. I think -1 gives the -4 nt....?
      negfour_nt = seq_context[local_cutsite]
      d['-4 nt'].append(negfour_nt)
      d['-3 nt'].append(negthree_nt)

      crit = (pred_all_df['Category'] == 'ins')
      highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for the most freq 1bp ins genotype
      crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
      highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])  # pred freq for most freq MH-based del genotype
      d['Highest Ins Rate'].append(highest_ins_rate)
      d['Highest Del Rate'].append(highest_del_rate)

  return


##
# nohups
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating nohup scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  w_dir = _config.SRC_DIR
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  curr_num = 0
  num_scripts = 0
  nums = {'exons': 36, 'introns': 32}
  for typ in nums:
    for split in range(nums[typ]):
      script_id = NAME.split('_')[0]
      command = 'python -u %s.py %s %s' % (NAME, typ, split)

      script_abbrev = NAME.split('_')[0]
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_abbrev, typ, split)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      curr_num += 1

      # Write qsub commands
      qsub_commands.append('qsub -m e -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (curr_num, qsubs_dir)
  return


##
# Main
##
@util.time_dec
def main(data_nm = '', split = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)           # check that databases exist
  util.ensure_dir_exists(exon_dfs_out_dir)
  util.ensure_dir_exists(intron_dfs_out_dir)

  if data_nm == '' and split == '':
    gen_qsubs()
    return

  # # Default params
  # DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_split/'
  # NAME = util.get_fn(__file__)
  # out_dir = _config.OUT_PLACE + NAME + '/'
  # exon_dfs_out_dir = out_dir + 'exon_dfs/'
  # intron_dfs_out_dir = out_dir + 'intron_dfs/'
  inp_fn = DEFAULT_INP_DIR + '%s_%s.fa' % (data_nm, split)
  init_rate_bp_models()                     # load fitted k-nn model, the bp_model dict and normalizer as global vars
  find_cutsites_and_predict(inp_fn, data_nm, split)
                          # input = (exon/intron file address, exons or introns dataframes, first 500 or 5000 objects)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(data_nm = sys.argv[1], split = sys.argv[2])
  else:
    main()
