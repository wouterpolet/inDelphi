import os
import pickle
import pandas as pd
import autograd.numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor
from functionality.helper import convert_oh_string_to_nparray
from functionality.author_helper import ensure_dir_exists, Timer


def featurize(rate_stats, Y_nm):
  fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
  threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])
  ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
  del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
  Y = np.array(rate_stats[Y_nm])
  print('Entropy Shape: %s, Fivebase Shape: %s, Deletion Score Shape: %s'
        % (ent.shape, fivebases.shape, del_scores.shape))
  print('Y_nm: %s' % Y_nm)

  normalizer = [
    (np.mean(fivebases.T[2]), np.std(fivebases.T[2])),
    (np.mean(fivebases.T[3]), np.std(fivebases.T[3])),
    (np.mean(threebases.T[0]), np.std(threebases.T[0])),
    (np.mean(threebases.T[2]), np.std(threebases.T[2])),
    (np.mean(ent), np.std(ent)),
    (np.mean(del_scores), np.std(del_scores))
  ]

  fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
  fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
  threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
  threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
  gtag = np.array([fiveG, fiveT, threeA, threeG]).T

  ent = (ent - np.mean(ent)) / np.std(ent)
  del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)

  X = np.concatenate((gtag, ent, del_scores), axis=1)
  feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
  print('Num. samples: %s, num. features: %s' % X.shape)

  return X, Y, normalizer


class InsertionModel:
  def __init__(self, model_dir, stat_dir):
    ensure_dir_exists(stat_dir)
    ensure_dir_exists(model_dir)
    self.out_dir_model = model_dir
    self.out_dir_stat = stat_dir

  def get_statistics(self, data_nm, total_values):
    ins_stat_dir = self.out_dir_stat + 'ins_stat.csv'
    bp_stat_dir = self.out_dir_stat + 'bp_stat.csv'

    if os.path.isfile(ins_stat_dir) and os.path.isfile(bp_stat_dir):
      print('Loading statistics...')
      ins_stat = pd.read_csv(ins_stat_dir, index_col=0)
      bp_stat = pd.read_csv(bp_stat_dir, index_col=0)
    else:
      print('Creating statistics...')
      ins_stat, bp_stat = self.prepare_statistics(data_nm, total_values)
      ins_stat.to_csv(ins_stat_dir)
      bp_stat.to_csv(bp_stat_dir)
    return ins_stat, bp_stat

  def prepare_statistics(self, data_nm, total_values):
    # Input: Dataset
    # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
    # Calculate statistics associated with each experiment by name
    bp_ins_df = defaultdict(list)
    ins_ratio_df = defaultdict(list)

    timer = Timer(total=len(total_values))
    exps = data_nm['Sample_Name'].unique()

    data_nm['delta'] = data_nm['Indel'].str.extract(r'(\d+)', expand=True)
    data_nm['nucleotide'] = data_nm['Indel'].str.extract(r'([A-Z]+)', expand=True)
    data_nm['delta'] = data_nm['delta'].astype('int32')

    for id, exp in enumerate(exps):
      exp_data = data_nm[data_nm['Sample_Name'] == exp]
      self.calc_ins_ratio_statistics(exp_data, exp, ins_ratio_df, total_values)
      self.calc_1bp_ins_statistics(exp_data, exp, bp_ins_df)
      timer.update()

    # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
    ins_stat = pd.DataFrame(ins_ratio_df)
    bp_stat = pd.DataFrame(bp_ins_df)
    return ins_stat, bp_stat

  def calc_ins_ratio_statistics(self, all_data, exp, alldf_dict, total_values):
    # Calculate statistics on df, saving to alldf_dict
    # Deletion positions
    total_ins_del_counts = sum(all_data['countEvents'])
    if total_ins_del_counts <= 1000:
      return

    editing_rate = 1  # always 1 since sum(in or del) / sum(in or del which aren't noise)
    ins_count = sum(all_data[(all_data['Type'] == 'INSERTION') & (all_data['delta'] == 1)]['countEvents'])
    del_count = sum(all_data[all_data['Type'] == 'DELETION']['countEvents'])
    mhdel_count = sum(all_data[(all_data['Type'] == 'DELETION') & (all_data['homologyLength'] != 0)]['countEvents'])

    ins_ratio = ins_count / total_ins_del_counts
    fivebase = exp[len(exp) - 4]

    if len(total_values[total_values['exp'] == exp]) > 0:
      del_score = total_values[total_values['exp'] == exp]['total_phi'].values[0]
      norm_entropy = total_values[total_values['exp'] == exp]['norm_entropy'].values[0]
    else:
      del_score = 0
      norm_entropy = 0

    if fivebase == 'A':
      fivebase_oh = np.array([1, 0, 0, 0])
    if fivebase == 'C':
      fivebase_oh = np.array([0, 1, 0, 0])
    if fivebase == 'G':
      fivebase_oh = np.array([0, 0, 1, 0])
    if fivebase == 'T':
      fivebase_oh = np.array([0, 0, 0, 1])

    threebase = exp[len(exp) - 3]

    if threebase == 'A':
      threebase_oh = np.array([1, 0, 0, 0])
    if threebase == 'C':
      threebase_oh = np.array([0, 1, 0, 0])
    if threebase == 'G':
      threebase_oh = np.array([0, 0, 1, 0])
    if threebase == 'T':
      threebase_oh = np.array([0, 0, 0, 1])

    alldf_dict['Editing Rate'].append(editing_rate)
    alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))
    alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
    alldf_dict['Ins1bp Ratio'].append(ins_ratio)
    alldf_dict['Fivebase'].append(fivebase)
    alldf_dict['Del Score'].append(del_score)
    alldf_dict['Entropy'].append(norm_entropy)
    alldf_dict['Fivebase_OH'].append(fivebase_oh)
    alldf_dict['Threebase'].append(threebase)
    alldf_dict['Threebase_OH'].append(threebase_oh)
    alldf_dict['_Experiment'].append(exp)
    return alldf_dict

  def calc_1bp_ins_statistics(self, all_data, exp, alldf_dict):
    # Normalize Counts
    total_count = sum(all_data['countEvents'])
    all_data['Frequency'] = all_data['countEvents'].div(total_count)

    insertions = all_data[all_data['Type'] == 'INSERTION']
    insertions = insertions[insertions['delta'] == 1]

    if sum(insertions['countEvents']) <= 100:
      return

    freq = sum(insertions['Frequency'])
    a_frac = sum(insertions[insertions['nucleotide'] == 'A']['Frequency']) / freq
    c_frac = sum(insertions[insertions['nucleotide'] == 'C']['Frequency']) / freq
    g_frac = sum(insertions[insertions['nucleotide'] == 'G']['Frequency']) / freq
    t_frac = sum(insertions[insertions['nucleotide'] == 'T']['Frequency']) / freq
    alldf_dict['Frequency'].append(freq)
    alldf_dict['A frac'].append(a_frac)
    alldf_dict['C frac'].append(c_frac)
    alldf_dict['G frac'].append(g_frac)
    alldf_dict['T frac'].append(t_frac)

    fivebase = exp[len(exp) - 4]
    alldf_dict['Base'].append(fivebase)

    alldf_dict['_Experiment'].append(exp)
    return alldf_dict

  def generate_models(self, X, y, bp_stats, Normalizer):
    # Train rate model
    model = KNeighborsRegressor()
    model.fit(X, y)
    with open(self.out_dir_model + 'rate_model.pkl', 'wb') as f:
      pickle.dump(model, f)

    # Obtain bp stats
    bp_model = dict()
    ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
    t_melt = pd.melt(bp_stats, id_vars=['Base'], value_vars=ins_bases, var_name='Ins Base', value_name='Fraction')
    for base in list('ACGT'):
      bp_model[base] = dict()
      mean_vals = []
      for ins_base in ins_bases:
        crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
        mean_vals.append(float(np.mean(t_melt[crit])))
      for bp, freq in zip(list('ACGT'), mean_vals):
        bp_model[base][bp] = freq / sum(mean_vals)

    with open(self.out_dir_model + 'bp_model.pkl', 'wb') as f:
      pickle.dump(bp_model, f)

    with open(self.out_dir_model + 'Normalizer.pkl', 'wb') as f:
      pickle.dump(Normalizer, f)

    return model, bp_model, Normalizer

  def train_knn(self, all_data, total_values):
    ensure_dir_exists(self.out_dir_stat)
    rate_stats, bp_stats = self.get_statistics(all_data, total_values)
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    x, y, normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
    return self.generate_models(x, y, bp_stats, normalizer)

