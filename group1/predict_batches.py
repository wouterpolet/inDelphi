import pickle
from collections import defaultdict

import pandas as pd

from all_func import load_models
from group1.prediction import predict_all, get_indel_len_pred
from helper import load_pickle
import os, datetime, util, helper
import numpy as np
import glob
import autograd.numpy as np
from scipy.stats import entropy


FOLDER_STAT_KEY = 'statistics/'
FOLDER_PARAM_KEY = 'parameters/'
FOLDER_GRAPH_KEY = 'plots/'
FOLDER_LOG_KEY = 'logs/'
FOLDER_PRED_KEY = 'predictions/'
FOLDER_INPUT_KEY = '/in/'
EXECUTION_PATH = os.path.dirname(os.path.dirname(__file__))


def initialize_files_and_folders(user_exec_id):
    # Set output location of model & params
    out_place = os.path.dirname(os.path.dirname(__file__)) + '/out/'
    util.ensure_dir_exists(out_place)
    exec_id = ''
    # num_folds = helper.count_num_folders(out_place)
    if user_exec_id == '' or helper.count_num_folders(out_place) < 1:
        exec_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    else:
        latest = datetime.datetime.strptime('1990/01/01', '%Y/%m/%d')
        for name in os.listdir(out_place):
            try:
                datetime.datetime.strptime(name, "%Y%m%d_%H%M")
            except ValueError:
                if name == user_exec_id:
                    exec_id = name
                    break
                else:
                    continue
            date_time_obj = datetime.datetime.strptime(name, "%Y%m%d_%H%M")
            if name == user_exec_id:
                latest = date_time_obj
                break
            if latest < date_time_obj:
                latest = date_time_obj
        if exec_id == '':
            exec_id = latest.strftime("%Y%m%d_%H%M")

    # if use_prev and num_folds >= 1:
    #   out_letters = helper.alphabetize(num_folds - 1)
    # else:
    #   out_letters = helper.alphabetize(num_folds)

    out_dir = out_place + exec_id + '/'
    util.ensure_dir_exists(out_dir + FOLDER_PRED_KEY)
    util.ensure_dir_exists(out_dir + FOLDER_GRAPH_KEY)
    util.ensure_dir_exists(out_dir + FOLDER_PRED_KEY + FOLDER_PARAM_KEY)
    util.ensure_dir_exists(out_dir + FOLDER_PARAM_KEY + FOLDER_STAT_KEY)
    util.ensure_dir_exists(out_dir + FOLDER_LOG_KEY)

    log_fn = out_dir + FOLDER_LOG_KEY + '_log_%s.out' % datetime.datetime.now().strftime("%Y%m%d_%H%M")
    with open(log_fn, 'w') as f:
        pass
    helper.print_and_log('out dir: ' + out_dir, log_fn)

    return out_dir, log_fn, exec_id


def predict(sequences, models):
    all_data = defaultdict(list)
    size = len(sequences)
    timer = util.Timer(total=size)
    nn_params = models['nn']
    nn2_params = models['nn_2']
    rate_model = models['rate']
    bp_model = models['bp']
    normalizer = models['norm']

    for seq in sequences:
        local_cutsite = 30
        grna = seq[13:33]
        all_data['Sequence Context'].append(seq)
        all_data['Local Cutsite'].append(local_cutsite)
        all_data['Cas9 gRNA'].append(grna)
        ans = predict_all(seq, local_cutsite, nn_params, nn2_params, rate_model, bp_model, normalizer)  # trained k-nn, bp summary dict, normalizer
        pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans
        indel_len_pred, fs = get_indel_len_pred(pred_all_df, 60 + 1)
        all_data['Indel Length Prediction'].append(indel_len_pred)
        all_data['Total Phi Score'].append(total_phi_score)
        all_data['1ins/del Ratio'].append(ins_del_ratio)

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
        for del_len in range(1, 61):
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

out_directory, log_file, execution_id = initialize_files_and_folders('')
global log_fn
log_fn = log_file
global out_dir
out_dir = os.path.dirname(os.path.dirname(out_directory)) + '/3f_all_samples/'
global exec_id
exec_id = execution_id
global input_dir
input_dir = EXECUTION_PATH + FOLDER_INPUT_KEY

out_nn_param_dir = out_dir + FOLDER_PARAM_KEY
out_stat_dir = out_dir + FOLDER_STAT_KEY
out_plot_dir = out_dir + FOLDER_GRAPH_KEY
model_folder = out_dir + 'fig_3/'
helper.print_and_log("Loading models...", log_fn)
models_3 = load_models(model_folder)

batches = glob.glob(input_dir + "introns_*") + glob.glob(input_dir + "exons_*")
total_samples = 1003524
samples_per_batch = int(total_samples / len(batches))
extra_samples_at_end = total_samples - (samples_per_batch * len(batches))

for batch in batches:
    helper.print_and_log(f'Starting on batch {batch}', log_fn)
    all_cutsites = load_pickle(batch)['Chromosome'].to_numpy()  # using chromosome because I messed up the saving
    if batch == batches[0]:
        cutsites = all_cutsites[np.random.choice(len(all_cutsites), size=(samples_per_batch + extra_samples_at_end), replace=False)]
    else:
        cutsites = all_cutsites[np.random.choice(len(all_cutsites), size=samples_per_batch, replace=False)]
    helper.print_and_log(f'Sampled {len(cutsites)} cutsites', log_fn)
    predictions = predict(cutsites, models_3)
    predictions_file = f'{out_dir + FOLDER_PRED_KEY}freq_distribution_{batch.split("/")[-1]}'
    helper.print_and_log('Saving predictions', log_fn)
    pickle.dump(predictions, open(predictions_file, 'wb'))

