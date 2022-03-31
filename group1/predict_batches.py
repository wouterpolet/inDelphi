import pickle

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
    size = len(sequences)
    timer = util.Timer(total=size)
    nn_params = models['nn']
    nn2_params = models['nn_2']
    rate_model = models['rate']
    bp_model = models['bp']
    normalizer = models['norm']
    for seq in sequences:
        local_cutsite = 30
        ans = predict_all(seq, local_cutsite, nn_params, nn2_params, rate_model, bp_model, normalizer)  # trained k-nn, bp summary dict, normalizer
        pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans
        indel_len_pred, fs = get_indel_len_pred(pred_all_df, 60 + 1)

        crit = (pred_all_df['Genotype Position'] != 'e')  # i.e. get only MH-based deletion and 1-bp ins genotypes
        s = pred_all_df[crit]['Predicted_Frequency']
        s = np.array(s) / sum(s)  # renormalised freq distrib of MH dels and 1-bp ins TODO: extract

        predictions.append((seq, indel_len_pred, s))
        timer.update()
    return predictions

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

predictions = []
for batch in batches:
    helper.print_and_log(f'Starting on batch {batch}', log_fn)
    all_cutsites = load_pickle(batch)['Chromosome'].to_numpy()  # using chromosome because I messed up the saving
    if batch == batches[0]:
        cutsites = all_cutsites[np.random.choice(len(all_cutsites), size=(samples_per_batch + extra_samples_at_end), replace=False)]
    else:
        cutsites = all_cutsites[np.random.choice(len(all_cutsites), size=samples_per_batch, replace=False)]
    predictions.extend(predict(cutsites, models_3))
predictions_file = f'{out_dir + FOLDER_PRED_KEY}freq_distribution.pkl'
pickle.dump(predictions, open(predictions_file, 'wb'))
