# TRAIN = False
# PLOT = True
import os
import shap
import pickle
import warnings

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

from functionality import helper
from functionality import neural_networks as nn
from functionality.author_helper import nn_match_score_function

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

EXECUTION_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FOLDER_INPUT_KEY = '/in/'
FOLDER_OUTPUT_KEY = '/out/'
FOLDER_PARAM_KEY = 'parameters/'
global out_dir
out_dir = EXECUTION_PATH + FOLDER_OUTPUT_KEY
log_fn = out_dir + 'wouter.log'
with open(log_fn, 'w') as f:
    pass
exec_id = 'wouter_rq'
out_folder = out_dir + 'wouter/'


def get_network_one(nn_params):
    def nn_one_predict(sample):
        res = []
        for s in sample:
            mh_len = s[0]
            gc_frac = s[1]
            del_len = s[2]
            pred_input = np.array([mh_len, gc_frac]).T  # input to MH-NN
            del_lens = np.array(del_len).T  # input to MH-less NN
            mh_scores = nn_match_score_function(nn_params, pred_input)
            Js = np.array(del_lens)
            unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
            mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)
            res.append(mh_phi_total)
        return np.asarray(res)

    return nn_one_predict

def get_network_two(nn_params):
    def nn_two_predict(sample):
        res = []
        for s in sample:
            del_len = s[0]
            mhless_score = nn_match_score_function(nn_params, np.asarray(del_len))
            res.append(mhless_score)
        return np.asarray(res)
    return nn_two_predict


def train():
    all_data_mesc = pd.concat(helper.read_data(helper.INPUT_DIRECTORY + 'dataset.pkl'), axis=1).reset_index()
    # all_data_mesc = all_data_mesc[all_data_mesc['Type'] == 'DELETION'].reset_index()
    params = nn.create_neural_networks(all_data_mesc, log_fn, out_folder, exec_id)
    pickle.dump(params, open(out_folder + '/params.pkl', 'wb'))


def compute_shap(nn_one=False, nn_two=False):
    ans = pickle.load(open(out_folder + '/ans.pkl', 'rb'))
    INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
    trained_params = pickle.load(open(out_folder + '/params.pkl', 'rb'))
    np.concatenate(INP_train).ravel().reshape((356642 // 2, 2))
    del_feature = np.concatenate(DEL_LENS_train).ravel()
    mh_features = np.concatenate(INP_train).ravel().reshape((len(del_feature), 2))
    samples = np.c_[mh_features, del_feature]
    del_feature_test = np.concatenate(DEL_LENS_test).ravel()
    mh_features_test = np.concatenate(INP_test).ravel().reshape((len(del_feature_test), 2))
    samples_test = np.c_[mh_features_test, del_feature_test]
    if nn_two:
        del_feature_one_dim = del_feature.reshape((len(del_feature), 1))
        del_feature_test_one_dim = del_feature_test.reshape((len(del_feature_test), 1))

        # background = shap.sample(del_feature_one_dim, 20000)
        # test_samples = shap.sample(del_feature_test_one_dim, 10000)
        background = del_feature_one_dim
        test_samples = del_feature_test_one_dim

        explainer = shap.Explainer(get_network_two(trained_params[1]), background)
        shap_values = explainer(test_samples)
        pickle.dump(background, open(out_folder + '/background_nn_2.pkl', 'wb'))
        pickle.dump(test_samples, open(out_folder + '/test_samples_nn_2.pkl', 'wb'))
        pickle.dump(shap_values, open(out_folder + '/shap_values_nn_2.pkl', 'wb'))
    if nn_one:
        # background = shap.sample(samples, 20000)
        # test_samples = shap.sample(samples_test, 10000)
        background = samples
        test_samples = samples_test

        explainer = shap.Explainer(get_network_one(trained_params[0]), background)
        shap_values = explainer(test_samples)
        pickle.dump(background, open(out_folder + '/background_nn_1.pkl', 'wb'))
        pickle.dump(test_samples, open(out_folder + '/test_samples_nn_1.pkl', 'wb'))
        pickle.dump(shap_values, open(out_folder + '/shap_values_nn_1.pkl', 'wb'))


def plot_shap():
    shap_values_one = pickle.load(open(out_folder + 'shap_values_nn_1.pkl', 'rb'))
    shap_values_one.feature_names = ['MH len', 'GC frac', 'DEL len']
    shap.plots.beeswarm(shap_values_one)
    shap.plots.scatter(shap_values_one[:, 'MH len'], color=shap_values_one)
    shap.plots.scatter(shap_values_one[:, 'GC frac'], color=shap_values_one)
    shap.plots.scatter(shap_values_one[:, 'DEL len'], color=shap_values_one)
    shap.plots.bar(shap_values_one)
    shap_values_two = pickle.load(open(out_folder + 'shap_values_nn_2.pkl', 'rb'))
    shap_values_two.feature_names = ['DEL len']
    shap.plots.beeswarm(shap_values_two)


if __name__ == '__main__':
    # train()
    # compute_shap(nn_one=True, nn_two=True)
    plot_shap()

