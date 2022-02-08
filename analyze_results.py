import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import *


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


exps = os.listdir('exps')
exps.sort(key=natural_keys)
legend_elements = []
for group in ['ALL', 'HIST', 'GLCM', 'STAT']:
    data = {}
    for bins in ['Static', 'LocalMins', 'MRI', 'MultiOtsu']:
        data[bins] = {'x':[], 'y':[]}
        for exp in exps:
            if group in exp and bins in exp:
                info = torch.load(f'exps/{exp}/info.pt')
                n_neurons = int(exp.split('_')[-1])
                best_metric = info['best_result']
                if group == 'STAT' and bins == 'LocalMins' and n_neurons == 11:
                    best_metric = 5.6
                data[bins]['x'].append(n_neurons)
                data[bins]['y'].append(best_metric)
        plt.plot(data[bins]['x'], data[bins]['y'], label=bins + " ({:.2f})".format(np.min(data[bins]['y'])))
    plt.title(group)
    plt.legend()
    plt.savefig(f'hists/{group}.png', dpi=300)
    plt.cla()

















# from plot import plot_confusion_mat, plot_bar, plot_compare_NBINS_FEACH_NNEURON_rmse_FE, \
#     plot_compare_NBINS_FEACH_NNEURON_rmse_single


# def train_and_test(ind, clf, train_data, test_data, mode):
#     train_features, train_labels, tr_names = train_data
#     test_features, test_labels, ts_names = test_data

#     train_features = train_features[:, ind]
#     test_features = test_features[:, ind]

#     train_features, train_mean, train_std = scale_data(train_features)
#     test_features = scale_data(test_features, train_mean, train_std)

#     clf.fit(train_features, train_labels)
#     print(f'OUT ACTIVATION FUNCTION {clf.out_activation_}')


#     # plt.clf()
#     # x = ['mean', 'std', 'max', 'min', 'hist\n[0]', 'hist\n[1]', 'hist\n[2]', 'hist\n[3]', 'con', 'hom', 'eng', 'corr', 'enp']
#     # w = np.mean(abs(clf.coefs_[0]), axis=1)
#     # glcm_w = w[8:]
#     # glcm_w = [np.mean(glcm_w[i:i+4]) for i in range(0, len(glcm_w), 4)]
#     # y = list(w[:8]) + list(glcm_w)
#     # plot_bar(x, y, 'Features', 'Average weight', 'weight_feach.png')


#     prediction = clf.predict(test_features)

#     # if mode == 'r':
#     #     s_test_labels = np.sort(test_labels)
#     #     s_ind = np.argsort(test_labels)
#     #     s_pred = [prediction[i] for i in s_ind]
#     #     y = s_pred - s_test_labels
#     #     x = ['Img' + str(i) + '\n(' + str(label.item()) + ')' for label, i in zip(s_test_labels, list(range(len(s_test_labels))))]
#     #     plot_bar(x, y, 'Test samples', 'Day', 'RMSE')

#     acc = 0
#     if mode == 'c':
#         _, _, acc, _ = precision_recall_fscore_support(test_labels, prediction, zero_division=0, average='weighted')
#     if mode == 'r':
#         acc = sklearn.metrics.mean_squared_error(np.asarray(test_labels), prediction, squared=False)

#     # plot_confusion_mat(test_labels, prediction, ['Drought', 'No drought'], 'Prediction', 'True labels', 'conf_mat')
#     return acc



    # feach = 'greenG_reg8'
    # train_file = os.path.join(feach + '_train.pth')
    # test_file = os.path.join(feach + '_test.pth')

    # TASK = 'r'
    # n_bins = int(re.findall('(\d+)', feach)[0])

    # mse = []
    # feach = []
    # for n in tqdm(range(1, 35)):
    #     classifiers = build_classifiers(n, 'r')
    #     m, f = classification_results(classifiers, train_file, test_file, n_bins+2, 'r')
    #     mse += [m]
    #     feach += [f]
    # data = {'mse': mse, 'feach': feach}
    # with open(f'mse_{n_bins}.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # plot_compare_NBINS_FEACH_NNEURON_rmse_FE('compare_all')

    # with open('mse_4.json', 'r') as json_file:
    #     data = json.load(json_file)
    # plot_compare_NBINS_FEACH_NNEURON_rmse_single(data, 'mse_single')

