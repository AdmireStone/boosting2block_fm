# coding:utf8
import sys
from scipy.io import mmread
import numpy as np
from math import exp
import random
import pandas as pd
import sys
import pickle
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import roc_auc_score
import time


def predict_b2b(W, Z, u, items,item_one_hot_dim,user_one_hot_dim):
    #     print type(items)
    items_onehot = sp.identity(item_one_hot_dim, dtype='f', format='csr')[list(items)]
    items_num = len(items)

    u_max = np.array([0.] * user_one_hot_dim * items_num).reshape(items_num, user_one_hot_dim)
    u_max[:, u] = 1.
    u_max = sp.csr_matrix(u_max)

    eval_data = sp.hstack([u_max, items_onehot])

    linear_term = safe_sparse_dot(eval_data, W).T

    qusdratic_term = (safe_sparse_dot(safe_sparse_dot(eval_data, Z), eval_data.T)).diagonal()

    return np.asarray(linear_term + qusdratic_term)
    # return np.asarray(linear_term + qusdratic_term)[0]


def getSingleContextAUC(s_c_ob, s_c_non_ob):
    acc = 0.0
    s_c_non_ob = np.ravel(s_c_non_ob)
    s_c_ob = np.ravel(s_c_ob)
    for s_ci in s_c_ob:
        acc += np.sum(s_c_non_ob < s_ci)
    return acc / (len(s_c_ob) * len(s_c_non_ob))


def load_test_required_data():
    datapath = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/ml-100k/'
    test_datapath = datapath+"bprmf_test_0_1.txt"
    tr_datapath = datapath+"bprmf_train_0_1.txt"
    data = mmread(test_datapath).tocsr()
    train_data = mmread(tr_datapath).tocsr()

    users_index, items_index = data.nonzero()
    users_unique = np.unique(users_index)
    # 获取所有的items集合
    test_items_set = set(np.unique(items_index))
    _, ite = train_data.nonzero()
    train_items_set = set(np.unique(ite))
    items_set = train_items_set | test_items_set

    return users_unique, data, train_data, items_set

def load_test_required_data_jester2():
    datapath = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/jester-2/'
    test_datapath = datapath+"coo_test_50_5_data.dat.txt"
    tr_datapath = datapath+"coo_train_50_5_data.dat.txt"
    data = mmread(test_datapath).tocsr()
    train_data = mmread(tr_datapath).tocsr()

    users_index, items_index = data.nonzero()
    users_unique = np.unique(users_index)
    # 获取所有的items集合
    test_items_set = set(np.unique(items_index))
    _, ite = train_data.nonzero()
    train_items_set = set(np.unique(ite))
    items_set = train_items_set | test_items_set

    return users_unique, data, train_data, items_set


def predic_auc(W, Z):

    # users_unique, data, train_data, items_set = load_test_required_data()
    users_unique, data, train_data, items_set = load_test_required_data_jester2()
    totoal_auc = 0.0
    count = 0
    context_num = len(users_unique)

    # 临时强行转换，只适合与jester-2
    item_one_hot_dim = 146
    user_one_hot_dim = len(users_unique)
    # file_name="/home/zju/dgl/dataset/recommend/ml-100k/12_20_22_44/u1_iter_59_save_weight.pkl"
    # W,Z=loadB2bModel(weight_file)
    for u in users_unique:
        positive = set(data[u].indices)
        # 获取当前用户在训练数据集中的正标签
        train_pos = set(train_data[u].indices)
        neg = items_set - train_pos - positive
        s_c_ob = predict_b2b(W, Z, u, positive,item_one_hot_dim,user_one_hot_dim)
        s_c_non_ob = predict_b2b(W, Z, u, neg,item_one_hot_dim,user_one_hot_dim)
        acc = getSingleContextAUC(s_c_ob, s_c_non_ob)
        totoal_auc += acc
        count += 1
    return totoal_auc / context_num


def predict_b2b_full_auc(W, Z, u, items):
    #     print type(items)
    items_onehot = sp.identity(1522, dtype='f', format='csr')[list(items)]
    items_num = len(items)

    u_max = np.array([0.] * 943 * items_num).reshape(items_num, 943)
    u_max[:, u] = 1.
    u_max = sp.csr_matrix(u_max)

    eval_data = sp.hstack([u_max, items_onehot])

    linear_term = safe_sparse_dot(eval_data, W).T

    qusdratic_term = (safe_sparse_dot(safe_sparse_dot(eval_data, Z), eval_data.T)).diagonal()

    return 1. / (1. + np.exp((-1) * np.asarray(linear_term + qusdratic_term)[0]))


def full_auc(W, Z, ground_truth):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = predict_b2b_full_auc(W, Z, user_id, pid_array)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)

if __name__=='__main__':
    users_unique, data, train_data, items_set=load_test_required_data_jester2()
    print users_unique