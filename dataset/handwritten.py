import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
from .util import normalize

def train_test_datasets(datadir, batch_size=256, seed=1, feat_norm_type='min_max'):
    dataset = sio.loadmat(os.path.join(datadir, 'handwritten_6views.mat'))
    view_number = int((len(dataset) - 5) / 2)
    X_train, X_test = list(), list()
    for v_num in range(view_number):
        X_train.append(normalize(dataset['x' + str(v_num + 1) + '_train'],feat_norm_type=feat_norm_type))
        X_test.append(normalize(dataset['x' + str(v_num + 1) + '_test'],feat_norm_type=feat_norm_type))
    y_train = dataset['gt_train'].squeeze(1)
    y_test = dataset['gt_test'].squeeze(1)

    if np.min(y_train) == 1:
        y_train = y_train - 1
    if np.min(y_test) == 1:
        y_test = y_test - 1
    train_size = X_train[0].shape[0]
    test_size = X_test[0].shape[0]

    X_train.append(y_train)
    X_test.append(y_test)
    train_set = tf.data.Dataset.from_tensor_slices(tuple(X_train)).shuffle(train_size, seed=seed).repeat().batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(X_test)).batch(batch_size)

    return train_set, test_set, train_size, test_size

def noisy_dataset(datadir, noisy_sets=[1,3,4], noise_std=1, batch_size=256, seed=1, feat_norm_type='min_max'):
    dataset = sio.loadmat(os.path.join(datadir, 'handwritten_6views.mat'))
    view_number = int((len(dataset) - 5) / 2)
    X_test = list()
    np.random.seed(seed)
    for v_num in range(view_number):
        X_test.append(normalize(dataset['x' + str(v_num + 1) + '_test'],feat_norm_type=feat_norm_type))
        if v_num in noisy_sets:
            X_test[v_num] = X_test[v_num] + np.random.normal(0.0, noise_std, X_test[v_num].shape)

    y_test = dataset['gt_test'].squeeze(1)

    if np.min(y_test) == 1:
        y_test = y_test - 1
    X_test.append(y_test)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(X_test)).batch(batch_size)
    
    return test_set