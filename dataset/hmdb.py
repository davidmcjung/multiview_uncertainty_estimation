import numpy as np
import tensorflow as tf
import os
import h5py
from .util import normalize

def train_test_datasets(datadir, batch_size=256, seed=1, feat_norm_type='min_max'):
    dataset = {}
    f = h5py.File(os.path.join(datadir, 'HMDB51_HOG_MBH.mat'))
    for k, v in f.items():
        dataset[k] = np.array(v)
    view_number = len(dataset)-1
    train_idx = np.load(os.path.join(datadir, 'train_idx.npy'))
    test_idx = np.load(os.path.join(datadir, 'test_idx.npy'))

    X_train, X_test = list(), list()
    for v_num in range(view_number):
        view_data = dataset[f'x{v_num+1}'].transpose()
        X_train.append(normalize(view_data[train_idx],feat_norm_type=feat_norm_type))
        X_test.append(normalize(view_data[test_idx],feat_norm_type=feat_norm_type))
    y_train = dataset['gt'][0][train_idx].astype('int32')
    y_test = dataset['gt'][0][test_idx].astype('int32')

    if np.min(y_train) == 1:
        y_train = y_train - 1
    if np.min(y_test) == 1:
        y_test = y_test - 1
    train_size = X_train[0].shape[0]
    test_size = X_test[0].shape[0]
    lengthscales_init = []
    for v_num in range(view_number):
        lengthscales_init.append(np.std(X_train[v_num],axis=0)*np.sqrt(X_train[v_num].shape[1]))
    X_train.append(y_train)
    X_test.append(y_test)
    train_set = tf.data.Dataset.from_tensor_slices(tuple(X_train)).shuffle(train_size, seed=seed).repeat().batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(X_test)).batch(batch_size)

    return train_set, test_set, train_size, test_size, lengthscales_init

def noisy_dataset(datadir, noisy_sets=[1], noise_std=1, batch_size=256, seed=1, feat_norm_type='min_max'):
    dataset = {}
    f = h5py.File(os.path.join(datadir, 'HMDB51_HOG_MBH.mat'))
    for k, v in f.items():
        dataset[k] = np.array(v)
    view_number = len(dataset)-1
    test_idx = np.load(os.path.join(datadir, 'test_idx.npy'))

    X_test = list()
    np.random.seed(seed)
    for v_num in range(view_number):
        view_data = dataset[f'x{v_num+1}'].transpose()
        X_test.append(normalize(view_data[test_idx],feat_norm_type=feat_norm_type))
        if v_num in noisy_sets:
            X_test[v_num] = X_test[v_num] + np.random.normal(0.0, noise_std, X_test[v_num].shape)
    y_test = dataset['gt'][0][test_idx].astype('int32')
    if np.min(y_test) == 1:
        y_test = y_test - 1
    X_test.append(y_test)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(X_test)).batch(batch_size)
    
    return test_set