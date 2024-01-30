import numpy as np
import tensorflow as tf
import os
from .util import normalize

CIFAR10_NUM_TRAINING_SAMPLES = 50000
CIFAR10_NUM_TEST_SAMPLES = 10000

def train_test_datasets(datadir, corruptions=['gaussian_noise', 'glass_blur', 'snow'], severities=[1,1,1], batch_size=256, seed=1):
    # Train dataset
    train_dataset = []
    for corruption,severity in zip(corruptions,severities):
        loaded_dataset = np.load(os.path.join(datadir, 'train', 'feature', f'{corruption}_severity_{severity}_inceptionv3_feat.npy'))
        loaded_dataset = normalize(loaded_dataset, feat_norm_type='standard')
        train_dataset.append(loaded_dataset)
    train_label = np.load(os.path.join(datadir, 'train', 'labels.npy'))
    for severity in range(1,5):
        assert(tf.reduce_any(train_label[CIFAR10_NUM_TRAINING_SAMPLES*(severity-1):CIFAR10_NUM_TRAINING_SAMPLES*severity]!=
        train_label[CIFAR10_NUM_TRAINING_SAMPLES*severity:CIFAR10_NUM_TRAINING_SAMPLES*(severity+1)]) == False)
    train_label = train_label[:CIFAR10_NUM_TRAINING_SAMPLES]
    if np.min(train_label) == 1:
        train_label = train_label - 1
    train_size = len(train_label)

    # Test dataset
    test_dataset = []
    for corruption,severity in zip(corruptions,severities):
        loaded_dataset = np.load(os.path.join(datadir, 'test', 'feature', f'{corruption}_severity_{severity}_inceptionv3_feat.npy'))
        loaded_dataset = normalize(loaded_dataset, feat_norm_type='standard')
        test_dataset.append(loaded_dataset)
    test_label = np.load(os.path.join(datadir, 'test', 'labels.npy'))
    for severity in range(1,5):
        assert(tf.reduce_any(test_label[CIFAR10_NUM_TEST_SAMPLES*(severity-1):CIFAR10_NUM_TEST_SAMPLES*severity]!=
        test_label[CIFAR10_NUM_TEST_SAMPLES*severity:CIFAR10_NUM_TEST_SAMPLES*(severity+1)]) == False)
    test_label = test_label[:CIFAR10_NUM_TEST_SAMPLES]
    if np.min(test_label) == 1:
        test_label = test_label - 1
    test_size = len(test_label)

    # Create datasets
    train_dataset.append(train_label)
    train_set = tf.data.Dataset.from_tensor_slices(tuple(train_dataset)).shuffle(train_size, seed=seed).repeat().batch(batch_size)
    test_dataset.append(test_label)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(test_dataset)).batch(batch_size)

    return train_set, test_set, train_size, test_size

def svhn_ood_dataset(cifar10c_datadir, svhn_datadir, corruptions=['gaussian_noise', 'glass_blur', 'snow'], severities=[1,1,1], batch_size=256, seed=1):  
    # Load SVHN test dataset
    svhn_dataset = np.load(os.path.join(svhn_datadir, 'feature', 'test_inceptionv3_feat.npy'))
    svhn_dataset = normalize(svhn_dataset, feat_norm_type='standard')

    # Test dataset
    test_dataset = []
    num_ood_samples = CIFAR10_NUM_TEST_SAMPLES // 2

    # Random indicies
    np.random.seed(seed)
    cifar_idx = np.random.choice(CIFAR10_NUM_TEST_SAMPLES, num_ood_samples)
    np.random.seed(seed)
    svhn_idx = np.random.choice(svhn_dataset.shape[0], num_ood_samples)

    # Save datasets
    for corruption,severity in zip(corruptions,severities):
        loaded_cifar_dataset = np.load(os.path.join(cifar10c_datadir, 'test', 'feature', f'{corruption}_severity_{severity}_inceptionv3_feat.npy'))
        loaded_cifar_dataset = normalize(loaded_cifar_dataset, feat_norm_type='standard')
            
        test_dataset.append(np.concatenate([loaded_cifar_dataset[cifar_idx],svhn_dataset[svhn_idx]], axis=0))
    test_label = np.zeros(test_dataset[0].shape[0], dtype=np.int16)
    test_label[num_ood_samples:] = 1
    test_size = len(test_label)

    # Shuffle
    np.random.seed(seed)
    random_idx = np.arange(test_size)
    np.random.shuffle(random_idx)
    for i in range(len(corruptions)):
        test_dataset[i] = test_dataset[i][random_idx]
    test_label = test_label[random_idx]

    # Create datasets
    test_dataset.append(test_label)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(test_dataset)).batch(batch_size)

    return test_set

def cifar100_ood_dataset(cifar10c_datadir, cifar100_datadir, corruptions=['gaussian_noise', 'glass_blur', 'snow'], severities=[1,1,1], batch_size=256, seed=1):

    # Load SVHN test dataset
    cifar100_dataset = np.load(os.path.join(cifar100_datadir, 'feature', 'test_inceptionv3_feat.npy'))
    cifar100_dataset = normalize(cifar100_dataset, feat_norm_type='standard')

    # Test dataset
    test_dataset = []
    num_ood_samples = CIFAR10_NUM_TEST_SAMPLES // 2

    # Random indicies
    np.random.seed(seed)
    cifar_idx = np.random.choice(CIFAR10_NUM_TEST_SAMPLES, num_ood_samples)
    np.random.seed(seed)
    cifar100_idx = np.random.choice(cifar100_dataset.shape[0], num_ood_samples)

    # Save datasets
    for corruption,severity in zip(corruptions,severities):
        loaded_cifar_dataset = np.load(os.path.join(cifar10c_datadir, 'test', 'feature', f'{corruption}_severity_{severity}_inceptionv3_feat.npy'))
        loaded_cifar_dataset = normalize(loaded_cifar_dataset, feat_norm_type='standard')
            
        test_dataset.append(np.concatenate([loaded_cifar_dataset[cifar_idx],cifar100_dataset[cifar100_idx]], axis=0))
    test_label = np.zeros(test_dataset[0].shape[0], dtype=np.int16)
    test_label[num_ood_samples:] = 1
    test_size = len(test_label)

    # Shuffle
    np.random.seed(seed)
    random_idx = np.arange(test_size)
    np.random.shuffle(random_idx)
    for i in range(len(corruptions)):
        test_dataset[i] = test_dataset[i][random_idx]
    test_label = test_label[random_idx]

    # Create datasets
    test_dataset.append(test_label)
    test_set = tf.data.Dataset.from_tensor_slices(tuple(test_dataset)).batch(batch_size)

    return test_set
    