import os
from absl import logging

def get_datasets(opts):
    if opts.dataset == 'Handwritten':
        from .handwritten import train_test_datasets, noisy_dataset
        feat_dims = [240, 76, 216, 47, 64, 6]
        num_classes = 10
    elif opts.dataset == 'Scene15':
        from .scene15 import train_test_datasets, noisy_dataset
        feat_dims = [20, 59, 40]
        num_classes = 15
    elif opts.dataset == 'CUB':
        from .cub import train_test_datasets, noisy_dataset
        feat_dims = [1024, 300]
        num_classes = 10
    elif opts.dataset == 'Caltech101':
        from .caltech101 import train_test_datasets, noisy_dataset
        feat_dims = [4096, 4096]
        num_classes = 101
    elif opts.dataset == 'PIE':
        from .pie import train_test_datasets, noisy_dataset
        feat_dims = [484, 256, 279]
        num_classes = 68
    elif opts.dataset == 'HMDB':
        from .hmdb import train_test_datasets, noisy_dataset
        feat_dims = [1000, 1000]
        num_classes = 51
    
    if opts.dataset == 'CIFAR10-C':
        from .cifar10c import train_test_datasets, svhn_ood_dataset, cifar100_ood_dataset
        corruption_types = ['gaussian_noise','shot_noise','impulse_noise']
        train_dataset, test_dataset, train_size, test_size, lengthscales_init =  train_test_datasets(
                                                                    os.path.join(opts.data_dir, opts.dataset), 
                                                                    corruptions=corruption_types, 
                                                                    severities=opts.corruption_severities, 
                                                                    batch_size=opts.batch_size, 
                                                                    seed=opts.seed)
        cifar10c_svhn = svhn_ood_dataset(
                                os.path.join(opts.data_dir, opts.dataset),
                                os.path.join(opts.data_dir, 'SVHN'),
                                corruptions=corruption_types, 
                                severities=opts.corruption_severities, 
                                batch_size=opts.batch_size, 
                                seed=opts.seed)
        cifar10c_cifar100 = cifar100_ood_dataset(
                                os.path.join(opts.data_dir, opts.dataset),
                                os.path.join(opts.data_dir, 'CIFAR100/cifar-100-python'),
                                corruptions=corruption_types, 
                                severities=opts.corruption_severities, 
                                batch_size=opts.batch_size, 
                                seed=opts.seed)
        feat_dims = [2048 for _ in range(len(corruption_types))]
        num_classes = 10
        datasets = {'train_dataset': train_dataset, 
                    'train_size': train_size,
                    'test_dataset': test_dataset,
                    'test_size': test_size,
                    'feat_dims': feat_dims,
                    'num_classes': num_classes,
                    'svhn_ood_dataset': cifar10c_svhn,
                    'cifar100_ood_dataset': cifar10c_cifar100,
                    'lengthscales_init': lengthscales_init}
        return datasets
    elif feat_dims and num_classes:
        train_dataset, test_dataset, train_size, test_size, lengthscales_init = train_test_datasets(
                                                                    os.path.join(opts.data_dir, opts.dataset),
                                                                    batch_size=opts.batch_size, 
                                                                    seed=opts.seed, 
                                                                    feat_norm_type=opts.feat_norm_type)
        datasets = {'train_dataset': train_dataset, 
                    'train_size': train_size,
                    'test_dataset': test_dataset,
                    'noisy_dataset_fn': noisy_dataset,
                    'test_size': test_size,
                    'feat_dims': feat_dims,
                    'num_classes': num_classes,
                    'lengthscales_init': lengthscales_init}
        return datasets
    else:
        logging.fatal('Unknown dataset.')