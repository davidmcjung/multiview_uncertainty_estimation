import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
import random
import os
import time
from tqdm import tqdm
import robustness_metrics as rm
import itertools
from dataset import get_datasets
from models import get_model
import json

# Params for datasets
flags.DEFINE_string('dataset', 'Handwritten', '[Handwritten, CUB, Caltech101, HMDB, PIE, Scene15, CIFAR10-C]')
flags.DEFINE_string('data_dir', './data', 'Parent directory containing data.')
flags.DEFINE_list('corruption_severities', '1,3,5', 'Corruption severities for CIFAR10-C.')
flags.DEFINE_string('feat_norm_type', 'min_max', '[min_max,standard]')

# Params for experiments
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('epochs', 500, 'The total number of epochs.')
flags.DEFINE_float('lr', 1e-2, 'Initial learning rate.')
flags.DEFINE_float('end_lr', 1e-4, 'Final learning rate.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('n_thresh', 10000, 'Number of thresholds to test for AUROC.')
flags.DEFINE_float('clipnorm', None, 'Clipnorm for optimizer.')
flags.DEFINE_string('lr_scheduler', 'linear', '[linear,piecewise]')
flags.DEFINE_list('lr_epochs', [10,20], 'Epochs to change learning rate for piecewise scheduler.')
flags.DEFINE_list('lr_list', [1e-3,1e-4,1e-5], 'Learning rates for piecewise schduler.')

# Params for models
flags.DEFINE_string('model', 'MGP', '[MGP,MNP]')

# Params for MGP
flags.DEFINE_integer('num_inducing_points', 100, 'Number of inducing points.')
flags.DEFINE_float('a_eps', 1e-7, 'Alpha epsilon.')
flags.DEFINE_float('beta', 1.0, 'Beta.')
flags.DEFINE_string('train_weighting', 'poe', 'Weighting scheme for combining posteriors during training.')
flags.DEFINE_string('test_weighting', 'entropy', 'Weighting scheme for combining posteriors during testing.')
flags.DEFINE_bool('init_lengthscale_data', True, 'Wheter to intialise kernel lengthscale from data.')

# Params for MNP
flags.DEFINE_integer('r_dim', 512, 'Latent space dimension.')
flags.DEFINE_integer('n_context_points', 200, 'Total number of context points.')
flags.DEFINE_float('l2_mlp', 1., 'L2-regularization for MLP in MNP.')
flags.DEFINE_float('l2_lengthscale', 1., 'L2-regularization for lengthscale in RBF attention.')
flags.DEFINE_string('mlp_norm_type', 'layer_norm', 'Normalization layer type for MLP in MNP [layer_norm, batch_norm]')
flags.DEFINE_string('attention', 'adaptive_rbf', 'Attention type [adaptive_rbf,dot,scaledot]')
flags.DEFINE_string('attn_norm_type', 'sparsemax', 'Attention normalization type [sparsemax,softmax]')
flags.DEFINE_integer('n_z_samples', 5, 'Number of samples used for sampling latent distributions.') 
flags.DEFINE_bool('rand_mask', True, 'Wheter to randomly mask some modalities during training.')
flags.DEFINE_float('temp', 0.01, 'Temperature scale in contrastive learning.')
flags.DEFINE_integer('update_freq', 5, 'Update frequency for context memory.')
flags.DEFINE_integer('warmup_epochs', 100, 'The number of warm-up epochs.')
flags.DEFINE_float('warmup_lr', 1e-3, 'Initial learning rate for warm-up.')
flags.DEFINE_string('update_loss', 'mse', 'Loss used for updating context memory [mse,ce].')
flags.DEFINE_float('init_lengthscale', 10., 'Initial lengthscale.')
flags.DEFINE_string('context_memory_init', 'data', 'Whether to initialise context memory from data.')

FLAGS = flags.FLAGS

def main(argv):
    del argv  # unused arg
    
    # Set output directories
    output_dir = os.path.join('./experiments/', FLAGS.dataset, FLAGS.model, f'seed_{FLAGS.seed}')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    tf.io.gfile.makedirs(output_dir)
    tf.io.gfile.makedirs(tensorboard_dir)
    json_dict = json.dumps(FLAGS.flag_values_dict(), indent=4)
    with open(os.path.join(output_dir, 'flags.json'), 'w') as f:
        f.write(json_dict)

    # Set random seeds
    logging.info(f'Random seed: {FLAGS.seed}')
    tf.random.set_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Get datasets
    datasets = get_datasets(FLAGS)
    steps_per_epoch = datasets['train_size'] // FLAGS.batch_size
    steps_per_eval = datasets['test_size'] // FLAGS.batch_size if FLAGS.batch_size <= datasets['test_size'] else 1
    num_views = len(datasets['feat_dims'])    
    
    # Set logger
    writer = tf.summary.create_file_writer(tensorboard_dir)
    
    # Get model
    logging.info('Building model')
    model = get_model(opts=FLAGS, datasets=datasets)
    if FLAGS.lr_scheduler == 'linear':
        scheduler = tf.keras.optimizers.schedules.PolynomialDecay(FLAGS.lr, FLAGS.epochs*steps_per_epoch, FLAGS.end_lr)
    elif FLAGS.lr_scheduler == 'piecewise':
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay([int(epoch)*steps_per_epoch for epoch in FLAGS.lr_epochs], [float(lr) for lr in FLAGS.lr_list])
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler, clipnorm=FLAGS.clipnorm)

    warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(FLAGS.warmup_lr, FLAGS.warmup_epochs*steps_per_epoch, FLAGS.warmup_lr*1e-1)
    warmup_optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_scheduler)
    warmup_metrics = {'warmup/loss': tf.keras.metrics.Mean()}

    # Metrics for training
    during_training_metrics = {
        'train/accuracy_combined': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece_combined': rm.metrics.ExpectedCalibrationError(),
        'train/ms_per_epoch': tf.keras.metrics.Mean(),

        'test/loss': tf.keras.metrics.Mean(),
        'test/accuracy_combined': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece_combined': rm.metrics.ExpectedCalibrationError(),
        'test/ms_per_epoch': tf.keras.metrics.Mean(),
    }
    for v_num in range(num_views):
        during_training_metrics.update({
            f'test/accuracy_view{v_num+1}': tf.keras.metrics.SparseCategoricalAccuracy(),
            f'test/ece_view{v_num+1}': rm.metrics.ExpectedCalibrationError()})
        
    @tf.function
    def warmup_step(inputs):
        # Required only for MNP
        (*X, label) = inputs
        with tf.GradientTape() as tape:
            train_vars = [tuple(model.experts[i].trainable_variables) for i in range(num_views)]
            train_variables = sum(train_vars,())
            tape.watch(train_variables)
            loss = 0.0
            for m in range(num_views):
                loss += model.experts[m].contrastive_loss(X[m], label)
        gradients = tape.gradient(loss, train_variables)
        warmup_optimizer.apply_gradients(zip(gradients, train_variables))
        warmup_metrics['warmup/loss'].update_state(loss)
        return loss
    
    @tf.function
    def train_step(inputs, metrics):
        (*X, label) = inputs
        with tf.GradientTape() as tape:
            train_vars = [tuple(model.experts[i].trainable_variables) for i in range(num_views)]
            train_variables = sum(train_vars,())
            tape.watch(train_variables)

            if FLAGS.model == 'MGP':
                _, _, p_combined, _, loss = model(X, label, FLAGS.train_weighting)
            elif FLAGS.model == 'MNP':
                p_y_m_list, _, p_combined, _, loss, attn_list = model(X, label, training=True)

        if epoch % FLAGS.update_freq == 0 and FLAGS.model == 'MNP':
            model.update_context_memory(X, label, p_y_m_list, attn_list, method=FLAGS.update_loss)

        gradients = tape.gradient(loss, train_variables)
        optimizer.apply_gradients(zip(gradients, train_variables))

        metrics['train/loss'].update_state(loss)
        metrics['train/ece_combined'].add_batch(p_combined, label=label)
        metrics['train/accuracy_combined'].update_state(label, p_combined)
        return loss

    @tf.function
    def test_step(inputs, metrics, dataset_name):
        (*X, label) = inputs
        if FLAGS.model == 'MGP':
            p_experts, var_experts, p_combined, var_combined, loss = model(X, label, FLAGS.test_weighting)
            uncertainty_experts = [tf.reduce_sum(var_experts[i], axis=1) for i in range(num_views)]
            uncertainty_combined = tf.reduce_sum(var_combined, axis=1)
        elif FLAGS.model == 'MNP':
            p_experts, uncertainty_experts, p_combined, uncertainty_combined, loss, _ = model(X, label, training=False)

        if 'test' in dataset_name:
            metrics[f'{dataset_name}/accuracy_combined'].update_state(label, p_combined)
            if dataset_name != 'noisy_test':
                metrics[f'{dataset_name}/loss'].update_state(loss) if f'{dataset_name}/loss' in metrics else None
                metrics[f'{dataset_name}/ece_combined'].add_batch(p_combined, label=label)
            for view in range(num_views):
                metrics[f'{dataset_name}/accuracy_view{view+1}'].update_state(label, p_experts[view])
                if dataset_name != 'noisy_test':
                    metrics[f'{dataset_name}/ece_view{view+1}'].add_batch(p_experts[view], label=label)
        elif 'ood' in dataset_name:
            metrics[f'{dataset_name}/auroc_combined'].update_state(label, uncertainty_combined)
            for view in range(num_views):
                metrics[f'{dataset_name}/auroc_view{view+1}'].update_state(label, uncertainty_experts[view])

        return loss
    
    if FLAGS.model == 'MNP':
        # Warmup lengthscales
        train_iterator = iter(datasets['train_dataset'])
        for epoch in range(FLAGS.warmup_epochs):
            with tqdm(total=steps_per_epoch, desc='Warmup Epoch #{}'.format(epoch+1)) as t:
                for step in range(steps_per_epoch):
                    train_loss = warmup_step(next(train_iterator))

                    t.set_postfix({'loss': train_loss.numpy()})
                    t.update(1)

            total_results = {name: metric.result() for name, metric in warmup_metrics.items()}
            with writer.as_default():
                for tag, val in total_results.items():
                    tf.summary.scalar(tag, val, step=epoch+1)        
            for metric in warmup_metrics.values():
                metric.reset_states()

    train_iterator = iter(datasets['train_dataset'])
    for epoch in range(FLAGS.epochs):
        train_start_time = time.time()
        # Training
        with tqdm(total=steps_per_epoch, desc='Train Epoch #{}'.format(epoch+1)) as t:
            for step in range(steps_per_epoch):
                train_loss = train_step(next(train_iterator), during_training_metrics)
            
                t.set_postfix({'loss': train_loss.numpy()})
                t.update(1)
        s_per_epoch = (time.time() - train_start_time) * 1e3
        during_training_metrics['train/ms_per_epoch'].update_state(s_per_epoch)
        # Testing
        test_iterator = iter(datasets['test_dataset'])
        test_start_time = time.time()
        with tqdm(total=steps_per_eval, desc='Testing') as t:
            for step in range(steps_per_eval):
                test_loss = test_step(next(test_iterator), during_training_metrics, 'test')
                
                t.set_postfix({'loss': test_loss.numpy()})
                t.update(1)
        s_per_epoch = (time.time() - test_start_time) * 1e3
        during_training_metrics['test/ms_per_epoch'].update_state(s_per_epoch)
        
        total_results = {name: metric.result() for name, metric in during_training_metrics.items()}
        total_results = {
            k: (list(v.values())[0] if isinstance(v, dict) else v) # ECE is in dictionary, need to change to scalar
            for k, v in total_results.items()}
        with writer.as_default():
            for tag, val in total_results.items():
                tf.summary.scalar(tag, val, step=epoch+1)

        logging.info('Train Loss: %.4f', total_results['train/loss'])
        logging.info('Test Accuracy (combined): %.2f%%', total_results['test/accuracy_combined'] * 100)
        logging.info('Test ECE (combined): %.4f',total_results['test/ece_combined'])

        for metric in during_training_metrics.values():
            metric.reset_states()

    # Metrics for evaluations after training
    if FLAGS.dataset == 'CIFAR10-C':
        after_training_metrics = {
            'ood_svhn/auroc_combined': tf.keras.metrics.AUC(FLAGS.n_thresh, curve='ROC', from_logits=True),
            'ood_cifar100/auroc_combined': tf.keras.metrics.AUC(FLAGS.n_thresh, curve='ROC', from_logits=True)}
    else:
        after_training_metrics = {
            'noisy_test/accuracy_combined': tf.keras.metrics.SparseCategoricalAccuracy()}
    for v_num in range(num_views):
        if FLAGS.dataset == 'CIFAR10-C':
            after_training_metrics.update({
                f'ood_svhn/auroc_view{v_num+1}': tf.keras.metrics.AUC(FLAGS.n_thresh, curve='ROC', from_logits=True),
                f'ood_cifar100/auroc_view{v_num+1}': tf.keras.metrics.AUC(FLAGS.n_thresh, curve='ROC', from_logits=True)})
        else:
            after_training_metrics.update({
                f'noisy_test/accuracy_view{v_num+1}': tf.keras.metrics.SparseCategoricalAccuracy()})
            
    if FLAGS.dataset == 'CIFAR10-C':
        # OOD test
        svhn_ood_iterator = iter(datasets['svhn_ood_dataset'])
        with tqdm(total=steps_per_eval, desc='OOD Test (SVHN)') as t:
            for step in range(steps_per_eval):
                _ = test_step(next(svhn_ood_iterator), after_training_metrics, 'ood_svhn')

                t.update(1)

        cifar100_ood_iterator = iter(datasets['cifar100_ood_dataset'])
        with tqdm(total=steps_per_eval, desc='OOD Test (CIFAR100)') as t:
            for step in range(steps_per_eval):
                _ = test_step(next(cifar100_ood_iterator), after_training_metrics, 'ood_cifar100')

                t.update(1)
        
        total_results = {name: metric.result() for name, metric in after_training_metrics.items()}
        with writer.as_default():
            for tag, val in total_results.items():
                tf.summary.scalar(tag, val, step=epoch)
        logging.info('OOD (SVHN): %.4f',total_results['ood_svhn/auroc_combined'])
        logging.info('OOD (CIFAR100): %.4f',total_results['ood_cifar100/auroc_combined'])

    else:
        # Noisy Test
        num_noisy_test_runs = 10
        noise_std_list = [0.0]
        noise_std_list.extend(np.logspace(-2,1,num_noisy_test_runs).tolist())
        num_noisy_views = num_views // 2
        acc_results = []
        for i,noise_std in enumerate(noise_std_list):
            for noisy_sets in itertools.combinations(list(range(num_views)), num_noisy_views):
                noisy_sets = list(noisy_sets)
                noisy_test_dataset = datasets['noisy_dataset_fn'](
                    os.path.join(FLAGS.data_dir, FLAGS.dataset),
                    noisy_sets=noisy_sets, 
                    noise_std=noise_std,
                    batch_size=FLAGS.batch_size, 
                    seed=FLAGS.seed,
                    feat_norm_type=FLAGS.feat_norm_type)
                noisy_test_iterator = iter(noisy_test_dataset)
                with tqdm(total=steps_per_eval, desc='Noisy dataset testing (noise std: {:.4f})'.format(noise_std)) as t:
                    for step in range(steps_per_eval):
                        _ = test_step(next(noisy_test_iterator), after_training_metrics, 'noisy_test')
                        
                        t.update(1)

            total_results = {name: metric.result() for name, metric in after_training_metrics.items()}
            total_results.update({'noisy_test/noise_std': noise_std})

            with writer.as_default():
                for tag, val in total_results.items():
                    tf.summary.scalar(tag, val, step=i)
            acc_results.append(total_results['noisy_test/accuracy_combined'])
            for metric in after_training_metrics.values():
                metric.reset_states()
        for noise_std, acc in zip(noise_std_list, acc_results):
            logging.info('Noisy Test Accuracy (noise std: %.2f): %.2f%%', noise_std, acc * 100)

if __name__ == '__main__':
  app.run(main)