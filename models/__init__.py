from .MGP import MGP
from .MNP import MNP
import tensorflow as tf

def get_model(opts, datasets):
    if opts.model == 'MGP':
        model = MGP(
            C=datasets['num_classes'], 
            num_inducing_points=opts.num_inducing_points, 
            num_experts=len(datasets['feat_dims']), 
            a_eps=opts.a_eps,
            beta=opts.beta)
        model.initialize(
            X=next(iter(datasets['train_dataset'].unbatch().batch(2000).take(1)))[:len(datasets['feat_dims'])], 
            num_data=datasets['train_size'],
            lengthscales_init=datasets['lengthscales_init'],
            init_lengthscale_data=opts.init_lengthscale_data)
    elif opts.model == 'MNP':
        model = MNP(
            x_dim_list=datasets['feat_dims'],
            y_dim=datasets['num_classes'],
            context_memory_init_list=next(iter(datasets['train_dataset'].unbatch().batch(2000).take(1))),
            r_dim=opts.r_dim,
            n_context_points=opts.n_context_points//datasets['num_classes'],
            l2_mlp=opts.l2_mlp,
            l2_lengthscale=opts.l2_lengthscale,
            mlp_norm_type=opts.mlp_norm_type,
            attention_kwargs=dict(norm_type=opts.attn_norm_type),
            n_z_samples=opts.n_z_samples,
            rand_mask=opts.rand_mask,
            temp=opts.temp
        )
        model([tf.zeros([10,feat_dim]) for feat_dim in datasets['feat_dims']],
              tf.zeros(10),
              False)
    return model