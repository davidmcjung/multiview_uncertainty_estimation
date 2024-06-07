import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from functools import partial
from .utils import merge_flat_input, MultivariateNormalDiag, get_attender, powerset
from .MLP import MLP


class UnimodalModule(tf.keras.Model):
    def __init__(
            self, 
            x_dim, 
            y_dim, 
            r_dim, 
            context_memory_init=None, 
            n_context_points=None, 
            l2=None, 
            mlp_norm_type='batch_norm',
            attention="rbf",
            attention_kwargs={},
            temp=0.01,
            l2_lengthscale=1.,
            init_lengthscale=10.):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.n_context_points = n_context_points

        self.x_encoder = layers.Dense(
            self.r_dim, 
            activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            use_bias=True,
            input_shape=(self.x_dim,),
            kernel_regularizer=tf.keras.regularizers.L2(l2), 
            )
        base_encoder = merge_flat_input(partial(MLP, n_hidden_layers=0, hidden_size=self.r_dim), is_sum_merge=True)
        self.r_encoder = base_encoder(self.r_dim, self.y_dim, self.r_dim, l2=l2, is_out_bn=True, norm_type=mlp_norm_type)
        self.s_encoder = base_encoder(self.r_dim, self.y_dim, self.r_dim, l2=l2, is_out_bn=True, norm_type=mlp_norm_type)

        self.scale_transformer=lambda x: 0.01 + 0.99 *  tf.math.softplus(x)
        self.init_context_memory(None)
        attention_kwargs.update({'init_lengthscale': init_lengthscale})
        self.attender = get_attender(attention, self.r_dim, self.r_dim, self.r_dim, **attention_kwargs)

        self.temp = temp
        self.l2_lengthscale = l2_lengthscale

    def init_context_memory(self, context_memory_init):
        if context_memory_init is None:
            init_x = tf.keras.initializers.RandomUniform(-1, 1)
            self.CmX = tf.Variable(init_x(shape=(self.y_dim, self.n_context_points, self.x_dim)), trainable=True, name='CmX')
            cmy = []
            for c in range(self.y_dim):
                cmy.append(tf.stack([tf.one_hot(tf.convert_to_tensor(c), self.y_dim) for _ in range(self.n_context_points)],axis=0))
            
            self.CmY = tf.Variable(tf.stack(cmy, axis=0), trainable=False, name='CmY')
        else:
            cmx, cmy = [], []
            for c in range(self.y_dim):
                true_x = context_memory_init[0][context_memory_init[1]==c]
                idx = np.random.choice(true_x.shape[0], self.n_context_points)
                cmx.append(tf.gather(true_x,idx,axis=0,batch_dims=1))
                cmy.append(tf.stack([tf.one_hot(tf.convert_to_tensor(c), self.y_dim) for _ in range(self.n_context_points)],axis=0))
            self.CmX = tf.Variable(tf.cast(tf.stack(cmx, axis=0),tf.float32), trainable=True, name='ZmX')
            self.CmY = tf.Variable(tf.cast(tf.stack(cmy, axis=0),tf.float32), trainable=False, name='ZmY')
    
    def encode_inducing_points(self,):
        return tf.reshape(self.CmX, [-1,self.x_dim]), tf.reshape(self.CmY,[-1,self.y_dim])
    
    def contrastive_loss(self, x_target, y_target):
        x_target = tf.cast(x_target, dtype=tf.float32)
        n_samples = tf.shape(x_target)[0]
        y_target = tf.expand_dims(y_target,-1)
        mask = tf.cast(tf.equal(y_target, tf.transpose(y_target)),tf.float32)
        x_target = self.x_encoder(x_target) / self.attender.lengthscale
        
        r2 = tf.reduce_sum((tf.expand_dims(x_target,0)-tf.expand_dims(x_target,1))**2,axis=-1)
        logits = tf.exp(-0.5*r2) * self.attender.variance

        logits = logits / self.temp

        logits_mask = tf.ones_like(mask)
        indices = tf.stack([tf.range(n_samples),tf.range(n_samples)],-1)
        updates = tf.ones(n_samples)
        logits_mask = tf.tensor_scatter_nd_sub(logits_mask,indices,updates)

        mask = mask * logits_mask
        log_prob = logits - tf.math.reduce_logsumexp(logits * logits_mask, axis=-1, keepdims=True)

        # compute mean of log-likelihood over positive
        idx_non_empty = tf.reduce_sum(mask, axis=-1)>0
        mean_log_prob_pos = tf.reduce_sum(mask[idx_non_empty] * log_prob[idx_non_empty], axis=-1) / tf.reduce_sum(mask[idx_non_empty], axis=-1)

        # loss
        loss = tf.reduce_mean(-mean_log_prob_pos) + self.l2_lengthscale * tf.nn.l2_loss(self.attender.lengthscale)
        return loss
    
    def call(self, X_trgt, training=True):
        X_cntxt, Y_cntxt = self.encode_inducing_points()
        X_cntxt = self.x_encoder(X_cntxt)
        X_trgt = self.x_encoder(X_trgt)

        r_m = self.r_encoder(X_cntxt, Y_cntxt, training=training)
        s_m = self.s_encoder(X_cntxt, Y_cntxt, training=training)
        s_m = self.scale_transformer(s_m)

        # We use the same encoders for r_encoder=u_encoder and s_encoder=q_encoder 
        # but separate encoders can be used
        u_m = r_m
        q_m = s_m
        u_m = tf.tile(tf.reduce_mean(u_m, 0, keepdims=True), [X_trgt.shape[0], 1])
        q_m = tf.tile(tf.reduce_mean(q_m, 0, keepdims=True), [X_trgt.shape[0], 1])

        r_star_m, attn_weight_m = self.attender(X_cntxt, X_trgt, r_m)
        s_star_m = tf.matmul(attn_weight_m, s_m)

        return r_star_m, s_star_m, u_m, q_m, attn_weight_m

class MNP(tf.keras.Model):
    def __init__(self, 
        x_dim_list, 
        y_dim, 
        context_memory_init_list,
        r_dim,
        n_context_points,
        l2_mlp,
        l2_lengthscale,
        mlp_norm_type,
        attention='adaptive_rbf',
        attention_kwargs={},
        scale_transformer=lambda y_scale: 0.01 + 0.99 *  tf.math.softplus(y_scale),
        n_z_samples=100,
        rand_mask=True,
        temp=0.01,
        init_lengthscale=10.):
        super().__init__()

        self.num_views = len(x_dim_list)
        self.y_dim = y_dim
        self.n_z_samples = n_z_samples

        self.experts = []
        for view in range(self.num_views):
            self.experts.append(
                UnimodalModule(
                    x_dim=x_dim_list[view],
                    y_dim=y_dim,
                    r_dim=r_dim,
                    context_memory_init=(context_memory_init_list[view],context_memory_init_list[-1]) if context_memory_init_list is not None else None,
                    n_context_points=n_context_points,
                    l2=l2_mlp,
                    mlp_norm_type=mlp_norm_type,
                    attention=attention,
                    attention_kwargs=attention_kwargs,
                    temp=temp,
                    l2_lengthscale=l2_lengthscale,
                    init_lengthscale=init_lengthscale
                )
            )
        
        self.decoder = MLP(
            r_dim, 
            2*y_dim, 
            n_hidden_layers=0,
            hidden_size=r_dim//2,
            is_bias=True,
            l2=l2_mlp, 
            is_bn=True, 
            is_out_bn=False, 
            norm_type=mlp_norm_type)
        self.PredictiveDistribution = MultivariateNormalDiag
        self.scale_transformer = scale_transformer

        self.rand_mask = rand_mask
        # Create subset masking for all possible 
        # combinations of choosing modalities.
        # e.g., For two modalities,
        #    [[1., 0.],
        #    [0., 1.],
        #    [1., 1.]]
        self.subset_mask = tf.Variable(powerset(self.num_views), dtype=tf.float32, trainable=False)

    def call(self, X_trgt_list, Y_trgt, training=True):
        Y_trgt_OH = tf.one_hot(tf.cast(Y_trgt, dtype=tf.int32), self.y_dim)
        X_trgt_list = [tf.cast(X_trgt, dtype=tf.float32) for X_trgt  in X_trgt_list]
        # Obtain r_star_m, s_star_m, u_m, and q_m from every modality
        r_star_list, s_star_list, u_list, q_list, attn_list = [], [], [], [], []
        for module_m, X_trgt in zip(self.experts, X_trgt_list):
            r_star_m, s_star_m, u_m, q_m, attn_weight_m = module_m(X_trgt, training=training)

            r_star_list.append(r_star_m)
            s_star_list.append(s_star_m)
            attn_list.append(attn_weight_m)
            u_list.append(u_m)
            q_list.append(q_m)
        r_star_list = tf.stack(r_star_list, axis=0)
        s_star_list = tf.stack(s_star_list, axis=0)
        u_list = tf.stack(u_list, 0) 
        q_list = tf.stack(q_list, 0) 

        # MBA Module
        if self.rand_mask and training:
            rnd_choices = np.random.choice(range(self.num_views, self.subset_mask.shape[0]), Y_trgt.shape[0])
            mask = tf.gather(self.subset_mask, rnd_choices, axis=0, batch_dims=1)
        else:
            mask = tf.tile(tf.expand_dims(self.subset_mask[-1], 0), [Y_trgt.shape[0], 1]) # same as ones
        multiplier = tf.tile(tf.expand_dims(mask, axis=1),[1, r_star_list.shape[-1], 1])
        multiplier = tf.transpose(multiplier, perm=[2,0,1])

        var_z = 1 / tf.reduce_sum((1/s_star_list + 1/q_list) * multiplier, axis=0)
        mu_z = var_z * tf.reduce_sum((r_star_list/s_star_list + u_list/q_list) * multiplier, axis=0)

        p_z = self.PredictiveDistribution(mu_z, var_z)

        # Sampling
        z = p_z.sample(self.n_z_samples)

        # Decode
        combined_pyCc = self.decode(z, training=training)
        merged_logits = combined_pyCc.sample(self.n_z_samples)
        merged_probs = tf.nn.softmax(merged_logits, axis=-1)
        if len(merged_logits.shape)==3:
            p_combined = tf.reduce_mean(merged_probs, axis=0)
            Y_trgt_exp = tf.tile(tf.expand_dims(Y_trgt_OH, 0), [merged_logits.shape[0],1,1])
            combined_entr = tf.reduce_mean(-tf.reduce_sum(merged_probs*tf.math.log(merged_probs),axis=-1),0)
        elif len(merged_logits.shape)==4:
            p_combined = tf.reduce_mean(tf.reduce_mean(merged_probs, axis=0),0)
            Y_trgt_exp = tf.tile(tf.expand_dims(tf.expand_dims(Y_trgt_OH, 0),0), [merged_logits.shape[0],merged_logits.shape[1],1,1])
            combined_entr = tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(merged_probs*tf.math.log(merged_probs),axis=-1),0),0)

        ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(Y_trgt_exp, merged_logits) 
        loss = tf.reduce_mean(ce)

        # Unimodal predictions and losses
        z_m_list = []
        for m in range(self.num_views):
            var_view = 1 / (1/s_star_list[m] + 1/q_list[m])
            mu_view = var_view * (r_star_list[m]/s_star_list[m] + u_list[m]/q_list[m])
            z_m_list.append(self.PredictiveDistribution(mu_view, var_view))
        f_y_m_list = [self.decode(z_m_list[view].sample(self.n_z_samples)) for view in range(self.num_views)]
        p_y_m_list, entr_list = [], []
        for f_y_m in f_y_m_list:
            ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(Y_trgt_exp, f_y_m.sample(self.n_z_samples))
            loss += tf.reduce_mean(tf.reduce_mean(ce, 0)) / self.num_views
            prob_ = tf.nn.softmax(f_y_m.sample(self.n_z_samples),axis=-1) 
            if len(prob_.shape) == 3:
                prob = tf.reduce_mean(prob_, axis=0)
                entr = tf.reduce_mean(-tf.reduce_sum(prob_*tf.math.log(prob_), axis=-1),0)
            elif len(prob_.shape) == 4:
                prob = tf.reduce_mean(tf.reduce_mean(prob_, axis=0),0)
                entr = tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(prob_*tf.math.log(prob_), axis=-1),0),0)

            p_y_m_list.append(prob)
            entr_list.append(entr)


        # Contrastive learning
        contrastive_loss = 0.0
        if training:
            for m in range(self.num_views):
                contrastive_loss += self.experts[m].contrastive_loss(X_trgt_list[m], Y_trgt)
        loss += contrastive_loss

        loss += tf.add_n(self.losses)
        return p_y_m_list, entr_list, p_combined, combined_entr, loss, attn_list
    
    def decode(self, R_trgt, training=True):
        p_y_suffstat = self.decoder(R_trgt, training=training)
        p_y_loc, p_y_scale = tf.split(p_y_suffstat, 2, axis=-1)
        p_y_scale = self.scale_transformer(p_y_scale)
        p_yCc = self.PredictiveDistribution(p_y_loc, p_y_scale)
        return p_yCc
    
    def update_context_memory(self, trgt_x_list, trgt_y, prob_list, attn_list, method='mse'):
        trgt_y = tf.one_hot(tf.cast(trgt_y, dtype=tf.int32), self.y_dim)
        for view in range(len(trgt_x_list)):
            trgt_x = tf.cast(trgt_x_list[view], dtype=tf.float32)
            prob = prob_list[view]
            attn = tf.reduce_sum(attn_list[view],0) 
            attn = tf.reshape(attn,[self.y_dim,-1]) 
            for c in range(self.y_dim):
                if tf.reduce_sum(trgt_y[:,c])>0:
                    temp_trgt_x = trgt_x[trgt_y[:,c]==1]
                    temp_prob_list = prob[trgt_y[:,c]==1]
                    temp_trgt_y = trgt_y[trgt_y[:,c]==1]
                    attn_c = attn[c] 
                    
                    if method == 'mse':
                        highest_idx = tf.math.argmax(tf.reduce_sum((temp_prob_list-temp_trgt_y)**2,axis=-1), axis=0)
                    else:
                        nll = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')(temp_trgt_y, temp_prob_list)
                        highest_idx = tf.math.argmax(nll, axis=0)

                    seletecd_Cm = tf.squeeze(tf.math.argmin(attn_c))

                    self.experts[view].CmX[c,seletecd_Cm].assign(temp_trgt_x[highest_idx])