import numpy as np
import tensorflow as tf
from .ResNetFeat import ResNetFeat


def KL(alpha ,K):
    beta=tf.constant(np.ones((1,K)),dtype=tf.float64)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
    
    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    return kl

 
def ce_loss(p, alpha, global_step, annealing_step, c=10):
    p = tf.cast(p, dtype=tf.int32)
    alpha = tf.cast(alpha, dtype=tf.float64)
    S = tf.math.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1
    label = tf.cast(tf.one_hot(p, depth=c), dtype=tf.float64)
    A = tf.math.reduce_sum(label * (tf.math.digamma(S) - tf.math.digamma(alpha)), axis=1, keepdims=True)

    annealing_coef = tf.math.minimum(tf.cast(1.0, dtype=tf.float64), global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * tf.cast(KL(alp, c), dtype=tf.float64)

    return tf.reduce_sum(tf.cast(A, dtype=tf.float64) + B)

class TMC(tf.keras.Model):
    def __init__(self,C, num_experts=2, lambda_epochs=10, resnet_layers=3, resnet_hidden=128):
        super().__init__()
        self.C = C
        self.num_experts = num_experts
        self.lambda_epochs = lambda_epochs
        self.loss_fn = ce_loss
        self.experts = []
        for i in range(self.num_experts):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input((2)))
            model.add(ResNetFeat(resnet_layers, resnet_hidden))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(
                C, 
                activation=None, 
                kernel_regularizer=tf.keras.regularizers.L2(1e-4), 
                bias_regularizer=tf.keras.regularizers.L2(1e-4)))
            model.add(tf.keras.layers.Activation('softplus'))
            self.experts.append(model)

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2 # alpha = evidence + 1 (batch_size X num_classes)
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = tf.math.reduce_sum(alpha[v], axis=1, keepdims=True) # (batch_size X 1)
                E[v] = alpha[v]-1 # (batch_size X num_classes)
                b[v] = E[v]/tf.repeat(S[v], repeats=E[v].shape[1], axis=1) # S[v].expand(E[v].shape) = batch_size X num_classes (duplicates of S columns num_classes times) -> batch_size X num_classes 
                u[v] = self.C/S[v] # (batch_size X 1)
            # b^0 @ b^(0+1)
            bb = tf.linalg.matmul(tf.reshape(b[0],[-1, self.C, 1]), tf.reshape(b[1],[-1, 1, self.C])) # (batch_size X num_classes X 1, batch_size X 1 X num_classes) -> batch_size X num_classes X num_classes
            # b^0 * u^1
            uv1_expand = tf.repeat(u[1], repeats=b[0].shape[1], axis=1) # (batch_size X num_classes)
            bu = tf.math.multiply(b[0], uv1_expand) # (batch_size X num_classes)
            # b^1 * u^0
            uv_expand = tf.repeat(u[0], repeats=b[0].shape[1], axis=1) # (batch_size X num_classes)
            ub = tf.math.multiply(b[1], uv_expand) # (batch_size X num_classes)
            # calculate C
            bb_sum = tf.math.reduce_sum(bb, axis=(1, 2)) # bach_size X 1
            bb_diag = tf.math.reduce_sum(tf.linalg.diag_part(bb), axis=-1) # batch_size X 1
            C = bb_sum - bb_diag # bath_size X 1

            # calculate b^a
            b_a = (tf.math.multiply(b[0], b[1]) + bu + ub)/(tf.repeat(tf.reshape(1-C, [-1, 1]), repeats=b[0].shape[1], axis=1)) # ()/(batch_size X num_classes) - > batch_size X num_classes
            # calculate u^a
            u_a = tf.math.multiply(u[0], u[1])/(tf.repeat(tf.reshape(1-C, [-1, 1]), repeats=u[0].shape[1], axis=1)) # batch_size X 1

            # calculate new S
            S_a = self.C / u_a # batch_size X 1
            # calculate new e_k
            e_a = tf.math.multiply(b_a, tf.repeat(S_a, repeats=b[0].shape[1], axis=1)) # batch_size X num_classes
            alpha_a = e_a + 1
            return alpha_a, u_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a, u_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a, u_a

    def call(self, X, y, global_step, training):
         # step one
        evidence = self.infer(X, training)
        loss = 0
        alpha, u_s = dict(), dict()
        for v_num in range(len(X)):
            # step two
            alpha[v_num] = evidence[v_num] + 1
            u_s[v_num] = self.C/tf.math.reduce_sum(alpha[v_num], axis=1, keepdims=True) # (batch_size X 1)
            # step three
            loss += self.loss_fn(y, alpha[v_num], global_step, self.lambda_epochs, self.C)
            loss += tf.cast(tf.add_n(self.experts[v_num].losses), dtype=tf.float64)
        # step four
        alpha_a, u_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += self.loss_fn(y, alpha_a, global_step, self.lambda_epochs, self.C)
        loss = tf.math.reduce_mean(loss)
        return evidence, evidence_a, u_s, u_a, loss

    def infer(self, input, training):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.num_experts):
            evidence[v_num] = self.experts[v_num](input[v_num], training=training)
        return evidence