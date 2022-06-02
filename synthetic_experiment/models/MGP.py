import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float
from gpflow.models.svgp import SVGP
from gpflow import logdensities 
from gpflow.likelihoods import Likelihood
from .ResNetFeat import ResNetFeat

class GaussianHeteroskedastic(Likelihood):
    """
    Dirichlet-based Likelihood.

    Refernce:
      Code is adopted from https://github.com/dmilios/dirichletGPC

    Citation:
      [1] D. Milios, R. Camoriano, P. Michiardi, L. Rosasco, and M. Filippone. Dirichlet-based gaussian
          processes for large-scale calibrated classification. In S. Bengio, H. Wallach, H. Larochelle,
          K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Pro-
          cessing Systems, volume 31. Curran Associates, Inc., 2018.
    """
    def __init__(self, num_class):
        super().__init__(latent_dim=num_class, observation_dim=None)

    def _log_prob(self, F, Y, vars):
        return logdensities.gaussian(F, Y, vars)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F, vars):
        return tf.fill(tf.shape(F), tf.squeeze(vars))

    def _predict_mean_and_var(self, Fmu, Fvar, vars):
        return tf.identity(Fmu), Fvar + vars

    def _predict_density(self, Fmu, Fvar, Y, vars):
        return logdensities.gaussian(Fmu, Y, Fvar + vars)

    def _predict_log_density(self, Fmu, Fvar, Y):
        return None
    
    def _variational_expectations(self, Fmu, Fvar, Y, vars):
        return -0.5 * tf.cast(np.log(2 * np.pi), dtype=tf.float64) - 0.5 * tf.math.log(vars) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / vars
    def _negative_log_likelihood(self, Fmu, Fvar, Y):
        return -0.5 * tf.cast(np.log(2 * np.pi), dtype=tf.float64) - 0.5 * tf.math.log(Fvar) \
               - 0.5 * tf.square(Y - Fmu) / Fvar

def elbo(X, Y, model, s2_tilde=None):
    """
    This gives a variational bound (the evidence lower bound or ELBO) on
    the log marginal likelihood of each expert.
    
    Reference:
      Code is adopted from GPflow package.
    """
    X = tf.cast(X, dtype=default_float())
    kl = model.prior_kl()
    f_mean, f_var = model.predict_f(X, full_cov=False, full_output_cov=False)
    if s2_tilde != None:
        var_exp = model.likelihood._variational_expectations(f_mean, f_var, Y, s2_tilde)
    else:
        var_exp = model.likelihood.variational_expectations(f_mean, f_var, Y)
    if model.num_data is not None:
        num_data = tf.cast(model.num_data, kl.dtype)
        minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
        scale = num_data / minibatch_size
    else:
        scale = tf.cast(1.0, kl.dtype)
    return tf.reduce_sum(var_exp) * scale - kl, f_mean, f_var


def combine_dist(f_mean, f_var, weighting='entropy'):
    """
    This aggregates posterior distributions.
    """
    # Compute gamma matrix
    num_dists = len(f_mean)
    if weighting == 'entropy':
        weight_matrix = [-(0.5*tf.math.log(2*np.pi*f_var[i])+0.5) for i in range(num_dists)] 
    elif weighting == 'poe':
        weight_matrix = tf.ones(num_dists)
    # Normalize so that sum=1
    if weighting != 'poe':
        sum_weight = weight_matrix[0]
        for i in range(1, num_dists):
            sum_weight += weight_matrix[i]
        for i in range(num_dists):
            if weighting == 'parameter':
                weight_matrix[i].assign(weight_matrix[i]/sum_weight)
            else:
                weight_matrix[i] = weight_matrix[i]/sum_weight
    
    weight_matrix = tf.cast(weight_matrix, dtype=tf.float64)

    # Precision
    prec = [1/f_var[i] for i in range(num_dists)]
    # Calculation
    denominator = tf.math.multiply(prec[0], weight_matrix[0])
    for i in range(1, num_dists):
        denominator += tf.math.multiply(prec[i], weight_matrix[i])
    var = 1/denominator
    summation = tf.math.multiply(tf.math.multiply(prec[0], weight_matrix[0]), f_mean[0])
    for i in range(1, num_dists):
        summation += tf.math.multiply(tf.math.multiply(prec[i], weight_matrix[i]), f_mean[i])
    mu = tf.math.multiply(var, summation)
    
    return mu, var

class MGP(tf.keras.Model):
    def __init__(self, C, num_inducing_points, num_experts=2, x_dims=20, a_eps=1e-4, resnet_layers=3, resnet_hidden=128):
        super().__init__()
        self.C = C
        self.num_inducing_points = num_inducing_points
        self.num_experts = num_experts
        self.feat_extractors = [tf.keras.models.Sequential([
            ResNetFeat(resnet_layers, resnet_hidden),
            tf.keras.layers.Flatten()]) for _ in range(num_experts)]

        lengthscales = [tf.convert_to_tensor([1.0] * dim, dtype=default_float()) for dim in x_dims] # lengthscale initialization
        self.kern = [gpflow.kernels.RBF(lengthscales=lengthscale) for lengthscale in lengthscales]

        self.likelihood = GaussianHeteroskedastic(self.C)
        self.softmax_likelihood = gpflow.likelihoods.Softmax(self.C)
        self.a_eps = a_eps        
        
    def initialize(self, X, num_data):        
        ivs = []
        for i in range(self.num_experts):
            feat = self.feat_extractors[i](X[i], training=False)
            Z = feat[:self.num_inducing_points]
            ivs.append(tf.convert_to_tensor(Z))

        self.experts = []
        for i in range(self.num_experts):
            expert = SVGP(
              kernel=self.kern[i], 
              likelihood=self.likelihood, 
              num_latent_gps=self.C, 
              inducing_variable=ivs[i], 
              whiten=False, 
              num_data=num_data)
            self.experts.append(expert)
        
        for expert in self.experts:
            gpflow.set_trainable(expert.inducing_variable, True)

    def transform_dir(self, y):
        y = tf.one_hot(y, self.C)
        s2_tilde = tf.cast(tf.math.log(1.0/(y+self.a_eps) + 1), dtype=tf.float64)
        Y_tilde = tf.cast(tf.cast(tf.math.log(y+self.a_eps), dtype=tf.float64) - 0.5 * s2_tilde, dtype=tf.float64)
        return Y_tilde, s2_tilde

    def call(self, X, y, weighting, training):
        # Extract features
        for view in range(self.num_experts):
            X[view] = self.feat_extractors[view](X[view], training=training)

        # Convert to y_tilde
        y, s2_tilde = self.transform_dir(y)

        # Posteriors and loss
        f_mean, f_var = dict(), dict()
        loss = 0.0
        for view in range(self.num_experts):
            elbo_expert, f_mean[view], f_var[view] = elbo(X[view], y, self.experts[view], s2_tilde=s2_tilde)
            loss += -elbo_expert

        # Combining distributions
        mu, var = combine_dist(f_mean, f_var, weighting)

        # Transform back to categorical distributions 
        p_combined, var_combined = self.softmax_likelihood.predict_mean_and_var(mu, var)
        p_experts, var_experts = zip(*[self.softmax_likelihood.predict_mean_and_var(f_mean[view], f_var[view]) for view in range(self.num_experts)])

        return p_experts, var_experts, p_combined, var_combined, loss