import numpy as np
import tensorflow as tf
import gpflow
from .utils import UniformSubsample
from gpflow.config import default_float
from models.SVGP import SVGP
from gpflow import logdensities 
from gpflow.likelihoods import Likelihood
from gpflow.quadrature import ndiag_mc

class MonteCarloLikelihood(Likelihood):
    def __init__(self, num_monte_carlo_points, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_monte_carlo_points = num_monte_carlo_points

    def _mc_quadrature(self, funcs, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
        return ndiag_mc(funcs, self.num_monte_carlo_points, Fmu, Fvar, logspace, epsilon, **Ys)

    def _predict_mean_and_var(self, Fmu, Fvar, epsilon=None):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           ∫∫ y p(y|f)q(f) df dy

        and the predictive variance

           ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²

        Here, we implement a default Monte Carlo routine.
        """

        def conditional_y_squared(*F):
            return self.conditional_variance(*F) + tf.square(self.conditional_mean(*F))

        E_y, E_y2 = self._mc_quadrature(
            [self.conditional_mean, conditional_y_squared], Fmu, Fvar, epsilon=epsilon
        )
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y  # [N, D]

    def _predict_log_density(self, Fmu, Fvar, Y, epsilon=None):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            log ∫ p(y=Y|f)q(f) df

        Here, we implement a default Monte Carlo routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, logspace=True, epsilon=epsilon),
            axis=-1,
        )

    def _variational_expectations(self, Fmu, Fvar, Y, epsilon=None):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)  - Fmu: [N, D]  Fvar: [N, D]

        and this object represents

            p(y|f)  - Y: [N, 1]

        then this method computes

           ∫ (log p(y|f)) q(f) df.


        Here, we implement a default Monte Carlo quadrature routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, epsilon=epsilon), axis=-1
        )

class Softmax(MonteCarloLikelihood):
    """
    The soft-max multi-class likelihood.  It can only provide a stochastic
    Monte-Carlo estimate of the variational expectations term, but this
    added variance tends to be small compared to that due to mini-batching
    (when using the SVGP model).
    """

    def __init__(self, num_classes, num_monte_carlo_points, **kwargs):
        super().__init__(latent_dim=num_classes, observation_dim=None, num_monte_carlo_points=num_monte_carlo_points, **kwargs)
        self.num_classes = self.latent_dim

    def _log_prob(self, F, Y):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:, 0])

    def _conditional_mean(self, F):
        return tf.nn.softmax(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - p ** 2

class GaussianHeteroskedastic(Likelihood):
    def __init__(self, num_class):
        super().__init__(latent_dim=num_class, observation_dim=None)

    def _log_prob(self, F, Y):
        return logdensities.gaussian(F, Y, self.variance)

    def _conditional_mean(self, F):
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar, vars):
        return tf.identity(Fmu), Fvar + vars

    def _predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Fmu, Y, Fvar + self.variance)

    def _predict_log_density(self, Fmu, Fvar, Y):
        return None
    
    def _variational_expectations(self, Fmu, Fvar, Y, vars):
        return -0.5 * tf.cast(np.log(2 * np.pi), dtype=tf.float64) - 0.5 * tf.math.log(vars) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / vars
    def _negative_log_likelihood(self, Fmu, Fvar, Y):
        return -0.5 * tf.cast(np.log(2 * np.pi), dtype=tf.float64) - 0.5 * tf.math.log(Fvar) \
               - 0.5 * tf.square(Y - Fmu) / Fvar
    
def elbo(X, Y, model, s2_tilde=None, beta=1.):
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
    return tf.reduce_sum(var_exp) * scale - beta*kl, f_mean, f_var

def combine_dist(f_mean, f_var, weighting='uniform'):
    num_dists = len(f_mean)
    if 'entropy' in weighting:
        weight_matrix = [-(0.5*tf.math.log(2*np.pi*f_var[i])+0.5) for i in range(num_dists)] 
    elif 'poe' in weighting:
        weight_matrix = tf.ones(num_dists)
                
    # Normalize weight matrix
    if weighting != 'poe':
        if 'softmax' in weighting:
            sum_weight = tf.math.exp(weight_matrix[0])
            for i in range(1, num_dists):
                sum_weight += tf.math.exp(weight_matrix[i])
            for i in range(num_dists):
                weight_matrix[i] = tf.math.exp(weight_matrix[i])/sum_weight
        else:
            sum_weight = weight_matrix[0]
            for i in range(1, num_dists):
                sum_weight += weight_matrix[i]
            for i in range(num_dists):
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
    def __init__(self, 
                 C, 
                 num_inducing_points, 
                 num_experts=2, 
                 a_eps=1e-4, 
                 beta=1.0,
                 num_monte_carlo_points=100):
        super().__init__()
        self.C = C
        self.num_inducing_points = num_inducing_points
        self.num_experts = num_experts
        self.likelihood = GaussianHeteroskedastic(self.C)
        self.softmax_likelihood = Softmax(self.C, num_monte_carlo_points)
        self.a_eps = a_eps
        self.beta = beta
        
    def initialize(self, X, num_data, lengthscales_init=None, init_lengthscale_data=False): 
        if init_lengthscale_data:
            lengthscales = [tf.cast(x, dtype=default_float()) for x in lengthscales_init]
        else:
            lengthscales = [tf.convert_to_tensor([1.0] * x.shape[1], dtype=default_float()) for x in X]
        self.kern = [gpflow.kernels.RBF(lengthscales=lengthscale) for lengthscale in lengthscales]

        ivs = []
        for i in range(self.num_experts):
            init_method = UniformSubsample()
            Z = init_method.compute_initialisation(X[i], self.num_inducing_points, self.kern[i])[0]
            ivs.append(tf.convert_to_tensor(Z))

        self.experts = []
        for i in range(self.num_experts):
            expert = SVGP(kernel = self.kern[i], likelihood = self.likelihood, num_latent_gps = self.C, inducing_variable = ivs[i], whiten=False, num_data=num_data)
            self.experts.append(expert)
        
        for expert in self.experts:
            gpflow.set_trainable(expert.inducing_variable, True)
        self.num_data = num_data

    def transform_dir(self, y):
        y = tf.cast(y, dtype=tf.int32)
        y = tf.one_hot(y, self.C)
        s2_tilde = tf.cast(tf.math.log(1.0/(y+self.a_eps) + 1), dtype=tf.float64)
        Y_tilde = tf.cast(tf.cast(tf.math.log(y+self.a_eps), dtype=tf.float64) - 0.5 * s2_tilde, dtype=tf.float64)
        return Y_tilde, s2_tilde

    def call(self, X, y, weighting):
        # Convert to y_tilde
        if isinstance(self.likelihood, GaussianHeteroskedastic):
            y, s2_tilde = self.transform_dir(y)
        else:
            s2_tilde = None

        # Posteriors and loss
        f_mean, f_var = dict(), dict()
        loss = 0.0
        for view in range(self.num_experts):
            elbo_expert, f_mean[view], f_var[view] = elbo(X[view], y, self.experts[view], s2_tilde=s2_tilde, beta=self.beta)
            loss += -elbo_expert

        # Combining distributions
        mu, var = combine_dist(f_mean, f_var, weighting)

        # Transform to probability distributions
        p_combined, var_combined = self.softmax_likelihood.predict_mean_and_var(mu, var)
        p_experts, var_experts = zip(*[self.softmax_likelihood.predict_mean_and_var(f_mean[view], f_var[view]) for view in range(self.num_experts)])

        return p_experts, var_experts, p_combined, var_combined, loss