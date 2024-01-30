from .MLP import MLP
import tensorflow as tf
import abc
import math
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from itertools import chain, combinations
import numpy as np
import warnings
from typing import Callable, Optional

np.seterr(all='raise')


def sample_discrete(unnormalized_probs):
    unnormalized_probs = np.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = np.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        warnings.warn("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]

class InducingPointInitializer:
    def __init__(self, seed: Optional[int] = 0, randomized: Optional[bool] = True, **kwargs):
        self._randomized = randomized
        self.seed = seed if self.randomized else None

    def __call__(self, training_inputs: np.ndarray, M: int,
                 kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        if self.seed is not None:
            restore_random_state = np.random.get_state()
            np.random.seed(self.seed)
        else:
            restore_random_state = None

        Z = self.compute_initialisation(training_inputs, M, kernel)

        if self.seed is not None:
            np.random.set_state(restore_random_state)

        return Z

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        raise NotImplementedError

    @property
    def randomized(self):
        return self._randomized

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if k not in ['_randomized']])
        return f"{type(self).__name__}({params})"


class FirstSubsample(InducingPointInitializer):
    def __init__(self, **kwargs):
        super().__init__(randomized=False, **kwargs)

    def compute_initialisation(self, training_inputs: np.ndarray, M: int, *args, **kwargs):
        return training_inputs[:M], np.arange(M)


class UniformSubsample(InducingPointInitializer):
    def __init__(self, seed: Optional[int] = 0, **kwargs):
        super().__init__(seed=seed, randomized=True, **kwargs)

    def compute_initialisation(self, training_inputs: np.ndarray, M: int, *args, **kwargs):
        """
        Returns a subset of inducing inputs Z ⊂ 𝑋, with |Z| = M such that the probability of any such set is equal.
        :param training_inputs: An array of training inputs X ⊂ 𝑋, with |X| = N < ∞. We frequently assume X= ℝ^D
        and this is [N,D]
        :param M: integer, number of inducing points to return
        :param kernel: unused argument
        :return: Z, indices, [M,D], M inducing inputs, indices of inducing points in X
        """
        N = training_inputs.shape[0]
        indices = np.random.choice(N, size=M, replace=False)  # we always sample without replacement
        return tf.gather(training_inputs, indices), indices
  
def powerset(num_views):
    s = list(range(num_views))
    sub_sets = chain.from_iterable(combinations(s,r) for r in range(1,len(s)+1))
    zeros = []
    for sub_set in sub_sets:
        temp = np.zeros(num_views)
        temp[list(sub_set)] = 1
        zeros.append(temp)
    zeros = np.stack(zeros,0)
    return zeros

def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if len(loc.shape) < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return tfp.distributions.Independent(tfp.distributions.Normal(loc, scale_diag), 1)

def get_attender(attention, kq_size, value_size, out_size, **kwargs):

    if not isinstance(attention, str):
        return attention(kq_size, value_size, out_size, **kwargs)

    attention = attention.lower()
    if attention == "scaledot":
        attender = DotAttender(kq_size, value_size, out_size, is_scale=True, **kwargs)
    elif attention == "dot":
        attender = DotAttender(kq_size, value_size, out_size, is_scale=False, **kwargs)
    elif attention == "adaptive_rbf":
        attender = RBFAttender(kq_size, value_size, out_size, **kwargs)
    else:
        raise ValueError("Unknown attention method {}".format(attention))

    return attender

class BaseAttender(abc.ABC, layers.Layer):

    def __init__(self, kq_size, value_size, out_size, is_normalize=True, dropout=0, norm_type='sparsemax'):
        super().__init__()
        self.kq_size = kq_size
        self.value_size = value_size
        self.out_size = out_size
        self.is_normalize = is_normalize
        self.dropout = layers.Dropout(dropout) if dropout > 0 else tf.identity
        self.is_resize = self.value_size != self.out_size
        self.norm_type = norm_type

        if self.is_resize:
            self.resizer = layers.Dense(
                self.out_size,
                activation=None,
                input_shape=(self.value_size,))


    def call(self, keys, queries, values, **kwargs):
        logits = self.score(keys, queries, **kwargs)
        attn = self.logits_to_attn(logits)
        attn = self.dropout(attn)
        values = tf.matmul(attn, values)

        if self.is_resize:
            values = self.resizer(values)

        return values, attn

    def logits_to_attn(self, logits):
        if self.is_normalize:
            if self.norm_type == 'sparsemax':
                attn = tfa.activations.sparsemax(logits,axis=-1)
            elif self.norm_type == 'softmax':
                attn = tf.nn.softmax(logits, axis=-1)
            else:
                print('Unknown attention norm type')
        else:
            attn = logits
        return attn

    @abc.abstractmethod
    def score(keys, queries, **kwargs):
        """Score function which returns the logits between keys and queries."""
        pass


class DotAttender(BaseAttender):
    def __init__(self, *args, is_scale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_scale = is_scale

    def score(self, keys, queries, **kwargs):
        if len(keys.shape) == 3:
            keys_shape = "bkd" 
            queries_shape = "bqd"
            logits = tf.einsum(
                "{},{}->bqk".format(keys_shape, queries_shape), keys, queries
            )
        else:
            keys_shape = "kd" 
            queries_shape = "qd"

            logits = tf.einsum(
                "{},{}->qk".format(keys_shape, queries_shape), keys, queries
            )

        if self.is_scale:
            kq_size = queries.shape[-1]
            logits = logits / math.sqrt(kq_size)

        return logits

class RBFAttender(BaseAttender):
    def __init__(self, *args, is_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_weight = is_weight
        if self.is_weight:
            self.weighter = layers.Dense(
                self.kq_size,
                activation=None,
                input_shape=(self.kq_size,))

        self.lengthscale = tf.Variable(tf.ones(self.kq_size, dtype=tf.float32)*10., trainable=True, name='rbf_lengthscale')
        self.variance = tf.Variable(1., trainable=True)

    def score(self, keys, queries, **kwargs):
        if len(queries.shape) == 2:
            keys = tf.expand_dims(keys/self.lengthscale,0)
            queries = tf.expand_dims(queries/self.lengthscale,1)
        else:
            keys = tf.expand_dims(keys/self.lengthscale,1)
            queries = tf.expand_dims(queries/self.lengthscale,2)

        r2 = tf.reduce_sum((keys-queries)**2,axis=-1)
        logits = tf.exp(-0.5*r2) * self.variance
        return logits


class MergeFlatInputs(tf.keras.layers.Layer):
    def __init__(self, FlatModule, x1_dim, x2_dim, n_out, is_sum_merge=False, is_dot_merge=False, **kwargs):
        super().__init__()
        self.is_sum_merge = is_sum_merge
        self.is_dot_merge = is_dot_merge

        if self.is_sum_merge or self.is_dot_merge:
            dim = x1_dim
            if x1_dim == x2_dim:
                self.resizer = tf.identity
            else:
                self.resizer = MLP(x2_dim, dim, l2=kwargs['l2'])  # transform to be the correct size            
        else:
            dim = x1_dim + x2_dim

        self.flat_module = FlatModule(dim, n_out, **kwargs)

    def call(self, x1, x2):
        if self.is_sum_merge:
            x2 = self.resizer(x2)
            out = tf.nn.relu(x1 + x2)
        elif self.is_dot_merge:
            x2 = self.resizer(x2)
            out = tf.nn.relu(x1 * x2)
        else:
            out = tf.concat((x1, x2), axis=-1)

        return self.flat_module(out)
    
    def skip_merge(self, x):
        return self.flat_module(x)

def merge_flat_input(module, is_sum_merge=False, **kwargs):
    def merged_flat_input(x_shape, flat_dim, n_out, **kwargs2):
        assert isinstance(x_shape, int)
        return MergeFlatInputs(
            module,
            x_shape,
            flat_dim,
            n_out,
            is_sum_merge=is_sum_merge,
            **kwargs2,
            **kwargs
        )

    return merged_flat_input
