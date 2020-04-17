import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import random_ops
from tensorflow.python.keras import initializers


class BNSTAR_cell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, t_max=784, training=True,
                 **kwargs):
        '''
        t_max should be a float value corresponding to the longest possible
        time dependency in the input.
        '''
        self.num_units = num_units
        self.t_max = 784
        self.training  = training
        super(BNSTAR_cell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, scope=None):
        """BN-STAR."""
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                h, _ = state
            else:
                h, _ = tf.split(value=state, num_or_size_splits=2, axis=1)


            x_size = x.get_shape().as_list()[1]
            
            W_zx = tf.get_variable('W_xh_z',
                                   [x_size, 1 * self.num_units], initializer=initializers.get('orthogonal'))            
            W_Kx = tf.get_variable('W_xh_K',
                                   [x_size, 1 * self.num_units], initializer=initializers.get('orthogonal'))
            W_Kh = tf.get_variable('W_hh',
                                   [self.num_units, 1 * self.num_units], initializer=initializers.get('orthogonal'))
            

            if self.t_max is None:
                print('Zero initializer ')
                bias = tf.get_variable('bias', [2 * self.num_units],
                                       initializer=bias_initializer(2))
            else:
                print('Using chrono initializer ...')
                bias = tf.get_variable('bias', [2 * self.num_units],
                                       initializer=chrono_init(self.t_max,
                                                               2))
            
            bias_f = bias[self.num_units:,...]
            bias_j = bias[:self.num_units,...]            

            fx = tf.matmul(x, W_Kx)
            fh = tf.matmul(h, W_Kh)
            
            j = tf.matmul(x, W_zx)
            
            
            bn_f = batch_norm(fx, 'fx', self.training) + batch_norm(fh, 'fh', self.training)
            bn_j = batch_norm(j, 'j', self.training)

            
            bn_f = tf.nn.bias_add(bn_f, bias_f)
            bn_j = tf.nn.bias_add(bn_j, bias_j)

            beta = 1
            new_h = tf.sigmoid(bn_f)*h + (1-tf.sigmoid(bn_f-beta))*tf.tanh(bn_j)
            
#            bn_f = tf.sigmoid(bn_f)
#            bn_j = tf.tanh(bn_j)
#            new_h = bn_f * h + (1-bn_f) * bn_j
            new_h = tf.tanh(new_h)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_h, new_h)
            else:
                new_state = tf.concat([new_h, new_h], 1)
            return new_h, new_state


def chrono_init(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))

        bias_j = tf.zeros(num_units)
        bias_f = uni_vals

        return tf.concat([bias_j, bias_f], 0)

    return _initializer


def bias_initializer(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        p = np.zeros(shape)
        num_units = int(shape[0]//num_gates)
        # i, j, o, f
        # f:
        p[-num_units:] = np.ones(num_units)

        return tf.constant(p, dtype)

    return _initializer


def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)
