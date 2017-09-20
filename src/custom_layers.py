import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Recurrent, ELU, Wrapper
from keras import activations, initializations, regularizers, constraints
from keras.layers.recurrent import time_distributed_dense


class ELUGRU(Recurrent):
    def __init__(self, output_dim, init='glorot_uniform', inner_init='orthogonal', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.inner_init = initializations.get(inner_init)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        self.init = "he_normal"
        self.activation = lambda x: K.elu(x, 1.0)

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(ELUGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':
            self.W = self.add_weight((self.input_dim, 3 * self.output_dim), initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer)
            self.U = self.add_weight((self.output_dim, 3 * self.output_dim), initializer=self.inner_init,
                                     name='{}_U'.format(self.name), regularizer=self.U_regularizer)
            self.b = self.add_weight((self.output_dim * 3,), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)
        else:
            self.W_z = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_z'.format(self.name), regularizer=self.W_regularizer)
            self.U_z = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_z'.format(self.name), regularizer=self.W_regularizer)
            self.b_z = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_z'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_r = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_r'.format(self.name), regularizer=self.W_regularizer)
            self.U_r = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_r'.format(self.name), regularizer=self.W_regularizer)
            self.b_r = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_r'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_h = self.add_weight((self.input_dim, self.output_dim), initializer=self.init,
                                       name='{}_W_h'.format(self.name), regularizer=self.W_regularizer)
            self.U_h = self.add_weight((self.output_dim, self.output_dim), initializer=self.init,
                                       name='{}_U_h'.format(self.name), regularizer=self.W_regularizer)
            self.b_h = self.add_weight((self.output_dim,), initializer='zero', name='{}_b_h'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W, input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W, input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W, input_dim, self.output_dim, timesteps)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(3)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim, 'inner_init': self.inner_init.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W, 'dropout_U': self.dropout_U}
        base_config = super(ELUGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BetterTimeDistributed(Wrapper):
    def __init__(self, layer, n_distribution_axes=1, **kwargs):
        self.supports_masking = True
        self.n_distribution_axes = n_distribution_axes
        super(BetterTimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if not self.layer.built:
            print "build: input_shape", input_shape
            child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
            print "build: child_input_shape", child_input_shape
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(BetterTimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        distribution_axes = tuple(input_shape[1:self.n_distribution_axes + 1])
        output_shape = (child_output_shape[0],) + distribution_axes + child_output_shape[1:]
        return output_shape

    def compute_output_shape(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[self.n_distribution_axes + 1:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        distribution_axes = tuple(input_shape[1:self.n_distribution_axes + 1])
        output_shape = (child_output_shape[0],) + distribution_axes + child_output_shape[1:]
        return output_shape

    def call(self, inputs, mask=None):
        actual_input_shape = K.shape(inputs)

        reshaped_axis_size = actual_input_shape[0]
        for i in range(self.n_distribution_axes):
            reshaped_axis_size *= actual_input_shape[i + 1]
        reshaped_input_shape = (reshaped_axis_size,) + tuple(actual_input_shape[self.n_distribution_axes + 1:])

        inputs = K.reshape(inputs, reshaped_input_shape, ndim=len(reshaped_input_shape))
        y = self.layer.call(inputs)  # (nb_samples * timesteps, ...)
        actual_output_shape = K.shape(y)

        reshaped_output_shape = tuple(actual_input_shape[:self.n_distribution_axes + 1]) + tuple(
            actual_output_shape[1:])

        y = K.reshape(y, reshaped_output_shape, ndim=len(reshaped_output_shape))

        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        return y
