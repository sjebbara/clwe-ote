from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Recurrent, Wrapper
from keras.layers.recurrent import _time_distributed_dense


def _elu(x):
    return K.elu(x, 1.0)


class ELUGRU(Recurrent):
    def __init__(self, units, recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='he_normal',
                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None, dropout=0., recurrent_dropout=0., **kwargs):
        super(ELUGRU, self).__init__(**kwargs)
        self.units = units
        self.activation = _elu
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = InputSpec(shape=(None, self.units))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3), name='kernel',
                                      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3), name='recurrent_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,), name='bias', initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units:
        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = _time_distributed_dense(inputs, self.kernel_z, self.bias_z, self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = _time_distributed_dense(inputs, self.kernel_r, self.bias_r, self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_h = _time_distributed_dense(inputs, self.kernel_h, self.bias_h, self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        dp_mask = states[1]  # dropout matrices for recurrent units
        rec_dp_mask = states[2]

        if self.implementation == 2:
            matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            matrix_inner = K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.units:]
            recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0], self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + recurrent_h)
        else:
            if self.implementation == 0:
                x_z = inputs[:, :self.units]
                x_r = inputs[:, self.units: 2 * self.units]
                x_h = inputs[:, 2 * self.units:]
            elif self.implementation == 1:
                x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
                x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
                x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
                if self.use_bias:
                    x_z = K.bias_add(x_z, self.bias_z)
                    x_r = K.bias_add(x_r, self.bias_r)
                    x_h = K.bias_add(x_h, self.bias_h)
            else:
                raise ValueError('Unknown `implementation` mode.')
            z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_z))
            r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2], self.recurrent_kernel_h))
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self.units, 'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias, 'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint), 'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
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
