import tensorflow as tf
from tensorflow.python.keras import backend


class LSTMPeepholeCell(tf.keras.layers.LSTMCell):
    """Cell class for the LSTM peephole layer.

    For more info about peephole connections, read [original paper Gres F. et al., 2002](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf) and [Greff K. et al., 2017](https://arxiv.org/pdf/1503.04069.pdf) for visualization and comparison with other LSTM variants.

    Cell layer can be used as argument for `tf.keras.layers.RNN` layer. It is similar to `tf.keras.layers.LSTMCell` with additional peephole connections.

    See [the LSTMCell Api documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell) for more information about arguments and methods.

    The whole sequence can then be processed using the tf.keras.layers.RNN class with this cell class as argument as shown in the example below:
    >>> inputs = tf.random.normal([32, 10, 8])
    >>> cells = LSTMPeepholeCell(20)
    >>> peephole_lstm = tf.keras.layers.RNN(cells)
    >>> my_output = peephole_lstm(inputs)
    >>> print(my_output.shape)
    (32, 20)
    >>> peephole_lstm = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = peephole_lstm(inputs)
    >>> print(whole_seq_output.shape)
    (32, 10, 20)
    >>> print(final_memory_state.shape)
    (32, 20)
    >>> print(final_carry_state.shape)
    (32, 20)
    """

    def __init__(
        self,
        units,
        **kwargs,
    ):
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # Add peephole weights
        self.peephole_kernel = self.add_weight(
            shape=(self.units * 3,),
            name="peephole_kernel",
            initializer="glorot_uniform",  # self.kernel_initializer # TODO provide kernel initializer with seed
            trainable=True,
        )

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Modification of _compute_carry_and_output method of LSTMCell to include peephole connections.
        Original call method accessed on 9/9/2023: https://github.com/keras-team/keras/blob/c5eb85b66bf44640114986c841b0839fdcc2dea3/keras/layers/rnn/lstm.py#L240
        """
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i
            + backend.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
            + self.peephole_kernel[: self.units] * c_tm1
        )
        f = self.recurrent_activation(
            x_f
            + backend.dot(
                h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            + self.peephole_kernel[self.units : self.units * 2] * c_tm1
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + backend.dot(
                h_tm1_c,
                self.recurrent_kernel[:, self.units * 2 : self.units * 3],
            )
        )
        o = self.recurrent_activation(
            x_o
            + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
            + self.peephole_kernel[self.units * 2 :] * c
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Modification of _compute_carry_and_output_fused method of LSTMCell to include peephole connections.
        Original call method accessed on 9/9/2023: https://github.com/keras-team/keras/blob/c5eb85b66bf44640114986c841b0839fdcc2dea3/keras/layers/rnn/lstm.py#L266
        """
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0 + self.peephole_kernel[: self.units] * c_tm1)
        f = self.recurrent_activation(
            z1 + self.peephole_kernel[self.units : self.units * 2] * c_tm1
        )
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3 + self.peephole_kernel[self.units * 2 :] * c)
        return c, o


class ResidualLSTMCell(tf.keras.layers.LSTMCell):
    """Cell class for the residual LSTM layer.

    Cell layer can be used as argument for `tf.keras.layers.RNN` layer.
    It is similar to `tf.keras.layers.LSTMCell` with additional residual connection as described by [Kim J., 2017](https://arxiv.org/pdf/1701.03360.pdf).

    See [the LSTMCell Api documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell) for more information about arguments and methods.

    The whole sequence can then be processed using the `tf.keras.layers.RNN` class with this cell class as argument as shown in the example below:
    >>> inputs = tf.random.normal([32, 10, 8])
    >>> cells = LSTMResidualCell(20)
    >>> residual_lstm = tf.keras.layers.RNN(cells)
    >>> my_output = residual_lstm(inputs)
    >>> print(my_output.shape)
    (32, 20)
    >>> residual_lstm = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = residual_lstm(inputs)
    >>> print(whole_seq_output.shape)
    (32, 10, 20)
    >>> print(final_memory_state.shape)
    (32, 20)
    >>> print(final_carry_state.shape)
    (32, 20)
    """

    def __init__(
        self,
        units,
        **kwargs,
    ):
        super().__init__(units, **kwargs)

    def call(self, inputs, states, training=None):
        """Modification of _compute_carry_and_output method of LSTMCell to include peephole connections.
        Original call method accessed on 9/9/2023:
        https://github.com/keras-team/keras/blob/c5eb85b66bf44640114986c841b0839fdcc2dea3/keras/layers/rnn/lstm.py#L275
        """

        # Original unchanged code:
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        if self.implementation == 1:
            if 0 < self.dropout < 1.0:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
            x_i = backend.dot(inputs_i, k_i)
            x_f = backend.dot(inputs_f, k_f)
            x_c = backend.dot(inputs_c, k_c)
            x_o = backend.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
                x_i = backend.bias_add(x_i, b_i)
                x_f = backend.bias_add(x_f, b_f)
                x_c = backend.bias_add(x_c, b_c)
                x_o = backend.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            if 0.0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            z = backend.dot(inputs, self.kernel)
            z += backend.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = backend.bias_add(z, self.bias)

            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        # Our change: add residual connection from input x to output of cell state c if shapes of input and output of cell state match
        if inputs.shape[-1] == c.shape[-1]:
            h = o * (self.activation(c) + inputs)
        else:
            h = o * self.activation(c)
        return h, [h, c]
