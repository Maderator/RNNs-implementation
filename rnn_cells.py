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
