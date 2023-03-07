"""
Deep Modular Co-Attention Networks for Visual Question Answering
Ref: https://arxiv.org/abs/1906.10770
Ref: https://github.com/MILVLG/mcan-vqa

This implementation is based on a Tensorflow-based version of the code given in the repository linked above
"""

import tensorflow as tf


# multi layer perceptron
class MLP(tf.keras.layers.Layer):
    def __init__(self, mid_size, out_size, rate, init_range=0.02, use_activation=None, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.mid_size = mid_size
        self.out_size = out_size
        self.rate = rate
        self.use_activation = use_activation

        self.fc = tf.keras.layers.Dense(
            self.mid_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=init_range),
        )
        self.linear = tf.keras.layers.Dense(
            self.out_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=init_range),
        )

        if self.rate > 0:
            self.dropout = tf.keras.layers.Dropout(self.rate)

        if self.use_activation == "relu":
            self.activation = tf.keras.activations.relu
        elif self.use_activation == "gelu" or self.use_activation == "gelu_new":
            self.activation = tf.keras.activations.gelu
        elif self.use_activation == "silu":
            self.activation = tf.keras.activations.swish

    def call(self, inputs, training):
        x = self.fc(inputs)

        if self.use_activation == "gelu_new":
            x = self.activation(x, approximate=True)
        elif self.use_activation:
            x = self.activation(x)

        if self.rate > 0:
            x = self.dropout(x, training=training)

        x = self.linear(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mid_size': self.mid_size,
            'out_size': self.out_size,
            'rate': self.rate,
            'init_range': self.init_range,
            'use_activation': self.use_activation,
        })
        return config


# self top down attention module used to flatten features
class AttentionReduction(tf.keras.layers.Layer):
    def __init__(self, hidden_size, out_size, rate, use_activation, init_range, **kwargs):
        super(AttentionReduction, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.rate = rate
        self.use_activation = use_activation
        self.init_range = init_range

        self.mlp = MLP(
            self.hidden_size,
            1,
            self.rate,
            self.init_range,
            self.use_activation
        )

        self.linear_merge = tf.keras.layers.Dense(
            self.out_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.init_range),
        )

    def call(self, inputs, mask, training):
        att = self.mlp(inputs, training=training)
        att = tf.keras.layers.Reshape((att.shape[1],))(att)
        att = tf.keras.layers.Softmax(axis=1)(att, mask)
        att = tf.keras.layers.Reshape((att.shape[-1], 1))(att)

        x_atted = att * inputs
        x_atted = tf.math.reduce_sum(x_atted, axis=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'out_size': self.out_size,
            'rate': self.rate,
            'use_activation': self.use_activation,
            'init_range': self.init_range,
        })
        return config
