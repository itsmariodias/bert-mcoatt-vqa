"""
Deep Modular Co-Attention Networks for Visual Question Answering
Ref: https://arxiv.org/abs/1906.10770
Ref: https://github.com/MILVLG/mcan-vqa

This implementation is based on a Tensorflow-based version of the code given in the repository linked above
We also refer to the code samples given at https://keras.io/examples/nlp/text_classification_with_transformer/
"""

import tensorflow as tf

from models.bert_mcan.mcan_utils import MLP


# self attention module using in decoder networks in transformers
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate, use_activation, layer_norm_eps, init_range, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.use_activation = use_activation
        self.layer_norm_eps = layer_norm_eps
        self.init_range = init_range

        self.att1 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.init_range),
        )
        self.att2 = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.init_range),
        )

        self.ffn = MLP(
            self.ff_dim,
            self.embed_dim,
            self.rate,
            self.init_range,
            self.use_activation
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.dropout3 = tf.keras.layers.Dropout(self.rate)

    def call(self, inputs, mask, training):
        x, y = inputs
        x_mask, y_mask = mask
        attn_output1 = self.att1(x, x, attention_mask=x_mask[:, tf.newaxis, tf.newaxis, :])
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(x + attn_output1)

        attn_output2 = self.att2(out1, y, attention_mask=y_mask[:, tf.newaxis, tf.newaxis, :])
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)

        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'use_activation': self.use_activation,
            'layer_norm_eps': self.layer_norm_eps,
            'init_range': self.init_range
        })
        return config


# decoder module used in encoder-decoder transformer networks
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate, use_activation, layer_norm_eps, init_range,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.use_activation = use_activation
        self.layer_norm_eps = layer_norm_eps
        self.init_range = init_range

        self.decoder = [
            DecoderBlock(
                self.embed_dim,
                self.num_heads,
                self.ff_dim,
                self.rate,
                self.use_activation,
                self.layer_norm_eps,
                self.init_range,
            ) for _ in range(self.num_layers)
        ]

    def call(self, inputs, mask, training):
        x, y = inputs
        for layer in self.decoder:
            x = layer([x, y], mask, training=training)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'use_activation': self.use_activation,
            'layer_norm_eps': self.layer_norm_eps,
            'init_range': self.init_range,
        })
        return config
