"""
Based on Parallel Co-Attention mechanism introduced in HieCoAtt.
Replaced tanh activation with ReLU. Added Layernorm before each activation.
"""

import tensorflow as tf


class CoAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.image_corr = tf.keras.layers.Dense(self.hidden_size)

        self.question_atten_dense = tf.keras.layers.Dense(self.hidden_size)
        self.image_atten_dense = tf.keras.layers.Dense(self.hidden_size)
        self.question_atten_dropout = tf.keras.layers.Dropout(self.dropout)
        self.image_atten_dropout = tf.keras.layers.Dropout(self.dropout)

        self.ques_atten = tf.keras.layers.Dense(1)
        self.img_atten = tf.keras.layers.Dense(1)

        self.activation = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()
        self.layernorm4 = tf.keras.layers.LayerNormalization()
        self.layernorm5 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        # Parallel CoAttention
        img_feat, ques_feat = inputs
        img_corr = self.image_corr(img_feat)

        weight_matrix = tf.keras.backend.batch_dot(ques_feat, img_corr, axes=(2, 2))
        weight_matrix = self.activation(self.layernorm1(weight_matrix))

        ques_embed = self.question_atten_dense(ques_feat)
        img_embed = self.image_atten_dense(img_feat)

        # atten for question feature
        transform_img = tf.keras.backend.batch_dot(weight_matrix, img_embed)
        ques_atten_sum = self.activation(self.layernorm2(transform_img + ques_embed))
        ques_atten_sum = self.question_atten_dropout(ques_atten_sum)
        ques_atten = self.ques_atten(ques_atten_sum)
        ques_atten = tf.keras.layers.Reshape((ques_atten.shape[1],))(ques_atten)
        ques_atten = self.softmax(self.layernorm3(ques_atten))

        # atten for image feature
        transform_ques = tf.keras.backend.batch_dot(weight_matrix, ques_embed, axes=(1, 1))
        img_atten_sum = self.activation(self.layernorm4(transform_ques + img_embed))
        img_atten_sum = self.image_atten_dropout(img_atten_sum)
        img_atten = self.img_atten(img_atten_sum)
        img_atten = tf.keras.layers.Reshape((img_atten.shape[1],))(img_atten)
        img_atten = self.softmax(self.layernorm5(img_atten))

        ques_atten = tf.keras.layers.Reshape((1, ques_atten.shape[1]))(ques_atten)
        img_atten = tf.keras.layers.Reshape((1, img_atten.shape[1]))(img_atten)

        ques_atten_feat = tf.keras.backend.batch_dot(ques_atten, ques_feat)
        ques_atten_feat = tf.keras.layers.Reshape((ques_atten_feat.shape[-1],))(ques_atten_feat)

        img_atten_feat = tf.keras.backend.batch_dot(img_atten, img_feat)
        img_atten_feat = tf.keras.layers.Reshape((img_atten_feat.shape[-1],))(img_atten_feat)

        return [img_atten_feat, ques_atten_feat]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
        })
        return config
