"""
Hierarchical Question-Image Co-Attention for Visual Question Answering
Ref: https://arxiv.org/abs/1606.00061
Ref: https://github.com/jiasenlu/HieCoAttenVQA

Based on Alternating Co-Attention mechanism introduced in HieCoAtt.
Replaced tanh activation with ReLU.
"""

import tensorflow as tf


class AlternatingCoAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.ques_embed_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.h1_dense = tf.keras.layers.Dense(1)
        self.ques_embed_img_dense = tf.keras.layers.Dense(self.hidden_size)
        self.img_embed_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.h2_dense = tf.keras.layers.Dense(1)
        self.img_embed_dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.ques_embed_dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.h3_dense = tf.keras.layers.Dense(1)

        self.ques_embed_dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.img_embed_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ques_embed_dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        img_feat, img_mask, ques_feat, ques_mask = inputs

        ques_embed = self.ques_embed_dense1(ques_feat)

        feat = self.ques_embed_dropout1(tf.keras.layers.ReLU()(ques_embed))
        h1 = self.h1_dense(feat)
        ques_atten = tf.keras.layers.Reshape((h1.shape[1],))(h1)
        ques_atten = tf.keras.layers.Softmax()(ques_atten, ques_mask)
        ques_atten_feat_1 = tf.keras.backend.batch_dot(ques_atten, ques_feat)

        # Image Attention
        ques_embed_img = self.ques_embed_img_dense(ques_atten_feat_1)

        img_embed = self.img_embed_dense1(img_feat)

        ques_replicate = tf.keras.layers.RepeatVector(img_embed.shape[1])(ques_embed_img)

        feat = self.img_embed_dropout(tf.keras.layers.ReLU()(img_embed + ques_replicate))
        h2 = self.h2_dense(feat)
        img_atten = tf.keras.layers.Reshape((h2.shape[1],))(h2)
        img_atten = tf.keras.layers.Softmax()(img_atten, img_mask)
        img_atten_feat = tf.keras.backend.batch_dot(img_atten, img_feat)
        img_atten_feat = tf.keras.layers.Reshape((img_atten_feat.shape[-1],))(img_atten_feat)

        # Question Attention
        img_embed = self.img_embed_dense2(img_atten_feat)
        img_replicate = tf.keras.layers.RepeatVector(ques_feat.shape[1])(img_embed)

        ques_embed = self.ques_embed_dense2(ques_feat)

        feat = self.ques_embed_dropout2(tf.keras.layers.ReLU()(ques_embed + img_replicate))
        h3 = self.h3_dense(feat)
        probs3dim = tf.keras.layers.Reshape((h3.shape[1],))(h3)
        probs3dim = tf.keras.layers.Softmax()(probs3dim, ques_mask)
        ques_atten_feat = tf.keras.backend.batch_dot(probs3dim, ques_feat)
        ques_atten_feat = tf.keras.layers.Reshape((ques_atten_feat.shape[-1],))(ques_atten_feat)

        return [img_atten_feat, ques_atten_feat]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
        })
        return config


class CoAttendFeatures(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                                                       dropout=self.dropout), merge_mode='sum')
        self.coattention = AlternatingCoAttention(self.hidden_size, self.dropout)
        self.ques_dense = tf.keras.layers.Dense(self.hidden_size)
        self.img_dense = tf.keras.layers.Dense(self.hidden_size)

    def call(self, inputs):
        img_feat, img_mask, ques_feat, ques_mask = inputs

        ques_lstm_feat = self.lstm(ques_feat, mask=tf.cast(ques_mask, tf.bool))
        img_att, ques_att = self.coattention([img_feat, img_mask, ques_lstm_feat, ques_mask])

        img_feat = self.img_dense(img_att)
        img_feat = tf.keras.layers.ReLU()(img_feat)

        ques_feat = self.ques_dense(ques_att)
        ques_feat = tf.keras.layers.ReLU()(ques_feat)

        joint_feat = tf.keras.layers.Multiply()([img_feat, ques_feat])

        return joint_feat

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
        })
        return config
