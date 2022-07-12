"""
Hierarchical Question-Image Co-Attention for Visual Question Answering
Ref: https://arxiv.org/abs/1606.00061
Ref: https://github.com/jiasenlu/HieCoAttenVQA

This implementation is based on work done by Harsha Reddy (https://github.com/harsha977) Ref:
https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180
"""

import tensorflow as tf


class PhraseFeatures(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.unigram_conv = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                   kernel_size=1, strides=1, padding='valid',
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=41))
        self.bigram_conv = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                  kernel_size=2, strides=1, padding='same',
                                                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=58),
                                                  dilation_rate=2)
        self.trigram_conv = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                   kernel_size=3, strides=1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=89),
                                                   dilation_rate=2)
        self.max_pool = tf.keras.layers.MaxPool2D((3, 1))
        self.phrase_dropout = tf.keras.layers.Dropout(self.dropout)

        self.tanh = tf.keras.layers.Activation('tanh')
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, words):
        unigrams = tf.expand_dims(self.tanh(self.unigram_conv(words)), 1)
        bigrams = tf.expand_dims(self.tanh(self.bigram_conv(words)), 1)
        trigrams = tf.expand_dims(self.tanh(self.trigram_conv(words)), 1)

        phrase = tf.squeeze(self.max_pool(tf.concat((unigrams, bigrams, trigrams), 1)), axis=1)
        phrase = self.tanh(phrase)
        phrase = self.phrase_dropout(phrase)

        return phrase

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
        })
        return config


class ParallelCoAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.image_corr = tf.keras.layers.Dense(self.hidden_size,
                                                kernel_initializer=tf.keras.initializers.glorot_normal(seed=29))

        self.question_atten_dense = tf.keras.layers.Dense(self.hidden_size,
                                                          kernel_initializer=tf.keras.initializers.glorot_uniform(
                                                              seed=17))
        self.image_atten_dense = tf.keras.layers.Dense(self.hidden_size,
                                                       kernel_initializer=tf.keras.initializers.glorot_uniform(seed=28))
        self.question_atten_dropout = tf.keras.layers.Dropout(self.dropout)
        self.image_atten_dropout = tf.keras.layers.Dropout(self.dropout)

        self.ques_atten = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=21))
        self.img_atten = tf.keras.layers.Dense(1,
                                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=33))

        self.tanh = tf.keras.layers.Activation('tanh')
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        # Parallel CoAttention
        img_feat, ques_feat = inputs
        img_corr = self.image_corr(img_feat)

        weight_matrix = tf.keras.backend.batch_dot(ques_feat, img_corr, axes=(2, 2))
        weight_matrix = self.tanh(weight_matrix)

        ques_embed = self.question_atten_dense(ques_feat)
        img_embed = self.image_atten_dense(img_feat)

        # atten for question feature
        transform_img = tf.keras.backend.batch_dot(weight_matrix, img_embed)
        ques_atten_sum = self.tanh(transform_img + ques_embed)
        ques_atten_sum = self.question_atten_dropout(ques_atten_sum)
        ques_atten = self.ques_atten(ques_atten_sum)
        ques_atten = tf.keras.layers.Reshape((ques_atten.shape[1],))(ques_atten)
        ques_atten = self.softmax(ques_atten)

        # atten for image feature
        transform_ques = tf.keras.backend.batch_dot(weight_matrix, ques_embed, axes=(1, 1))
        img_atten_sum = self.tanh(transform_ques + img_embed)
        img_atten_sum = self.image_atten_dropout(img_atten_sum)
        img_atten = self.img_atten(img_atten_sum)
        img_atten = tf.keras.layers.Reshape((img_atten.shape[1],))(img_atten)
        img_atten = self.softmax(img_atten)

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


class RecursiveCoAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.hw_dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name="h_w_Dense",
                                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=32),
                                              input_shape=(hidden_size,))
        self.hp_dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name="h_p_Dense",
                                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=32),
                                              input_shape=(2 * hidden_size,))
        self.hs_dense = tf.keras.layers.Dense(hidden_size, activation='tanh', name="h_s_Dense",
                                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=32),
                                              input_shape=(2 * hidden_size,))
        self.z_dense = tf.keras.layers.Dense(2 * hidden_size, activation='relu', name="z_Dense",
                                             kernel_initializer=tf.keras.initializers.he_normal(seed=84))

        self.feat_w_dropout = tf.keras.layers.Dropout(self.dropout)
        self.feat_p_dropout = tf.keras.layers.Dropout(self.dropout)
        self.feat_s_dropout = tf.keras.layers.Dropout(self.dropout)
        self.z_dropout = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        # Recursive Attention
        q_word, v_word, q_phrase, v_phrase, q_sent, v_sent = inputs
        feat_w = self.feat_w_dropout(tf.add(q_word, v_word))
        h_w = self.hw_dense(feat_w)

        feat_p = self.feat_p_dropout(tf.concat((tf.add(q_phrase, v_phrase), h_w), axis=1))
        h_p = self.hp_dense(feat_p)

        feat_s = self.feat_s_dropout(tf.concat((tf.add(q_sent, v_sent), h_p), axis=1))
        h_s = self.hs_dense(feat_s)

        z = self.z_dropout(h_s)
        z = self.z_dense(z)

        return z

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
        })
        return config
