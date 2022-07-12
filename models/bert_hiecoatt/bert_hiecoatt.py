"""
Hierarchical Question-Image Co-Attention for Visual Question Answering
Ref: https://arxiv.org/abs/1606.00061
Ref: https://github.com/jiasenlu/HieCoAttenVQA

This implementation is based on work done by Harsha Reddy (https://github.com/harsha977)
Ref: https://medium.com/@harshareddykancharla/visual-question-answering-with-hierarchical-question-image-co-attention-c5836684a180

Some inspiration was also taken from work done by Tulrose Deori (https://github.com/arya46)
Ref: https://github.com/arya46/VQA-Flask-App
"""

from models.bert_hiecoatt import coattention

import tensorflow as tf
import transformers

CUSTOM_OBJECTS = {
    'PhraseFeatures': coattention.PhraseFeatures,
    'ParallelCoAttention': coattention.ParallelCoAttention,
    'RecursiveCoAttention': coattention.RecursiveCoAttention,
    'TFBertMainLayer': transformers.TFBertMainLayer,
}


def BERT_HieCoAtt(C):
    # inputs 
    image_input = tf.keras.layers.Input(
        shape=(C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE,),
        name='Image_Input'
    )
    ques_input = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype='int32',
        name='Question_Input'
    )
    ques_mask = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype='int32',
        name='Question_Mask'
    )

    bert_model = transformers.TFBertModel.from_pretrained(
        C.BERT_MODEL_PATH['target'], from_pt=True
    )
    bert_model.trainable = False

    # Word level
    image_feat = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='tanh', name="Image_Feat_Dense",
                                       kernel_initializer=tf.keras.initializers.glorot_normal(seed=15))(image_input)
    image_feat = tf.keras.layers.Dropout(C.DROPOUT_RATE)(image_feat)

    bert_output = bert_model.bert([ques_input, ques_mask])

    word_feat = bert_output.last_hidden_state

    v_word, q_word = coattention.ParallelCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE,
                                                     name='CoAttention_Word')([image_feat, word_feat])

    # Phrase level
    conv_feat = coattention.PhraseFeatures(C.HIDDEN_SIZE, C.DROPOUT_RATE,
                                           name="PhraseLevelFeatures")(word_feat)

    v_phrase, q_phrase = coattention.ParallelCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE,
                                                         name='CoAttention_Phrase')([image_feat, conv_feat])

    # Question/Sentence level
    core_output = tf.keras.layers.LSTM(C.HIDDEN_SIZE, return_sequences=True, dropout=C.DROPOUT_RATE,
                                       kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                                       recurrent_initializer=tf.keras.initializers.orthogonal(seed=54))(conv_feat)

    v_sent, q_sent = coattention.ParallelCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE,
                                                     name='CoAttention_Sentence')([image_feat, core_output])

    # Recursive Attention
    z = coattention.RecursiveCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE)(
        [q_word, v_word, q_phrase, v_phrase, q_sent, v_sent])

    result = tf.keras.layers.Dense(C.NUM_CLASSES, activation='softmax', name="result_Dense",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=91),
                                   input_shape=(2 * C.HIDDEN_SIZE,))(z)

    model = tf.keras.models.Model(inputs=[image_input, ques_input, ques_mask], outputs=result, name=C.MODEL_NAME)
    model.build(input_shape=(
        (None, C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE), (None, C.QUES_SEQ_LEN), (None, C.QUES_SEQ_LEN)
    ))

    return model


def BERT_SharedHieCoAtt(C):
    # inputs 
    image_input = tf.keras.layers.Input(
        shape=(196, 512,),
        name='Image_Input'
    )
    ques_input = tf.keras.layers.Input(
        shape=(max_len,),
        dtype='int32',
        name='Question_Input'
    )
    ques_mask = tf.keras.layers.Input(
        shape=(max_len,),
        dtype='int32',
        name='Question_Mask'
    )

    bert_model = transformers.TFBertModel.from_pretrained(
        C.BERT_MODEL_PATH['target'], from_pt=True
    )
    bert_model.trainable = False

    coattention = coattention.ParallelCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE, name='CoAttention')

    # Word level
    image_feat = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='tanh', name="Image_Feat_Dense",
                                       kernel_initializer=tf.keras.initializers.glorot_normal(seed=15))(image_input)
    image_feat = tf.keras.layers.Dropout(C.DROPOUT_RATE)(image_feat)

    bert_output = bert_model.bert([ques_input, ques_mask])

    word_feat = bert_output.last_hidden_state

    v_word, q_word = coattention([image_feat, word_feat])

    # Phrase level
    conv_feat = coattention.PhraseFeatures(C.HIDDEN_SIZE, C.DROPOUT_RATE,
                                           name="PhraseLevelFeatures")(word_feat)

    v_phrase, q_phrase = coattention([image_feat, conv_feat])

    # Question/Sentence level
    core_output = tf.keras.layers.LSTM(C.HIDDEN_SIZE, return_sequences=True, dropout=C.DROPOUT_RATE,
                                       kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                                       recurrent_initializer=tf.keras.initializers.orthogonal(seed=54))(conv_feat)

    v_sent, q_sent = coattention([image_feat, core_output])

    # Recursive Attention
    z = coattention.RecursiveCoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE)(
        [q_word, v_word, q_phrase, v_phrase, q_sent, v_sent])

    result = tf.keras.layers.Dense(C.NUM_CLASSES, activation='softmax', name="result_Dense",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=91),
                                   input_shape=(2 * C.HIDDEN_SIZE,))(z)

    model = tf.keras.models.Model(inputs=[image_input, ques_input, ques_mask], outputs=result)
    model.build(input_shape=(
        (None, C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE), (None, C.QUES_SEQ_LEN), (None, C.QUES_SEQ_LEN)
    ))

    return model
