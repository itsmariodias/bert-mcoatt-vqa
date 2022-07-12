from models.bert_mcoatt import coattention

import tensorflow as tf
import transformers

CUSTOM_OBJECTS = {
    'CoAttention': coattention.CoAttention,
    'TFBertMainLayer': transformers.TFBertMainLayer,
}


def BERT_MultipleCoAtt(C):
    # Inputs 
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

    # Feature extraction
    image_feat = tf.keras.layers.Dense(C.HIDDEN_SIZE, name="Image_Feat_Dense")(image_input)
    image_feat = tf.keras.layers.BatchNormalization(name="BatchNorm_image_feat")(image_feat)
    image_feat = tf.keras.layers.Activation('relu', name="ReLU_Activation_image_feat")(image_feat)
    image_feat = tf.keras.layers.Dropout(C.DROPOUT_RATE, name="Dropout_image_feat")(image_feat)

    bert_output = bert_model.bert([ques_input, ques_mask])

    word_feat = bert_output.last_hidden_state
    word_feat = tf.keras.layers.LayerNormalization(name="LayerNorm_word_feat")(word_feat)

    # Recursive Attention
    v_1, q_1 = coattention.CoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE, name='CoAttention_1')([image_feat, word_feat])
    feat_1 = tf.keras.layers.Add(name="Add_1")([q_1, v_1])
    feat_1 = tf.keras.layers.LayerNormalization(name="LayerNorm_feat_1")(feat_1)
    h_1 = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='relu', name="h_Dense_1")(feat_1)
    h_1 = tf.keras.layers.Dropout(C.DROPOUT_RATE, name="Dropout_h_1")(h_1)

    v_2, q_2 = coattention.CoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE, name='CoAttention_2')([image_feat, word_feat])
    feat_2 = tf.keras.layers.Concatenate(name="Concatenate_feat_2")(
        [tf.keras.layers.Add(name="Add_2")([q_2, v_2]), h_1])
    feat_2 = tf.keras.layers.LayerNormalization(name="LayerNorm_feat_2")(feat_2)
    h_2 = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='relu', name="h_Dense_2")(feat_2)
    h_2 = tf.keras.layers.Dropout(C.DROPOUT_RATE, name="Dropout_h_2")(h_2)

    v_3, q_3 = coattention.CoAttention(C.HIDDEN_SIZE, C.DROPOUT_RATE, name='CoAttention_3')([image_feat, word_feat])
    feat_3 = tf.keras.layers.Concatenate(name="Concatenate_feat_3")(
        [tf.keras.layers.Add(name="Add_3")([q_3, v_3]), h_2])
    feat_3 = tf.keras.layers.LayerNormalization(name="LayerNorm_feat_3")(feat_3)
    h_3 = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='relu', name="h_Dense_3")(feat_3)
    h_3 = tf.keras.layers.Dropout(C.DROPOUT_RATE, name="Dropout_h_3")(h_3)

    h = tf.keras.layers.LayerNormalization(name="LayerNorm_h")(h_3)
    z = tf.keras.layers.Dense(2 * C.HIDDEN_SIZE, activation='relu', name="z_Dense")(h)
    z = tf.keras.layers.LayerNormalization(name="LayerNorm_z")(z)

    result = tf.keras.layers.Dense(C.NUM_CLASSES, activation='softmax', name="result_Dense")(z)

    model = tf.keras.models.Model(inputs=[image_input, ques_input, ques_mask], outputs=result, name=C.MODEL_NAME)
    model.build(input_shape=((None, C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE), (None, C.QUES_SEQ_LEN), (None, C.QUES_SEQ_LEN)))

    return model
