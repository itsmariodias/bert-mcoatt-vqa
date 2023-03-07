"""
Hierarchical Question-Image Co-Attention for Visual Question Answering
Ref: https://arxiv.org/abs/1606.00061
Ref: https://github.com/jiasenlu/HieCoAttenVQA

This implementation is a modified version of the Hierarchical Alternating Co-Attention model.
"""

from models.bert_hiealtcoatt import coattention
from models.build_utils import loss

import tensorflow as tf
import transformers


CUSTOM_OBJECTS = {
    'BCEloss': loss,
    'AlternatingCoAttention': coattention.AlternatingCoAttention,
    'CoAttendFeatures': coattention.CoAttendFeatures,
    'TFBertMainLayer': transformers.TFBertMainLayer,
}


def BERT_HieAltCoAtt(C):
    # Image features from Bottom-Up Attention Model
    image_feats = tf.keras.layers.Input(
        shape=(C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE,),
        name='image_features'
    )
    # Image masks to indicate which regions to exclude due to adaptive K method.
    image_masks = tf.keras.layers.Input(
        shape=(C.IMG_SEQ_LEN,),
        name='image_masks'
    )

    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype=tf.int32,
        name="attention_masks"
    )

    bert_model = transformers.TFBertModel.from_pretrained(
        'bert-base-uncased', output_hidden_states=True
    )
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model.bert([input_ids, attention_masks])

    # concatenate consecutive 4 hidden layers
    sequence_output_1 = tf.keras.layers.Concatenate(name="concat_first4hidden")(bert_output.hidden_states[1:5])
    sequence_output_2 = tf.keras.layers.Concatenate(name="concat_middle4hidden")(bert_output.hidden_states[5:9])
    sequence_output_3 = tf.keras.layers.Concatenate(name="concat_last4hidden")(bert_output.hidden_states[9:])

    joint_feat_1 = coattention.CoAttendFeatures(
        C.HIDDEN_SIZE, C.DROPOUT_RATE, name="coattend_feats_1"
    )([image_feats, image_masks, sequence_output_1, attention_masks])
    joint_feat_2 = coattention.CoAttendFeatures(
        C.HIDDEN_SIZE, C.DROPOUT_RATE, name="coattend_feats_2"
    )([image_feats, image_masks, sequence_output_2, attention_masks])
    joint_feat_3 = coattention.CoAttendFeatures(
        C.HIDDEN_SIZE, C.DROPOUT_RATE, name="coattend_feats_3"
    )([image_feats, image_masks, sequence_output_3, attention_masks])

    concatenate_joint = tf.keras.layers.Concatenate(
        name="concat_joint_embeds")([joint_feat_1, joint_feat_2, joint_feat_3])

    fc = tf.keras.layers.Dense(C.HIDDEN_SIZE, activation='relu', name="fc")(concatenate_joint)
    fc = tf.keras.layers.Dropout(0.5, name="final_dropout")(fc)

    result = tf.keras.layers.Dense(C.NUM_CLASSES, name="result_Dense")(fc)

    model = tf.keras.models.Model(
        inputs=[image_feats, image_masks, input_ids, attention_masks],
        outputs=result,
        name=C.MODEL_NAME
    )

    model.build(
        input_shape=(
            (None, C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE),
            (None, C.IMG_SEQ_LEN),
            (None, C.QUES_SEQ_LEN),
            (None, C.QUES_SEQ_LEN)
        )
    )

    return model