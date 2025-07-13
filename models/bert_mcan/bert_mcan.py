"""
Deep Modular Co-Attention Networks for Visual Question Answering
Ref: https://arxiv.org/abs/1906.10770
Ref: https://github.com/MILVLG/mcan-vqa

This implementation is based on a Tensorflow-based version of the code given in the repository linked above.
"""

from models.bert_mcan.decoder import Decoder, DecoderBlock
from models.bert_mcan.mcan_utils import AttentionReduction, MLP
from models.build_utils import loss, score

import tensorflow as tf
import transformers


CUSTOM_OBJECTS = {
    'loss': loss,
    'score': score,
    'MLP': MLP,
    'DecoderBlock': DecoderBlock,
    'Decoder': Decoder,
    'AttentionReduction': AttentionReduction,
    'TFBertMainLayer': transformers.TFBertMainLayer,
    'AdamWeightDecay': transformers.AdamWeightDecay,
    'WarmUp': transformers.WarmUp,
}


def BERT_MCAN(C):
    # Image features from Bottom-Up Attention Model
    image_input = tf.keras.layers.Input(
        shape=(C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE,),
        name='image_features'
    )
    # Image masks to indicate which regions to exclude due to adaptive K method.
    image_mask = tf.keras.layers.Input(
        shape=(C.IMG_SEQ_LEN,),
        name='image_masks'
    )

    # Encoded token ids from BERT tokenizer.
    question_input = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype=tf.int32,
        name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    question_mask = tf.keras.layers.Input(
        shape=(C.QUES_SEQ_LEN,),
        dtype=tf.int32,
        name="attention_masks"
    )

    # initialize the main model modules
    img_feat_linear = tf.keras.layers.Dense(
        C.HIDDEN_SIZE,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=C.INIT_RANGE),
        name="img_feat_linear",
    )

    bert_config = transformers.BertConfig(
        hidden_size=C.HIDDEN_SIZE,
        num_hidden_layers=C.NUM_LAYERS,
        num_attention_heads=C.NUM_HEADS,
        intermediate_size=C.FF_SIZE,
        hidden_dropout_prob=C.DROPOUT_RATE,
        attention_probs_dropout_prob=C.DROPOUT_RATE,
        hidden_act=C.ACTIVATION,
        layer_norm_eps=C.LAYER_NORM_EPSILON,
        initializer_range=C.INIT_RANGE,
        max_position_embeddings=512,
    )

    if C.PRETRAIN:
        bert_model = transformers.TFBertModel.from_pretrained(
            C.BERT_MODEL_PATH['target'],
            config=bert_config,
            from_pt=True
        )
    else:
        bert_model = transformers.TFBertModel(config=bert_config)

    decoder = Decoder(
        C.NUM_LAYERS,
        C.HIDDEN_SIZE,
        C.NUM_HEADS,
        C.FF_SIZE,
        C.DROPOUT_RATE,
        C.ACTIVATION,
        C.LAYER_NORM_EPSILON,
        C.INIT_RANGE,
        name="decoder"
    )

    attflat_lang = AttentionReduction(
        C.FLAT_MLP_SIZE,
        C.FLAT_OUT_SIZE,
        C.DROPOUT_RATE,
        C.ACTIVATION,
        C.INIT_RANGE,
        name="flatten_question_feat"
    )

    attflat_img = AttentionReduction(
        C.FLAT_MLP_SIZE,
        C.FLAT_OUT_SIZE,
        C.DROPOUT_RATE,
        C.ACTIVATION,
        C.INIT_RANGE,
        name="flatten_image_feat"
    )

    proj_norm = tf.keras.layers.LayerNormalization(
        epsilon=C.LAYER_NORM_EPSILON,
        name="layernorm_result"
    )

    proj = tf.keras.layers.Dense(
        C.NUM_CLASSES,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=C.INIT_RANGE),
        name="dense_result"
    )

    # designing the model architecture
    image_feat = img_feat_linear(image_input)

    bert_output = bert_model.bert([question_input, question_mask])
    encoder_output = bert_output.last_hidden_state

    decoder_output = decoder([image_feat, encoder_output], [image_mask, question_mask])

    lang_feat = attflat_lang(encoder_output, question_mask)
    image_feat = attflat_img(decoder_output, image_mask)

    proj_feat = lang_feat + image_feat
    proj_feat = proj_norm(proj_feat)
    result = proj(proj_feat)

    model = tf.keras.models.Model(
        inputs=[image_input, image_mask, question_input, question_mask],
        outputs=result,
        name=C.MODEL_NAME
    )

    model.build(
        input_shape=(
            (None, C.IMG_SEQ_LEN, C.IMG_EMBED_SIZE),
            (None, C.IMG_SEQ_LEN),
            (None, C.QUES_SEQ_LEN),
            (None, C.QUES_SEQ_LEN),
        )
    )

    return model
