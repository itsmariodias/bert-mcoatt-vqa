from models.bert_hiecoatt import bert_hiecoatt
from models.bert_mcoatt import bert_mcoatt
from models.bert_mcan import bert_mcan
from models.bert_hiealtcoatt import bert_hiealtcoatt
from models.build_utils import get_lr_schedule, loss, score

import tensorflow as tf
import transformers
import os


def build_model(C, data_gen):
    if C.START_EPOCH > 0:
        # If resuming
        if C.VERBOSE: print("\nLoading model...")
        if C.MODEL_TYPE == "bert_mcoatt":
            custom_objects = bert_mcoatt.CUSTOM_OBJECTS
        elif C.MODEL_TYPE in ["bert_hiecoatt", "bert_hiecoatt_shared"]:
            custom_objects = bert_hiecoatt.CUSTOM_OBJECTS
        elif C.MODEL_TYPE == "bert_mcan":
            custom_objects = bert_mcan.CUSTOM_OBJECTS
        elif C.MODEL_TYPE == "bert_hiealtcoatt":
            custom_objects = bert_hiealtcoatt.CUSTOM_OBJECTS
        else:
            print(f'ERROR: Invalid model type. Given {C.MODEL_TYPE}.')
            exit(-1)

        model = tf.keras.models.load_model(
                C.CHECKPOINT_PATH.format(epoch=C.START_EPOCH), 
                custom_objects=custom_objects
            )

    else:
        # If starting from scratch
        if C.VERBOSE: print("\nBuilding model...")

        # To address cases where user wants to train on bert-large or default hiealtcoatt model
        if C.HIDDEN_SIZE == 1024 and C.NUM_LAYERS == 24 and C.NUM_HEADS == 16:
            if C.VERBOSE: print("WARNING: Configuration specifies bert-large-uncased model, loading the same...")
            C.BERT_MODEL_PATH['target'] = "bert-large-uncased"
        if C.HIDDEN_SIZE == 1024:
            if C.VERBOSE: print(
                "WARNING: There is no compact BERT model with size = 1024, loading bert-base-uncased instead...")
            C.BERT_MODEL_PATH['target'] = "bert-base-uncased"

        if C.MODEL_TYPE == "bert_hiecoatt":
            model = bert_hiecoatt.BERT_HieCoAtt(C)
        elif C.MODEL_TYPE == "bert_hiecoatt_shared":
            model = bert_hiecoatt.BERT_SharedHieCoAtt(C)
        elif C.MODEL_TYPE == "bert_mcoatt":
            model = bert_mcoatt.BERT_MultipleCoAtt(C)
        elif C.MODEL_TYPE == "bert_mcan":
            model = bert_mcan.BERT_MCAN(C)
        elif C.MODEL_TYPE == "bert_hiealtcoatt":
            model = bert_hiealtcoatt.BERT_HieAltCoAtt(C)
        else:
            print(f'ERROR: Invalid model type. Given {C.MODEL_TYPE}.')
            exit(-1)

        # Load optimizer for training, we use Adam for all our models.
        # If using weight decay, load the optimizer used by BERT from transformers.
        if C.ADAM_WEIGHT_DECAY > 0:
            lr_schedule = get_lr_schedule(C, data_gen)
            optimizer = transformers.AdamWeightDecay(
                learning_rate=lr_schedule,
                weight_decay_rate=C.ADAM_WEIGHT_DECAY,
                beta_1=C.ADAM_BETA_1,
                beta_2=C.ADAM_BETA_2,
                epsilon=C.ADAM_EPSILON,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=C.BASE_LEARNING_RATE,
                beta_1=C.ADAM_BETA_1,
                beta_2=C.ADAM_BETA_2,
                epsilon=C.ADAM_EPSILON
            )

        if C.VERBOSE: print("\nCompiling model...")
        if C.ANSWERS_TYPE == "softscore":
            model.compile(loss=loss, optimizer=optimizer, metrics=[score])
        else:
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        # Create all the weights prior to account for an issue where optimizer weights fail to load when resuming
        # training
        model.optimizer._create_all_weights(model.trainable_variables)

        # remove log file if starting from scratch
        if os.path.exists(C.LOG_PATH):
            os.remove(C.LOG_PATH)
    
    if C.VERBOSE: model.summary()

    if C.VERBOSE: print("Model loaded successfully.")
    
    return model
