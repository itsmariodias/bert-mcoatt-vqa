from models.bert_hiecoatt import bert_hiecoatt
from models.bert_mcoatt import bert_mcoatt

import tensorflow as tf
import os


def build_model(C):
    if C.START_EPOCH > 0:
        # If resuming
        if C.VERBOSE: print("\nLoading model...")
        if C.MODEL_TYPE == "bert_mcoatt":
            custom_objects = bert_mcoatt.CUSTOM_OBJECTS
        elif C.MODEL_TYPE in ["bert_hiecoatt", "bert_hiecoatt_shared"]:
            custom_objects = bert_hiecoatt.CUSTOM_OBJECTS
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
        if C.MODEL_TYPE == "bert_hiecoatt":
            model = bert_hiecoatt.BERT_HieCoAtt(C)
        elif C.MODEL_TYPE == "bert_hiecoatt_shared":
            model = bert_hiecoatt.BERT_SharedHieCoAtt(C)
        elif C.MODEL_TYPE == "bert_mcoatt":
            model = bert_mcoatt.BERT_MultipleCoAtt(C)
        else:
            print(f'ERROR: Invalid model type. Given {C.MODEL_TYPE}.')
            exit(-1)

        # Load optimizer for training, we use Adam for all our models.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=C.BASE_LEARNING_RATE, 
            beta_1=C.ADAM_BETA_1, 
            beta_2=C.ADAM_BETA_2, 
            epsilon=C.ADAM_EPSILON
        )

        if C.VERBOSE: print("\nCompiling model...")
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
