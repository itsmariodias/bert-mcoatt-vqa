from models.build import build_model
from utils.visualize import plot_metric_graph
from utils.data_gen import VQADataGenerator

import tensorflow as tf
import transformers
import pandas as pd
import os


def build_graphs(C):
    log = pd.read_csv(C.LOG_PATH)

    plot_metric_graph(log, os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_loss.png"), metric='loss', show_val=True)

    plot_metric_graph(log, os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_accuracy.png"), metric='accuracy', show_val=True)

    if C.VERBOSE: print(f"\nLoss and metric graphs saved at {C.OUTPUT_DIR}.")


def load_callbacks(C, train_gen):
    print("\nLoading training callbacks...")
    # Load training callbacks to monitor
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=C.CHECKPOINT_PATH,
        verbose=int(C.VERBOSE),
        save_freq=C.SAVE_PERIOD * len(train_gen)
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=int(C.VERBOSE))

    history_logger = tf.keras.callbacks.CSVLogger(C.LOG_PATH, separator=",", append=True)
    print("Callbacks loaded.")

    return [model_checkpoint, early_stop, history_logger]


def train(C):
    print("\nStarting training process...")
    train_gen = VQADataGenerator(C, mode="train")

    val_gen = VQADataGenerator(C, mode="val")

    model = build_model(C)

    callbacks_list = load_callbacks(C, train_gen)

    # Train model
    print('\nTraining started...')
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=C.START_EPOCH + C.NUM_EPOCHS,
        initial_epoch=C.START_EPOCH,
        verbose=1,
        callbacks=callbacks_list
    )
    print("Training done.")

    # Plot loss and accuracy graphs after training
    build_graphs(C)

    # Save final model
    model.save(C.CHECKPOINT_PATH.format(epoch=C.START_EPOCH + C.NUM_EPOCHS))
    if C.VERBOSE: print(f"Final model saved at {C.CHECKPOINT_PATH.format(epoch=C.START_EPOCH + C.NUM_EPOCHS)}")

    return model
