from models.build import build_model
from utils.visualize import plot_metric_graph
from utils.data_gen import VQADataGenerator

import tensorflow as tf
import pandas as pd
import os


# custom callback to track lr decay per epoch
class AdamLearningRateTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print("\nCurrent learning rate: {:0.7f}".format(current_decayed_lr))


def build_graphs(C):
    log = pd.read_csv(C.LOG_PATH)

    plot_metric_graph(log,
                      os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_loss.png"),
                      metric='loss',
                      show_val=True)

    if C.ANSWERS_TYPE == "softscore":
        plot_metric_graph(log,
                          os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_accuracy.png"),
                          metric='score',
                          show_val=True)
    else:
        plot_metric_graph(log,
                          os.path.join(C.OUTPUT_DIR, f"{C.MODEL_NAME}_accuracy.png"),
                          metric='accuracy',
                          show_val=True)

    if C.VERBOSE: print(f"\nLoss and metric graphs saved at {C.OUTPUT_DIR}.")


def load_callbacks(C, train_gen):
    # custom lr scheduler based on MCAN
    def lr_schedule(epoch, lr):
        if epoch > 9 and epoch % 2 == 0:
            return lr * 0.2

        if epoch < 4:
            return C.BASE_LEARNING_RATE * (epoch + 1) / 4

        return lr

    print("\nLoading training callbacks...")
    # Load training callbacks to monitor
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=C.CHECKPOINT_PATH,
        verbose=int(C.VERBOSE),
        save_freq=C.SAVE_PERIOD * len(train_gen)
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=int(C.VERBOSE))

    history_logger = tf.keras.callbacks.CSVLogger(C.LOG_PATH, separator=",", append=True)

    callbacks_list = [model_checkpoint, early_stop, history_logger]

    if C.ADAM_WEIGHT_DECAY > 0:
        lr_tracker = AdamLearningRateTracker()
        callbacks_list.append(lr_tracker)
    else:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks_list.append(lr_scheduler)

    print("Callbacks loaded.")
    return callbacks_list


def train(C):
    print("\nStarting training process...")
    train_gen = VQADataGenerator(C, mode="train")

    val_gen = VQADataGenerator(C, mode="val")

    model = build_model(C, train_gen)

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
