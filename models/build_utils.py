import tensorflow as tf
import transformers


# Custom loss based on BCE loss. Adapted from pytorch
def loss(y_true, y_pred):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    ce_loss = tf.reduce_mean(cross_entropy)
    return ce_loss*y_pred.shape[1]


# Custom score for VQA based on Bottom-Up Top-Down Model
def score(y_true, y_pred):
    n = y_pred.shape[1]
    y_pred = tf.argmax(y_pred, axis=1)
    one_hots = tf.one_hot(y_pred, n)
    scores = tf.multiply(y_true, one_hots)
    return tf.reduce_sum(scores, axis=-1)


# Custom lr scheduler based on the scheduler used to train BERT.
def get_lr_schedule(C, data_gen):
    """
    Learning rate scheduler adapted from BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding
    Ref: https://arxiv.org/abs/1810.04805
    """
    num_epochs = len(data_gen)
    num_steps = num_epochs * (C.END_EPOCH - C.START_EPOCH)

    if C.START_EPOCH > 0:
        warmup_steps = 0
    else:
        warmup_steps = int(num_steps * 0.1)

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=C.CURRENT_LEARNING_RATE,
                                                                decay_steps=num_steps - warmup_steps,
                                                                end_learning_rate=0.0,
                                                                power=1.0)

    if warmup_steps > 0:
        lr_schedule = transformers.WarmUp(initial_learning_rate=C.CURRENT_LEARNING_RATE,
                                          decay_schedule_fn=lr_schedule,
                                          warmup_steps=warmup_steps)

    return lr_schedule
