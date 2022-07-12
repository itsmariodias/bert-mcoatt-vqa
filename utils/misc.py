import random
import tensorflow as tf
import numpy as np


def set_seed(seed):
    """
    Function to set seeds for consistency
    """
    print(f"\nSeed is {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
