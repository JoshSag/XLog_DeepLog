import numpy as np
import tensorflow as tf
def seed(_seed=0):
    np.random.seed(_seed)
    tf.random.set_seed(_seed)

# K.clear_session()
# seed(_seed=1)