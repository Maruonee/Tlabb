from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf

tf.config.list_physical_devices('GPU')