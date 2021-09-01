import tensorflow as tf
from pydnet import *
import numpy as np
# tf.compat.v1.disable_v2_behavior()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.initialize_all_variables())
saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())
saver.restore(sess, './checkpoint/IROS18/')
