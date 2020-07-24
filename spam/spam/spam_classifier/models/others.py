#from keras.applications.resnet_v2 import ResNet152V2s
# 쓸만하다고 알려져있는 애들
# efficientnet
import tensorflow as tf

# rexnet (네이버꺼)
def scheduler(epoch):
    if epoch < 15:
        return 1e-4
    else:
        return 0.0001 * tf.math.exp(0.1 * (15 - epoch))

def scheduler10(epoch):
    if epoch < 10:
        return 1e-4
    else:
        return 0.0001 * tf.math.exp(0.1 * (10 - epoch))