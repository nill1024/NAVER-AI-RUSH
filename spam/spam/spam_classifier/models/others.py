#from keras.applications.resnet_v2 import ResNet152V2s
# 쓸만하다고 알려져있는 애들
# efficientnet
import math

# rexnet (네이버꺼)
def scheduler(epoch, x = 10):
    if epoch < x: #임시
        return 1e-4
    else:
        return (1e-4) * math.exp(0.1 * (x - epoch))
