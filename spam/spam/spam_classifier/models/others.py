#from keras.applications.resnet_v2 import ResNet152V2s
# 쓸만하다고 알려져있는 애들
# efficientnet
import math

# rexnet (네이버꺼)
def scheduler(epoch):
    if epoch < 15: # efn5 이걸로 해보기
        return 1e-4
    else:
        return 1e-4 * math.exp(0.1 * (15 - epoch))
