#from keras.applications.resnet_v2 import ResNet152V2s
import efficientnet.keras as efn #ducky님이 알려주심

from keras.applications.resnet_v2 import ResNet50V2
# 쓸만하다고 알려져있는 애들
# efficientnet

# rexnet (네이버꺼)

if __name__ == '__main__':
    en3 = efn.EfficientNetB3() # 찾을 수가 없다. 이거 왜 홈페이지에는 있는데 여기는 없는것이지..
    en3.summary()

    print("=============================================================")
    r50 = ResNet50V2()
    r50.summary()
    # r152 = ResNet152V2s()
    # r152.summary()


# def efn(input_size, n_classes, local_weights):
    