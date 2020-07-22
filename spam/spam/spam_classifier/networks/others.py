
from keras.applications.resnet_v2 import ResNet152V2s
from keras.applications import EfficientNetB3


#쓸만하다고 알려져있는 애들
#efficientnet (B3)


#rexnet (네이버꺼)

if __name__ == '__main__':
    en3 = EfficientNetB3() # 찾을 수가 없다. 이거 왜 홈페이지에는 있는데 여기는 없는것이지..
    en3.summary()

    print("=============================================================")

    r152 = ResNet152V2s()
    r152.summary()