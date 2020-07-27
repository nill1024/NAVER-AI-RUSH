from keras import Input, Model
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_resnet_v2 import InceptionResNetV2

#from kears.applications.resnet_v2 import ResNet152V2
from keras.layers import Flatten, Dense
from os import path
import efficientnet.keras as efn

def frozen_resnet(input_size, n_classes, local_weights="/resnets/resnet50v2_notop.h5"):
    if local_weights and path.exists(local_weights):
        print(f'Using {local_weights} as local weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size),
            weights=local_weights)
    else:
        print(
            f'Could not find local weights {local_weights} for ResNet. Using remote weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size))
    #여기까지 초기값 주는 부분

    for layer in model_.layers:
        layer.trainable = False # 전이학습을 위해 freeze 한다는것 같다.

    #좀 의문인게 freezing할거면 로컬웨잇을 쓰겠다는건데 그러면 학습이 안되는거 아닌감요? 심지어 코드상으로는 아무리 봐도 전체 레이어 프리징인데
    #근데 에폭 늘렸을때 학습이 잘 되는거 보면 그것도 이상함
    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    
    frozen_model = Model(model_.input, x)

    return frozen_model



# en3 = efn.EfficientNetB3() # 찾을 수가 없다. 이거 왜 홈페이지에는 있는데 여기는 없는것이지..
# en3.summary()

"""
이 모델에는 'channels_first' 데이터 포맷(채널, 높이, 넓이)과 'channels_last' 데이터 포맷(높이, 넓이, 채널) 둘 모두 사용할 수 있습니다.

이 모델의 디폴트 인풋 사이즈는 224x224입니다.

인수
include_top: 네트워크의 최상단에 완전 연결 레이어를 넣을지 여부.

weights: None (임의의 초기값 설정) 혹은 'imagenet' (ImageNet에 대한 선행 학습) 중 하나.
초기값이 그니까 저렇게 되어있다는 얘긴데 

input_tensor: 모델의 이미지 인풋으로 사용할 수 있는 선택적 케라스 텐서 (다시말해, layers.Input()의 아웃풋).
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
input_shape: 선택적 형태 튜플로, include_top이 False일 경우만 특정하십시오.
(그렇지 않다면 인풋의 형태가 (224, 224, 3)이고 'channels_last' 데이터 포맷을 취하거나 혹은 인풋의 형태가 (3, 224, 224)이고 'channels_first' 데이터 포맷을 취해야 합니다). 인풋 채널이 정확히 3개여야 하며 넓이와 높이가 32 미만이어서는 안됩니다. 예시. (200, 200, 3)은 유효한 값입니다.
pooling: 특성추출을 위한 선택적 풀링 모드로, include_top이 False일 경우 유효합니다.
None은 모델의 아웃풋이 마지막 컨볼루션 레이어의 4D 텐서 아웃풋임을 의미합니다.
'avg'는 글로벌 평균값 풀링이 마지막 컨볼루션 레이어의 아웃풋에 적용되어 모델의 아웃풋이 2D 텐서가 됨을 의미합니다.
'max'는 글로벌 최대값 풀링이 적용됨을 의미합니다.
classes: 이미지를 분류하기 위한 선택적 클래스의 수로, include_top이 True일 경우, 그리고 weights 인수가 따로 정해지지 않은 경우만 특정합니다.

"""

def frozen_resnet_i2(input_size, n_classes):
    model_ = InceptionResNetV2(
            include_top=False,
            input_tensor=Input(shape=input_size),
            )
    #여기까지 초기값 주는 부분
    #다른거 다 똑같이 하면 되는데 local weight같은 경우에 한번 찾아봐야 함

    #돌려봤는데, 인터넷에서 알아서 초기값을 받아온다. 그냥 괜찮은듯 성능도 한번 돌려본 바에 의하면 좋고

    for layer in model_.layers:
        layer.trainable = False # 전이학습을 위해 freeze 한다는것 같다.

    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    
    frozen_model = Model(model_.input, x)

    return frozen_model

def frozen_efnet3(input_size, n_classes):
    model_ = efn.EfficientNetB3(
            include_top=False,
            input_tensor=Input(shape=input_size),
            )
    #여기까지 초기값 주는 부분
    #다른거 다 똑같이 하면 되는데 local weight같은 경우에 한번 찾아봐야 함

    #돌려봤는데, 인터넷에서 알아서 초기값을 받아온다. 그냥 괜찮은듯 성능도 한번 돌려본 바에 의하면 좋고

    for layer in model_.layers:
        layer.trainable = False # 전이학습을 위해 freeze 한다는것 같다.

    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    
    frozen_model = Model(model_.input, x)

    return frozen_model


def frozen_efnet5(input_size, n_classes):
    model_ = efn.EfficientNetB5(
            include_top=False,
            input_tensor=Input(shape=input_size),
            )
    #여기까지 초기값 주는 부분
    #다른거 다 똑같이 하면 되는데 local weight같은 경우에 한번 찾아봐야 함

    #돌려봤는데, 인터넷에서 알아서 초기값을 받아온다. 그냥 괜찮은듯 성능도 한번 돌려본 바에 의하면 좋고

    for layer in model_.layers:
        layer.trainable = False # 전이학습을 위해 freeze 한다는것 같다.

    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    
    frozen_model = Model(model_.input, x)

    return frozen_model