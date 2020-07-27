#log 찍어보는 테스트 세션용 

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.EnsembleModel import EnsembleModel
from spam.spam_classifier.networks.ResNet50 import frozen_efnet3
from spam.spam_classifier.networks.ResNet50 import frozen_resnet_i2
from spam.spam_classifier.networks.ResNet50 import frozen_resnet

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': EnsembleModel,
    'fit_kwargs': {
        'batch_size': 32, #이거 근데 데이터 개수를 정확히 알아야 batch size로 나눌 수 있을 것 같은뎅
        'epochs_finetune': 1,
        'epochs_full': 1, # efn의 경우 에폭 15 정도만 줘도 될 듯 아마?
        'debug': False #디버그하고싶으면 이거 True 로 하면 됨.
    },
    'model_kwargs': {
        'network_fn': frozen_resnet, #어떤 모델을 선택할지 사실상 여기서 조정하면 됨.
        'network_fn2' : frozen_resnet_i2,
        'network_fn3' : frozen_efnet3,
        'network_kwargs': {
            'input_size': input_size,
            'n_classes': len(classes)
        },
        'dataset_cls': Dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}
