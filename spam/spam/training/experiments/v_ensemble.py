
from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.EnsembleModel import EnsembleModel
from spam.spam_classifier.networks.ResNet50 import frozen_efnet5
from spam.spam_classifier.networks.ResNet50 import frozen_resnet_i2
from spam.spam_classifier.networks.ResNet50 import frozen_resnet
from spam.spam_classifier.networks.ResNet50 import frozen_efnet3

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': EnsembleModel,
    'fit_kwargs': {
        'batch_size': 128, #resnet계열은 128, efn3는 64, efn5는 64는 모르겠고 32, efn7은 16정도 나와야 함(왠지는 잘...)
        'epochs_finetune': 3,
        'epochs_full': 25, # efn의 경우 에폭 15~20, resnet계열은 에폭 30정도면 충분하고 그 밑으로 줘도 대개는 잘 됨
        'debug': False # 디버그하고싶으면 이거 True 로 하면 됨.
    },
    'model_kwargs': {
        'network_fn': frozen_resnet, #어떤 모델을 선택할지 사실상 여기서 조정하면 됨.
        'network_fn2' : frozen_resnet_i2,
        'network_fn3' : frozen_efnet5,
        'network_fn4' : frozen_resnet,
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
