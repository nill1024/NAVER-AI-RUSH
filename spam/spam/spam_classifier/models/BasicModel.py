from typing import Callable, List

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD, Adam
import nsml
import numpy as np
import pandas as pd
from spam.spam_classifier.models.others import scheduler

from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier # vote로 할수도 있고, 값 평균값을 내서 하는 방식도 가능할 것 같다.

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate
# from adamp import AdamP, SGDP 이거 torch로 되어있어서 찾아봐야 할듯

class BasicModel:
    """
    A basic model that first finetunes the last layer of a pre-trained network, and then unfreezes all layers and
    train them.
    """
    

    def __init__(self, network_fn: Callable, network_fn2: Callable, network_fn3: Callable dataset_cls: Dataset, dataset_kwargs, network_kwargs):
        self.data: Dataset = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))
        self.network: keras.Model = network_fn(**kwargs_or_empty_dict(network_kwargs)) #frozen_resnet(input_size = input_size, n_classes)
        self.net2: keras.Model = network_fn2(**kwargs_or_empty_dict(network_kwargs))
        self.net3: keras.Model = network_fn3(**kwargs_or_empty_dict(network_kwargs)) 
        self.debug = False

    def fit(self, epochs_finetune, epochs_full, batch_size, debug=False):
        self.debug = debug
        self.data.prepare() # rearrange와 디렉토리 초기화

        # 이 과정까지 진행했을 때 디렉토리가 정렬이 되기 때문에 이시점에서 언더샘플링 하던지

        self.network.compile(
            loss=self.loss(),
            optimizer=self.optimizer('finetune'),
            metrics=self.fit_metrics()
        )

        steps_per_epoch_train = int(self.data.len('train') / batch_size) if not self.debug else 2
        model_path_finetune = 'model_finetuned.h5'
        train_gen, val_gen = self.data.train_val_gen(batch_size) # 이 부분이 train data가 들어오는 부분임
        
        #nsml.save(checkpoint='pretuned')# 근데 이게 왜 두개 있는지..?

    # 이 부분에서 데이터 수정이 들어가야 할 것 같다 아마도
    # fit generator 그냥 학습 시키는 거다. 그니까 여기가 사실상 fit 부분임

        self.network.fit_generator(generator=train_gen, #참고: 이미 network가 resnet 객체인고로 필요없당.
                                   steps_per_epoch=steps_per_epoch_train,
                                   epochs=epochs_finetune,
                                   callbacks=self.callbacks_ft(
                                       model_path=model_path_finetune,
                                       model_prefix='last_layer_tuning',
                                       patience=5,
                                       val_gen=val_gen,
                                       classes=self.data.classes),
                                   validation_data=val_gen,
                                   use_multiprocessing=True,
                                   workers=20)  # TODO change to be dependent on n_cpus

        # finetuning 먼저 하는 듯
        # finetuning은 초기값을 주기 위해서 하는 과정이라고 보면 된다.
        # 합성곱 신경망의 미세조정(finetuning): 무작위 초기화 대신, 신경망을 ImageNet 1000 데이터셋 등으로 미리 학습한 신경망으로 초기화합니다. 
        # 학습의 나머지 과정들은 평상시와 같습니다.

        self.network.load_weights(model_path_finetune)
        self.unfreeze()

        self.network.compile(
            loss=self.loss(),
            optimizer=self.optimizer('full'),
            metrics=self.fit_metrics()
        )

        model_path_full = 'model_full.h5'
        self.network.fit_generator(generator=train_gen,
                                   steps_per_epoch=steps_per_epoch_train,
                                   epochs=epochs_full,
                                   callbacks=self.callbacks(
                                       model_path=model_path_full,
                                       model_prefix='full_tuning',
                                       val_gen=val_gen,
                                       patience=10,
                                       classes=self.data.classes),
                                   validation_data=val_gen,
                                   use_multiprocessing=True,
                                   workers=20) # 아아 그리고 말인데 learning rate scheduler는 여기다만 적용하는게 어덜까?

        self.network.load_weights(model_path_full)

        val_pred = self.network.predict_generator(val_gen)
        print(val_pred.shape)
        print(val_pred)
        
        nsml.save(checkpoint='full') #이거 부를 때마다 모델 체크포인트를 남길 수 있는데 나중에 가면 많이 써야할 것 같음.
        #아마 콜백이 있어서 기존 체크포인트가 best였던 모양인데 원래 콜백은 그냥 기본이라고 볼 수 있으므로 full


        print('Done')
        self.metrics(gen=val_gen)

    def unfreeze(self) -> None:
        for layer in self.network.layers:
            layer.trainable = True

    def loss(self) -> str:
        loss = keras.losses.CategoricalCrossentropy()
        return loss

    def optimizer(self, stage: str) -> keras.optimizers.Optimizer:
        return {
            # 'finetune': SGDP(lr=1e-4, momentum=0.9), #여기 있었네 learning rate
            # 'full': AdamP(lr=1e-4), #요것들 사용법 좀 알아봐야 함

            #모멘텀은 원래 0.9를 쓰는게 일반적이라고 하는데, 어차피 다른걸 바꿀게 많기때문에 그냥 두는 것이 아마 좋을 듯

            'finetune': SGD(lr=1e-4, momentum=0.9), #여기 있었네 learning rate
            'full': Adam(lr=1e-4)
        }[stage]
    
    def fit_metrics(self) -> List[str]:
        return ['accuracy']


    def callbacks(self, model_path, model_prefix, patience, classes, val_gen): #이걸 수정하면 fit에 들어가는 옵션을 직접적으로 고치는 것임
        callbacks = [
            # TODO Change to the score we're using for ModelCheckpoint
            ReduceLROnPlateau(patience=3),  # TODO Change to cyclic LR
            NSMLReportCallback(prefix=model_prefix),
            Metrics(name=model_prefix,
                    classes=classes,
                    val_data=val_gen,
                    n_val_samples=self.data.len('val') if not self.debug else 256),
            ModelCheckpoint(model_path, monitor=f'val/{model_prefix}/macro avg/f1-score', verbose=1,
                            save_best_only=True, mode='max'),
            # TODO Change to the score we're using for ModelCheckpoint
            #LearningRateScheduler(scheduler, verbose=1), # 이렇게 하는거 맞나용..?
            EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
            
            #추가

        ]
        return callbacks

    def callbacks_ft(self, model_path, model_prefix, patience, classes, val_gen): #이걸 수정하면 fit에 들어가는 옵션을 직접적으로 고치는 것임
        callbacks = [
            # TODO Change to the score we're using for ModelCheckpoint
            ReduceLROnPlateau(patience=3),  # TODO Change to cyclic LR
            NSMLReportCallback(prefix=model_prefix),
            Metrics(name=model_prefix,
                    classes=classes,
                    val_data=val_gen,
                    n_val_samples=self.data.len('val') if not self.debug else 256),
            ModelCheckpoint(model_path, monitor=f'val/{model_prefix}/macro avg/f1-score', verbose=1,
                            save_best_only=True, mode='max'),
            # TODO Change to the score we're using for ModelCheckpoint

            EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
            
        ]
        return callbacks

    def evaluate(self, test_dir: str) -> pd.DataFrame:
        """

        이거 오로지 테스트에 쓰이는 함수라는거

        Args:
            test_dir: Path to the test dataset.

        Returns:
            ret: A dataframe with the columns filename and y_pred. One row is the prediction (y_pred)
                for that file (filename). It is important that this format is used for NSML to be able to evaluate
                the model for the leaderboard.

        """
        gen, filenames = self.data.test_gen(test_dir=test_dir, batch_size=64) # 지금 이게 test data를 가지고 계산하는 애임
        y_pred = self.network.predict_generator(gen) # 아무래도 fit된 모델 기반으로 예측값을 만드는 것 같음. 이걸 기준으로 다시 모델에 피드백시킬 수 있을 것 같다.(특히 unlabeled data)

        ret = pd.DataFrame({'filename': filenames, 'y_pred': np.argmax(y_pred, axis=1)})

        
        return ret

    def metrics(self, gen) -> None: # 현재 validation dataset을 이걸로 검증함
        """
        Generate and print metrics.

        Args:
            gen: Keras generator for which to get metrics
            n_batches: How many batches that can be fetched from the data generator.
        """
        y_true, y_pred = evaluate(data_gen=gen, model=self.network) #주의 : 위에 있는 evaluate랑 다른 함수다. 이거 ㅁ르면 빅엿 먹는다.
        y_true, y_pred = [np.argmax(y, axis=1) for y in [y_true, y_pred]]

        cls_report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )
        print(
            f'Classification report for validation dataset:\n-----------------------------\n{cls_report}\n=============\n')


def bind_model(model: BasicModel):
    """
    Utility function to make the model work with leaderboard submission.
    """
    #data = 상위 폴더의 class dataset을 말함
    #network = keras.model

    def load(dirname, **kwargs):
        model.network.load_weights(f'{dirname}/model')

    def save(dirname, **kwargs):
        filename = f'{dirname}/model'
        print(f'Trying to save to {filename}')
        
        model.network.save_weights(filename)

    def infer(test_dir, **kwargs):
        return model.evaluate(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
