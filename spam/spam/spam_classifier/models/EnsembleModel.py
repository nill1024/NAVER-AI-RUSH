from typing import Callable, List

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD, Adam
import nsml
import numpy as np
import pandas as pd
from spam.spam_classifier.models.others import scheduler
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier # vote로 할수도 있고, 값 평균값을 내서 하는 방식도 가능할 것 같다.

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate
# from adamp import AdamP, SGDP 이거 torch로 되어있어서 찾아봐야 할듯

class EnsembleModel:
    """
    A basic model that first finetunes the last layer of a pre-trained network, and then unfreezes all layers and
    train them.
    """
    
    def __init__(self, network_fn: Callable, network_fn2: Callable, network_fn3: Callable, network_fn4: Callable, dataset_cls: Dataset, dataset_kwargs, network_kwargs):
        self.data: Dataset = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))
        self.net1: keras.Model = network_fn(**kwargs_or_empty_dict(network_kwargs)) #frozen_resnet(input_size = input_size, n_classes)
        self.net2: keras.Model = network_fn2(**kwargs_or_empty_dict(network_kwargs))
        self.net3: keras.Model = network_fn3(**kwargs_or_empty_dict(network_kwargs))
        self.net4: keras.Model = network_fn4(**kwargs_or_empty_dict(network_kwargs))
        self.debug = False

    def fit(self, epochs_finetune, epochs_full, batch_size, debug=False):
        self.debug = debug
        # self.data.prepare() # rearrange와 디렉토리 초기화

        # # 이 과정까지 진행했을 때 디렉토리가 정렬이 되기 때문에 이시점에서 언더샘플링 하던지

        # # self.net1.compile(
        # #     loss=self.loss(),
        # #     optimizer=self.optimizer('finetune'),
        # #     metrics=self.fit_metrics()
        # # )

        # steps_per_epoch_train = int(self.data.len('train') / batch_size) if not self.debug else 2
        
        # train_gen, val_gen, unl_gen, unl_files = self.data.train_val_gen(batch_size) # 이 부분이 train data가 들어오는 부분임
        
        #nsml.save(checkpoint='pretuned')# 근데 이게 왜 두개 있는지..?

        # 이 부분에서 데이터 수정이 들어가야 할 것 같다 아마도
        # fit generator 그냥 학습 시키는 거다. 그니까 여기가 사실상 fit 부분임

        # model_path_finetune = 'model_finetuned.h5'
        # self.net1.fit_generator(generator=train_gen, #참고: 이미 network가 resnet 객체인고로 필요없당.
        #                            steps_per_epoch=steps_per_epoch_train,
        #                            epochs=1,
        #                            callbacks=self.callbacks_ft(
        #                                model_path=model_path_finetune,
        #                                model_prefix='last_layer_tuning',
        #                                patience=5,
        #                                val_gen=val_gen,
        #                                classes=self.data.classes),
        #                            validation_data=val_gen,
        #                            use_multiprocessing=True,
        #                            workers=20)  # TODO change to be dependent on n_cpus
        # self.net1.load_weights(model_path_finetune)
        
        # # finetuning 먼저 하는 듯
        # # finetuning은 초기값을 주기 위해서 하는 과정이라고 보면 된다.
        # # 합성곱 신경망의 미세조정(finetuning): 무작위 초기화 대신, 신경망을 ImageNet 1000 데이터셋 등으로 미리 학습한 신경망으로 초기화합니다. 
        # # 학습의 나머지 과정들은 평상시와 같습니다.

        # self.unfreeze()

        # self.net1.compile(
        #     loss=self.loss(),
        #     optimizer=self.optimizer('full'),
        #     metrics=self.fit_metrics()
        # )

        # model_path_full = 'model_full.h5'
        # self.net1.fit_generator(generator=train_gen,
        #                            steps_per_epoch=steps_per_epoch_train,
        #                            epochs=1,
        #                            callbacks=self.callbacks(
        #                                model_path=model_path_full,
        #                                model_prefix='full_tuning',
        #                                val_gen=val_gen,
        #                                patience=10,
        #                                classes=self.data.classes),
        #                            validation_data=val_gen,
        #                            use_multiprocessing=True,
        #                            workers=20) 

        # self.net1.load_weights(model_path_full)

        #nsml.load(checkpoint='full',session='nill1024/spam-3/8') # 3번 resnet 92mb net 1 net_fn 1 (아닐 가능성 있으니 주의)

        nsml.load(checkpoint='full',session='nill1024/spam-3/35')
        nsml.load(checkpoint='full',session='nill1024/spam-1/53')
        nsml.load(checkpoint='best',session='nill1024/spam-1/10')
        nsml.load(checkpoint='full',session='nill1024/spam-3/66')
        
        # unl_pred1 = self.net2.predict_generator(unl_gen)
        # unl_pred2 = self.net3.predict_generator(unl_gen)
        # unl_pred3 = self.net1.predict_generator(unl_gen)

        # unl_pred = (unl_pred1+unl_pred2+unl_pred3)/3
        # arg_unl = np.argmax(unl_pred,axis=1)

        # print("prediction completed")
        # for i in range(len(arg_unl)):
        #     if unl_pred[i][arg_unl[i]] < 0.95:
        #         arg_unl[i] = -1

        # df_u = pd.DataFrame({'filename': unl_files, 'unl_pred': arg_unl})
        
        # print(df_u.head())

        # unl_train, unl_val = self.data.labeld_unl(df_u,batch_size)

        # model_path_full = 'model_full.h5'
        # self.net1.fit_generator(generator=unl_train,
        #                     steps_per_epoch=steps_per_epoch_train,
        #                     epochs=epochs_full,
        #                     callbacks=self.callbacks(
        #                         model_path=model_path_full,
        #                         model_prefix='full_tuning',
        #                         val_gen=val_gen, # 이거 일부러 unl_val은 안쓴다. 참고
        #                         patience=10,
        #                         classes=self.data.classes),
        #                     validation_data=val_gen,
        #                     use_multiprocessing=True,
        #                     workers=20)
        # self.net1.load_weights(model_path_full)

        nsml.save(checkpoint='full') #이거 부를 때마다 모델 체크포인트를 남길 수 있는데 나중에 가면 많이 써야할 것 같음.
        #아마 콜백이 있어서 기존 체크포인트가 best였던 모양인데 원래 콜백은 그냥 기본이라고 볼 수 있으므로 full

        print('Done')
        # self.metrics(gen=val_gen)

    def unfreeze(self) -> None:
        for layer in self.net1.layers:
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

    def evaluate(self, test_dir: str, vote=True) -> pd.DataFrame:
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
        y_pred1 = self.net1.predict_generator(gen) # 아무래도 fit된 모델 기반으로 예측값을 만드는 것 같음. 이걸 기준으로 다시 모델에 피드백시킬 수 있을 것 같다.(특히 unlabeled data)
        y_pred2 = self.net2.predict_generator(gen)
        y_pred3 = self.net3.predict_generator(gen)
        y_pred4 = self.net4.predict_generator(gen)

        y_average = (y_pred1+y_pred2+y_pred3) #모델 하나 더 추가할것. 우선은 voting만 만듬

        p1 = np.argmax(y_pred1,axis=1)
        p2 = np.argmax(y_pred2,axis=1)
        p3 = np.argmax(y_pred3,axis=1)
        p4 = np.argmax(y_pred4,axis=1)

        y_pred = np.argmax(y_average, axis=1)

        if vote == False:
            ret = pd.DataFrame({'filename': filenames, 'y_pred': y_pred})
            return ret

        for i in range(len(p1)): #voting이 싫다면 이 for문 전체를 없애면 됨
            ma = [0,0,0,0]
            ma[p1[i]] += 1
            ma[p2[i]] += 1
            ma[p3[i]] += 1
            ma[p4[i]] += 1 #efn3 에 1표를 다 주기 때문에 조금 안좋아 지는듯 함. 아무래도 단일 모델 성능이 안좋은 경우면 앙상블에서 voting방식의 성능에 영향을 줄 수 있음
                            #이 점 때문에 voting에서 한표를 다 안주는 방식으로 알고리즘을 짜게 되면, 기본적으로 average방식과 그 알고리즘이 유사해짐. average도 가중치를 주고 할 수 있는 거라
                            #물론 얘를 가중치 주는거랑 저 위에걸 가중치 주는거랑은 살짝 다르긴 함. cross_entropy 때문에
            mx = 0
            f = True
            idx = -1
            for mi in range(len(ma)): #지금 동작방식은 voting이 기본이지만, 만약 voting이 똑같을 경우 확률로 계산함
                if mx < ma[mi]:
                    mx = ma[mi]
                    idx = mi
                    f = True
                elif mx == ma[mi]:
                    f = False
            
            if f == True:
                if idx == -1:
                    print("exception occured") #현재 다 완성했는데 코드가 약간 길어서 오류가 생길 수 있으므로 함수로 만들고 테스트해보는걸 추천. 안그러면 성능이 떨어질때 뭐 때문인지 알 수 없을수 있음
                y_pred[i] = idx

        ret = pd.DataFrame({'filename': filenames, 'y_pred': y_pred})

        return ret

    def metrics(self, gen) -> None: # 현재 validation dataset을 이걸로 검증함
        """
        Generate and print metrics.

        Args:
            gen: Keras generator for which to get metrics
            n_batches: How many batches that can be fetched from the data generator.
        """
        y_true, y_pred = evaluate(data_gen=gen, model=self.net1) #주의 : 위에 있는 evaluate랑 다른 함수다. 이거 ㅁ르면 빅엿 먹는다.
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


def bind_model(model: EnsembleModel):
    """
    Utility function to make the model work with leaderboard submission.
    """
    #data = 상위 폴더의 class dataset을 말함
    #network = keras.model

    def load(dirname, **kwargs):
        try:
            if str(Path(f'{dirname}/model').stat().st_size)[0] == '2':
                model.net2.load_weights(f'{dirname}/model')
                print("net 2 loaded")
            elif str(Path(f'{dirname}/model').stat().st_size)[0] == '1':
                model.net3.load_weights(f'{dirname}/model')
                print("net 3 loaded")
            elif str(Path(f'{dirname}/model').stat().st_size)[0] == '9':
                model.net4.load_weights(f'{dirname}/model')
                print("net 4 loaded")
            elif str(Path(f'{dirname}/model').stat().st_size)[0] == '4': #현재 resnet이 net4로 로드됨
                model.net1.load_weights(f'{dirname}/model1')
                print("net 1 loaded")
        except:
            model.net1.load_weights(f'{dirname}/model1')
            model.net2.load_weights(f'{dirname}/model2')
            model.net3.load_weights(f'{dirname}/model3')
            model.net4.load_weights(f'{dirname}/model4')


    def save(dirname, **kwargs):
        filename = f'{dirname}/model'
        print(f'Trying to save to {filename}'+'1')
        model.net1.save_weights(filename+'1')
        print(f'Trying to save to {filename}'+'2')
        model.net2.save_weights(filename+'2')
        print(f'Trying to save to {filename}'+'3')
        model.net3.save_weights(filename+'3')
        print(f'Trying to save to {filename}'+'4')
        model.net4.save_weights(filename+'4')

    def infer(test_dir, **kwargs):
        return model.evaluate(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
