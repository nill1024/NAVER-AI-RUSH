import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Tuple
from warnings import warn

import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
import pandas as pd
from nsml.constants import DATASET_PATH


class EmptyContentError(Exception):
    pass


UNLABELED = -1


"""

시도해 볼 것 :
데이터 랜덤샘플링 (잘 될지는 모르겠음, 왜냐면 epoch을 늘리면 어차피 거기서 거기가 될 것이라서)
이거 대신에 시도해 볼 수 있는 것으로 under sampling이 있는데, 내가 생각하는 그게 맞다.
normal class의 수를 줄이는 거 말함.

over sampling은 느낌상 under sampling보다 훨씬 효과가 없을 것 같은게 그 이유가 normal만 엄청나게 많고 나머지가 적은거라
normal 외에 다른 모든걸 더 올려야하는 over보다는 훨 좋을 수밖에 없음


efficientnet : 많이들 쓰는거 보니 좋은 것 같음. 근데 모델 적용보다는 데이터셋 샘플링이 더 중요할것 같음

앙상블
앙상블은 모델링을 종합하는 개념으로 사실상 꼭 해야하긴 함
지금은 내가 그냥 허졉이라 못하는 중

데이터 nomalization / batch normalization
nomalization의 경우 효과가 있다고 함.


아래는 1등님이 정리해주신 효과 있는것들과 없는것들

## Dataset Information
* normal: 61945
* monotone: 649
* screenshot: 1284
* unknown: 4601
* unlabeled: 128396
image size: (256, 256)
mean:  [0.55232704 0.51815085 0.48528248]
std:  [0.21313286 0.21373375 0.21965458]

우선 라벨 보는법부터 알아야 하는데, trainset의 경우에 label을 보는데에 무리가 없겠지만 testset의 라벨 혹은 f1 score를 볼 수 있는 것인지?
정확히 말하면 trainset의 recall과 precision을 알 수 있는 방법이 있을지 모르겠다 (이거 알아야 하는뎅,,)

그리고 f1 score 특성상 recall이나 precision 둘다 무난한 편이 한쪽만 더 잘나오는것보다 훨씬 우수하게 점수가 뜨기 때문에
언더샘플링이 훨씬 좋을 것 같음

아 근데 언더샘플링 하게 되면 사실상 f1 score로 모델을 train시키는건 의미가 없고 accuracy로 시켜야 할 것 같은데
이미 accuracy로 되어 있는듯?
fit 에 network.compile의 fit_metrics 참조

기억 안나서 recall이랑 precision 정리


recall : 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
precision : 모델이 True라고 분류한 것 중에서 실제 True인 것의 비율


일단 데이터가 아마 모델 자체보다 훨씬 크게 영향이 있을거라.. 이걸 핸들링하는걸 최우선으로 함.

## work
* Scheduler: ReduceLROnPlateau < CosineAnnealing
* Augmentation: horizontal flip, vertical flip, ColorJitter, AutoAugment(https://github.com/DeepVoltaire/AutoAugment)
* Normalize with mean, std from dataset
* Under-sampling


## Doesn't work
* Resize (224, 224), (386, 386)
* RandomResizedCrop
* label smoothing (0.1), label smoothing (0.05)
* Rotation
* CutMix, MixUp
* FocalLoss (edited) 



"""

class Dataset:
    """
    Basic dataset that can be used in combination with Keras fit_generator.
    Reorders the data to have one folder per class.
    """

    def __init__(self, classes, input_size):
        self.classes = classes
        self.img_size = input_size
        self.base_dir = Path(mkdtemp())
        self._len = None
        self.validation_fraction = 0.2

    def __del__(self):
        """
        Deletes the temporary folder that we created for the dataset.
        """
        shutil.rmtree(self.base_dir)

    def train_val_gen(self, batch_size: int):
        """
        Splits the train_data folder into train/val generators. Applies some image augmentation for the train dataset.

        Args:
            batch_size: int

        Returns:
            train_generator: Keras data generator.
            val_generator: Keras data generator.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=self.validation_fraction
        )

        train_generator = train_datagen.flow_from_directory( # 이 디렉토리 안에 있음
            directory=self.base_dir / 'train',
            shuffle=True,
            batch_size=batch_size, # 여기 batch_size는 인자로 들어옴
            target_size=self.img_size[:-1],
            classes=self.classes,
            subset='training')

        val_generator = train_datagen.flow_from_directory( # validation dataset 
            directory=self.base_dir / 'train',
            batch_size=batch_size,
            target_size=self.img_size[:-1],
            classes=self.classes,
            shuffle=True,
            subset='validation')
        assert self.classes == list(iter(train_generator.class_indices))

        return train_generator, val_generator

    def test_gen(self, test_dir: str, batch_size: int):
        """
        test dataset은 rearrange가 안되어 있다고 함.

        Args:
            test_dir: Path to the test dataseet.
            batch_size: Number of examples per batch. Reduce if encountering memory issues.

        Returns:
            gen: Keras generator for the test dataset.
            files: [str]
                A list of files. These are the same order as the images returned from the generator.

        """
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        files = [str(p.name) for p in (Path(test_dir) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]
        metadata = pd.DataFrame({'filename': files})
        gen = datagen.flow_from_dataframe(metadata, directory=f'{test_dir}/test_data', x_col='filename',
                                          class_mode=None, shuffle=False, batch_size=batch_size)
        return gen, files

    def len(self, dataset):
        """
        Utility function to compute the number of datapoints in a given dataset.
        """
        if self._len is None:
            self._len = {
                dataset: sum([len(files) for r, d, files in os.walk(self.base_dir / dataset)]) for dataset in
                ['train']}
            self._len['train'] = int(self._len['train'] * (1 - self.validation_fraction))
            self._len['val'] = int(self._len['train'] * self.validation_fraction)
        return self._len[dataset]

    def prepare(self):
        """
        The resulting folder structure is compatible with the Keras function that generates a dataset from folders.
        """
        dataset = 'train'
        self._initialize_directory(dataset)
        self._rearrange(dataset) #여기서 rearrange로 알아서 정렬하는 것 같음

    def _initialize_directory(self, dataset: str) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / dataset
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()

    def _rearrange(self, dataset: str) -> None:
        """
        Then rearranges the files based on the attached metadata. The resulting format is
        --
         |-train
             |-normal
                 |-img0
                 |-img1
                 ...
             |-montone
                 ...
             |-screenshot
                 ...
             |_unknown
                 ...
        """
        output_dir = self.base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                continue
            src = src_dir / 'train_data' / row['filename']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / self.classes[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)
