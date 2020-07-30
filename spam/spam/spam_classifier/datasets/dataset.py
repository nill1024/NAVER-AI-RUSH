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
from spam.spam_classifier.models import EnsembleModel

from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(0,1))

class EmptyContentError(Exception):
    pass


UNLABELED = -1


class Dataset:
    """
    Basic dataset that can be used in combination with Keras fit_generator.
    Reorders the data to have one folder per class.
    """

    def __init__(self, classes, input_size):
        self.classes = classes
        self.img_size = input_size # (256, 256, 3)
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
            preprocessing_function=preprocess_input, # keras.applications.resnet_v2.preprocess_input
            horizontal_flip=True,
            # 이 부분 추가됨
            vertical_flip=True,
            # featurewise_std_normalization=True, #현재 18 세션이 이게 적용되어 있음
            # samplewise_std_normalization=True,
            # This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=self.validation_fraction,
            #rescale=
        )

        # horizontal_flip: 불리언. 인풋을 무작위로 가로로 뒤집습니다.
        # vertical_flip: 불리언. 인풋을 무작위로 세로로 뒤집습니다.
        # featurewise_std_normalization: 불리언. 인풋을 각 특성 내에서 데이터셋의 표준편차로 나눕니다.
        # samplewise_std_normalization: 불리언. 각 인풋을 표준편차로 나눕니다.
        # 이거 두개 다 된다고 함

        #resampling & undersampling
        #여기서 해야 함

        train_generator = train_datagen.flow_from_directory( # 이 디렉토리 안에 있음
            directory=self.base_dir / 'train',
            shuffle=True,
            batch_size=batch_size, # 여기 batch_size는 인자로 들어옴
            target_size=self.img_size[:-1],
            classes=self.classes,
            subset='training',
            #class_mode='categorical'
            )

        # 찾아봤는데 min max scaler가 normalization에 해당된다고 함. 

        val_generator = train_datagen.flow_from_directory( # validation dataset 
            directory=self.base_dir / 'train',
            batch_size=batch_size,
            target_size=self.img_size[:-1],
            classes=self.classes,
            shuffle=True,
            subset='validation',
            )

        unl_gen = train_datagen.flow_from_directory(
            directory=self.base_dir / 'unlabeled',
            batch_size=batch_size,
            target_size=self.img_size[:-1],
            shuffle=True,
            classes=['unlabeled']
        )

        unl_files = [str(p.name) for p in (Path(self.base_dir) / 'unlabeled').glob('*.*') if p.suffix not in ['.gif', '.GIF']]

        nsml.load(checkpoint='full',session='nill1024/spam-3/6')
        nsml.load(checkpoint='full',session='nill1024/spam-3/9')
        nsml.load(checkpoint='full',session='nill1024/spam-3/8')
        
        unl_pred1 = EnsembleModel.net2.predict_generator(unl_gen)
        unl_pred2 = EnsembleModel.net3.predict_generator(unl_gen)
        unl_pred3 = EnsembleModel.net4.predict_generator(unl_gen)

        unl_pred = (unl_pred1+unl_pred2+unl_pred3)/3
        arg_unl = np.argmax(unl_pred,axis=1)

        for i in range(len(arg_unl)):
            if unl_pred[i][arg_unl[i]] < 0.9:
                arg_unl[i] = -1

        df_u = pd.DataFrame({'filename': unl_files, 'unl_pred': arg_unl})

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
        gen = datagen.flow_from_dataframe(metadata, directory=
        f'{test_dir}/test_data', x_col='filename',
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
        self._rearrange(dataset)
        #self._rearrange_under(dataset) #여기서 rearrange로 알아서 정렬하는 것 같음

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
        unl = self.base_dir / 'unlabeled'
        unl.mkdir()
        unl2 = unl / 'unlabeled'
        unl2.mkdir()
        
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        for _, row in metadata.iterrows():
            src = src_dir / 'train_data' / row['filename']
            if row['annotation'] == UNLABELED:
                
                dst = unl2 / row['filename']
                shutil.copy(src=src, dst=dst)
                continue
            
            if not src.exists():
                raise FileNotFoundError
            
            dst = output_dir / self.classes[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)
