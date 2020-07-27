from typing import Callable, List

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.layers import Dense
import nsml
import numpy as np
import pandas as pd
from spam.spam_classifier.models.others import scheduler

from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier # vote로 할수도 있고, 값 평균값을 내서 하는 방식도 가능할 것 같다.

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate


# def build_model():
#     model = keras.Sequential([
#         Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
#         Dense(64, activation='relu'),
#         Dense(1)
#     ])

#     optimizer = tf.keras.optimizers.Adam()

#     model.compile(loss='mse',
#                     optimizer=optimizer,
#                     metrics=['mae', 'mse'])
#     return model