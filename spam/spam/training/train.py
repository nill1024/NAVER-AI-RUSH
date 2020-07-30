from importlib import import_module

import nsml

from spam.spam_classifier.models.EnsembleModel import bind_model


def train(experiment_name: str = 'v_res', pause: bool = False, mode: str = 'train'):
    config = import_module(f'spam.training.experiments.{experiment_name}').config
    model = config['model'](**config['model_kwargs']) #BasicModel()

    # print(type(model))
    # print(type(config))

    bind_model(model)
    if pause:
        nsml.paused(scope=locals())
    if mode == 'train':
        model.fit(**config['fit_kwargs'])
