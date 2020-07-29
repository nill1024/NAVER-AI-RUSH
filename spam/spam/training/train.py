from importlib import import_module

import nsml

import spam.spam_classifier.models.BasicModel as bm
import spam.spam_classifier.models.EnsembleModel as em


def train(experiment_name: str = 'v_res', pause: bool = False, mode: str = 'train'):
    config = import_module(f'spam.training.experiments.{experiment_name}').config
    model = config['model'](**config['model_kwargs']) #BasicModel()

    # print(type(model))
    # print(type(config))
    if experiment_name == 'v_ensemble':
        em.bind_model(model)
    else:
        bm.bind_model(model)

    if pause:
        nsml.paused(scope=locals())
    if mode == 'train':
        model.fit(**config['fit_kwargs'])
