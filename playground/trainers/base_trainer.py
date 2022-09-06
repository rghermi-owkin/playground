from typing import Union
import pathlib

from omics_rpz.utils import load_pickle, save_pickle


class BaseTrainer:

    def __init__(self):
        pass

    def train(self, X_train, y_train, X_valid, y_valid):
        raise NotImplementedError

    def evaluate(self, X, y):
        raise NotImplementedError

    def predict(self, X, y):
        raise NotImplementedError

    def save(self, path: Union[pathlib.Path, str]):
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Union[pathlib.Path, str]):
        del cls
        return load_pickle(path)
