from typing import Union
import pathlib
import pickle


def load_pickle(path: Union[str, pathlib.Path]):
    """Save an object as a .pkl file.
    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to load the object.
    Returns
    -------
    None
    """
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_pickle(path: Union[str, pathlib.Path], obj) -> None:
    """Save an object as a .pkl file.
    Parameters
    ----------
    path : Union[str, pathlib.Path]
        Path to save the object.
    obj : Object
        Object to save.
    Returns
    -------
    None
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
