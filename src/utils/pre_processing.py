# from joblib import Parallel, delayed
from collections import defaultdict
from logging import getLogger
from typing import Callable

from numpy import array, ndarray
from pandas import Series, concat
from tqdm import tqdm

from src.utils import prepare_data_for_concatenation

logger = getLogger("pre_processing")


def concate_session_data(
    data_dict: defaultdict[str, dict[str, dict[str, Series]]], progress: bool = False) -> dict[str, dict[str, Series]]:
    """Concatenate data from different sessions for each user.

    Args:
        data_dict (defaultdict[str, dict[str, dict[str, Series]]]): [description]

    Returns:
        dict[str, dict[str, Series]]: [description]
    """
    data_dict: defaultdict[str, dict[str, Series]] = {
        side: {
            user: concat(
                [
                    prepare_data_for_concatenation(
                        data=data_dict[side][user][session], session_name=session
                    )
                    for session in data_dict[side][user].keys()
                ],
                axis=0,
                join="outer",
            ).sort_index()
            for user in tqdm(data_dict[side].keys(), desc=f'Concatenating user data for side {side}', colour='green')
        }
        for side in data_dict.keys()
    }
    return data_dict


def rescaling(
    data: defaultdict[str, dict[str, dict[str, Series]]], rescaling_method: Callable
) -> defaultdict[str, dict[str, dict[str, Series]]]:
    """Rescale data using the specified method.

    Args:
        data (defaultdict[str, dict[str, dict[str, Series]]]): data to rescale
        rescaling_method (Callable): method to use for rescaling

    Returns:
        defaultdict[str, dict[str, dict[str, Series]]]: rescaled data
    """
    data: defaultdict[str, dict[str, Series]] = {
        side: {
            user: {
                session: Series(
                    rescaling_method(data[side][user][session]),
                    index=data[side][user][session].index,
                )
                for session in data[side][user].keys()
            }
            for user in data[side].keys()
        }
        for side in data.keys()
    }
    return data


# TODO: probably remove and use some third party library
def standardize(eda_signal: Series | ndarray | list) -> ndarray:
    """Simple method to standardize an EDA signal.

    Parameters
    ----------
    eda_signal : Series | ndarray | list
        eda signal to standardize

    Returns
    -------
    ndarray
        returns an array standardized
    """
    y: ndarray = array((eda_signal))
    yn: ndarray = (y - y.mean()) / y.std()
    return yn
