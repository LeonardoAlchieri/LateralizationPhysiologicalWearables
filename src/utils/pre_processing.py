# from joblib import Parallel, delayed
from collections import defaultdict
from logging import getLogger
from typing import Callable

from joblib import Parallel, delayed
from numpy import array, ndarray, stack, nanmean, nanstd
from pandas import Series, concat, DataFrame
from tqdm import tqdm

from src.utils import prepare_data_for_concatenation

logger = getLogger("pre_processing")


def concate_session_data(
    data_dict: defaultdict[str, dict[str, dict[str, Series]]], n_jobs: int = 1
) -> dict[str, dict[str, Series]]:
    """Concatenate data from different sessions for each user.

    Args:
        data_dict (defaultdict[str, dict[str, dict[str, Series]]]): [description]

    Returns:
        dict[str, dict[str, Series]]: [description]
    """
    if n_jobs == 1:
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
                for user in tqdm(
                    data_dict[side].keys(),
                    desc=f"Concatenating user data for side {side}",
                    colour="green",
                )
                if len(data_dict[side][user]) > 0
            }
            for side in data_dict.keys()
        }
    elif n_jobs > 1 or n_jobs == -1:
        data_dict: defaultdict[str, dict[str, Series]] = {
            side: {
                user: concat(
                    Parallel(n_jobs=n_jobs)(
                        delayed(prepare_data_for_concatenation)(
                            data=data_dict[side][user][session], session_name=session
                        )
                        for session in data_dict[side][user].keys()
                    ),
                    axis=0,
                    join="outer",
                ).sort_index()
                for user in tqdm(
                    data_dict[side].keys(),
                    desc=f"Concatenating user data for side {side}",
                    colour="green",
                )
                if len(data_dict[side][user]) > 0
            }
            for side in data_dict.keys()
        }
    else:
        raise ValueError(f"Invalid value for n_jobs: {n_jobs} (must be >= 1 or -1)")
    return data_dict


def rescaling(
    data: defaultdict[str, dict[str, dict[str, Series]]],
    rescaling_method: Callable,
    n_jobs: int = 1,
) -> defaultdict[str, dict[str, dict[str, Series]]]:
    """Rescale data using the specified method.

    Args:
        data (defaultdict[str, dict[str, dict[str, Series]]]): data to rescale
        rescaling_method (Callable): method to use for rescaling

    Returns:
        defaultdict[str, dict[str, dict[str, Series]]]: rescaled data
    """
    # TODO: do this using a decorator function
    if n_jobs == 1:
        data: defaultdict[str, dict[str, DataFrame]] = {
            side: {
                user: {
                    session: DataFrame(
                        rescaling_method(data[side][user][session])
                        if isinstance(data[side][user][session], Series)
                        else stack(
                            [
                                rescaling_method(data[side][user][session].iloc[:, 0]),
                                data[side][user][session].iloc[:, 1].values,
                            ],
                            axis=1,
                        ),
                        index=data[side][user][session].index,
                        columns=data[side][user][session].columns,
                    )
                    for session in data[side][user].keys()
                }
                for user in tqdm(
                    data[side].keys(),
                    desc=f"Rescaling data for side {side}",
                    colour="blue",
                )
            }
            for side in data.keys()
        }
    elif n_jobs > 1 or n_jobs == -1:

        def support_rescaling(session: str, session_data: DataFrame):
            return (
                session,
                DataFrame(
                    rescaling_method(session_data.iloc[:, 0].values),
                    index=session_data.index,
                    columns=session_data.columns,
                ),
            )

        data = {
            side: {
                user: Parallel(n_jobs=n_jobs)(
                    delayed(support_rescaling)(session, data[side][user][session])
                    for session in data[side][user].keys()
                )
                for user in tqdm(
                    data[side].keys(),
                    desc=f"Rescaling data for side {side}",
                    colour="blue",
                )
            }
            for side in data.keys()
        }
        data = {
            side: {
                user: {
                    session_name: session_data
                    for (session_name, session_data) in user_acct_data
                }
                for user, user_acct_data in data[side].items()
            }
            for side in data.keys()
        }
    else:
        raise ValueError(f"Invalid value for n_jobs: {n_jobs} (must be >= 1 or -1)")
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

    yn: ndarray = (y - nanmean(y)) / nanstd(y)
    return yn
