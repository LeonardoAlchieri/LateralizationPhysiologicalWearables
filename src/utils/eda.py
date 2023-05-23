from numpy import ndarray, stack
from pandas import Series, DataFrame
from cvxEDA import cvxEDA

from src.utils import blockPrinting
from src.utils.pre_processing import standardize

# See https://github.com/lciti/cvxEDA for more EDA analysis methdos


@blockPrinting
def decomposition(
    eda_signal: Series | ndarray | list, frequency: int = 4
) -> dict[str, ndarray]:
    """This method will apply the cvxEDA decomposition to an EDA signal. The cvxEDA
    implementation is the one from Greco et al.

    Parameters
    ----------
    eda_signal : Series | ndarray | list
        eda signal to be decomposed
    Fs : int, optional
        frequency of the input signal, e.g. 64Hz

    Returns
    -------
    dict[str, ndarray]
        the method returns a dictionary with the decomposed signals
        (see cvxEDA for more details)
    """
    if isinstance(eda_signal, Series) or (isinstance(eda_signal, ndarray) and eda_signal.ndim == 1):
        # TODO: see if the standardization is actually something we want!
        yn = standardize(signal=eda_signal)
    elif isinstance(eda_signal, DataFrame):
        yn = standardize(signal=eda_signal.iloc[:, 0])
    elif isinstance(eda_signal, ndarray) and eda_signal.ndim == 2:
        # TODO: see if the standardization is actually something we want!
        yn = standardize(signal=eda_signal[:,0])
    else:
        raise TypeError(f'eda_signal must be a Series, DataFrame or ndarray, not {type(eda_signal)}')
    return cvxEDA(yn, 1.0 / frequency)
