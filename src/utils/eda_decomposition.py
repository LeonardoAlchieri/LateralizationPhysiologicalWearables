from collections import defaultdict
from numpy import ndarray
from collections import defaultdict
from logging import getLogger
from sys import path
from joblib import Parallel, delayed

# from joblib import Parallel, delayed
from time import time
from numpy import ndarray, stack
from pandas import DataFrame, IndexSlice, RangeIndex, Series, concat, read_csv
from tqdm.auto import tqdm


path.append(".")

from src.utils import (
    blockPrinting,
    make_timestamp_idx,
    parallel_iteration,
    remove_empty_sessions,
    segment_over_experiment_time,
    correct_session_name,
)
from src.utils.eda import decomposition
from src.utils.experiment_info import ExperimentInfo
from src.utils.filters import butter_lowpass_filter_filtfilt
from src.utils.io import (
    filter_sleep_nights,
    load_and_prepare_data,
    load_config,
    load_processed_data,
)
from src.utils.plots import make_lineplot
from src.utils.pre_processing import (
    concate_session_data,
    get_rescaling_technique,
    rescaling,
)

logger = getLogger("eda_decomposition")


def single_eda_singal_decomposition(
    session_data: DataFrame | Series, sampling_frequecy: int = 4, **kwargs
) -> tuple[DataFrame, DataFrame]:
    data_to_decompose = session_data.values

    decomposition_result = decomposition(
        data_to_decompose,
        sampling_frequecy,
        name=f"{kwargs.get('user', None)}_{kwargs.get('session', None)}_{kwargs.get('side', None)}",
    )
    tonic_component = decomposition_result["tonic component"]
    phasic_component = decomposition_result["phasic component"]

    if isinstance(session_data, Series) or (
        isinstance(session_data, DataFrame) and len(session_data.columns) == 1
    ):
        tonic_component = DataFrame(
            tonic_component,
            index=session_data.index,
            columns=(
                session_data.columns if isinstance(session_data, DataFrame) else None
            ),
        )

        phasic_component = DataFrame(
            phasic_component,
            index=session_data.index,
            columns=(
                session_data.columns if isinstance(session_data, DataFrame) else None
            ),
        )
    else:
        phasic_component = stack(
            [
                phasic_component,
                session_data[:, 1]
                if isinstance(session_data, ndarray)
                else session_data.iloc[:, 1],
            ],
            axis=1,
        )
        tonic_component = stack(
            [
                tonic_component,
                session_data[:, 1]
                if isinstance(session_data, ndarray)
                else session_data.iloc[:, 1],
            ],
            axis=1,
        )

    return ((kwargs['side'], kwargs['user'], kwargs['session']), (tonic_component, phasic_component))


def apply_cvxeda_decomposition(
    eda_data: dict[str, dict[str, dict[str, ndarray]]], sampling_frequecy: int = 4, n_jobs: int = -1
) -> tuple[
    dict[str, dict[str, dict[str, ndarray]]], dict[str, dict[str, dict[str, ndarray]]]
]:
    start = time()
    # TODO: better handling of sampling frequency. If it is not defined, now
    # it will take 4Hz.
    decomposition_results: list[
        tuple[tuple[str, str, str], tuple[DataFrame, DataFrame]]
    ] = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(single_eda_singal_decomposition)(
                session_data=session_data,
                sampling_frequecy=sampling_frequecy,
                side=side,
                user=user,
                session=session,
            )
        for side in eda_data.keys()
        for user, user_data in eda_data[side].items()
        for session, session_data in user_data.items()
    )

    eda_data_tonic = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    )
    eda_data_phasic = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    )
    for result in decomposition_results:
        side = result[0][0]
        user = result[0][1]
        session = result[0][2]

        tonic_component = result[1][0]
        phasic_component = result[1][1]
        eda_data_tonic[side][user][session] = tonic_component
        eda_data_phasic[side][user][session] = phasic_component

    logger.debug("Total phasic component calculation: %.2f s" % (time() - start))

    return eda_data_tonic, eda_data_phasic
