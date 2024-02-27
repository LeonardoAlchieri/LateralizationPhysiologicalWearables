from copy import deepcopy
from warnings import warn
from collections import defaultdict

from numpy import nan
from pandas import DataFrame, MultiIndex, concat, to_datetime, Series, IndexSlice

from src.utils import (
    INTENSITIES_MAPPING,
    SESSIONS_GROUPINGS,
    slice_user_over_experiment_time,
)
from src.utils.io import read_experiment_info


def add_events_to_signal_data(
    signal_data: DataFrame,
    experiment_info: DataFrame,
    experimento_info_w_laugh: DataFrame,
    sessions_groupings_w_laugh: list[str],
) -> DataFrame:
    """This method adds the events to the signal dataframe, using a multi-index
    structure

    Parameters
    ----------
    signal_data : DataFrame
        raw data with the signal values
    experiment_info : DataFrame
        dataframe with information regarding the experiment
    experimento_info_w_laugh : DataFrame
        dataframe with the experiment info and the laughter info
    sessions_groupings_w_laugh : list[str]
        list of ways to group the different events

    Returns
    -------
    DataFrame
        dataframe with a 3-level multi index structure, w/ event, user, timeframe
    """
    signal_data = signal_data.groupby(level=0, axis=0, group_keys=False).apply(
        slice_user_over_experiment_time,
        experimento_info=experiment_info,
        slicing_col="experiment",
    )
    signal_data.columns = signal_data.columns.droplevel(1)
    # TODO: add parallel computation here
    different_groupings_signal_data_w_laugh: dict[str, DataFrame] = concat(
        [
            signal_data.groupby(level=0, axis=0, group_keys=False).apply(
                slice_user_over_experiment_time,
                experimento_info=experimento_info_w_laugh,
                slicing_col=session,
            )
            for session_group in sessions_groupings_w_laugh.values()
            for session in session_group
        ],
        keys=[
            f"{group_name}%{event}"
            for group_name, group_item in sessions_groupings_w_laugh.items()
            for event in group_item
        ],
        names=["grouping"],
    )
    different_groupings_signal_data_w_laugh.index = MultiIndex.from_tuples(
        [
            (el[0].split("%")[0], el[0].split("%")[1], el[1], el[2])
            for el in different_groupings_signal_data_w_laugh.index
        ],
        names=["group", "event", "user", "timestamp"],
    )

    return different_groupings_signal_data_w_laugh


def add_laughter_to_experiment_info(
    laughter_info_data: DataFrame, experiment_info: DataFrame
) -> tuple[DataFrame, list[str]]:
    """Simple method to add laughter episodess to the experiment info dataframe.
    Will also perform some simple cleaning.

    Parameters
    ----------
    laughter_info_data : DataFrame
        dataframe with laughter data
    experiment_info : DataFrame
        dataframe with experiment info, i.e., informations regarding the
        events performed

    Returns
    -------
    DataFrame
        returns a dataframe with all of the experiment info and the laughter info,
        with a multiindex structure; and a list of the experiment events
        grouping with the laughter as well
    """
    laughter_info_data.index = MultiIndex.from_tuples(
        [tuple(idx.split("_")) for idx in laughter_info_data.index]
    )
    laughter_info_data = laughter_info_data.sort_index()
    laughter_info_data["intensity"] = (
        laughter_info_data["intensity"]
        .apply(lambda x: INTENSITIES_MAPPING.get(x, nan))
        .astype(float)
    )

    laughter_info_data = laughter_info_data[laughter_info_data["intensity"] > 0]
    laughter_info_data[["start", "end"]] = laughter_info_data[["start", "end"]].apply(
        to_datetime
    )

    sessions_groupings_w_laugh = deepcopy(SESSIONS_GROUPINGS)
    sessions_groupings_w_laugh["laughter_episodes"] = list(
        laughter_info_data.index.get_level_values(1).unique()
    )

    # Join the laughter info with the experimento info
    experimento_info_w_laugh = concat(
        [laughter_info_data.iloc[:, :3], experiment_info], axis=0
    ).sort_index()

    return experimento_info_w_laugh, sessions_groupings_w_laugh


class ExperimentInfo:
    def __init__(self, path: str, mode: int = 1):
        self.data = read_experiment_info(path, mode)
        self.mode = mode
        self.path = path

    def to_dict(self) -> dict[str, Series]:
        if self.mode == 2:
            return {
                participant: self.data.loc[IndexSlice[participant, :], :]
                for participant in list(self.data.index.get_level_values(0).unique())
            }
        elif self.mode == 1:
            return {
                participant: self.data.loc[participant, :]
                for participant in list(self.data.index.unique())
            }
        else:
            raise ValueError(f"Mode must be 1 or 2. Got {self.mode}")

    def to_df(self) -> DataFrame:
        return self.data

    def filter_correct_times(self, inplace: bool = False) -> DataFrame | None:
        if self.mode == 2:
            self.data[self.data["actual_bed_time"] < self.data["wake_up_time"]]
        elif self.mode == 1:
            warn("This method is not implemented for mode 1", RuntimeWarning)
        else:
            raise ValueError(f"Mode must be 1 or 2. Got {self.mode}")

        if not inplace:
            return self.data

    def get_mode(self) -> int:
        return self.mode


def segmentation_over_label(
    session_data: DataFrame, user_experiment_info: Series,  mode: int, session_name: str | None = None,
) -> DataFrame:
    if mode == 1:
        event_keywords: list[str] = list(
            set(
                [
                    event_name.replace("start_", "")
                    if "start" in event_name
                    else event_name.replace("end_", "")
                    for event_name in user_experiment_info.index
                ]
            )
        )
        return {
            event: session_data.loc[
                user_experiment_info["start_" + event] : user_experiment_info[
                    "end_" + event
                ]
            ]
            for event in event_keywords
        }
    elif mode == 2:
        if session_name is None:
            raise ValueError(f'For mode 2, session_name must be provided. Got {session_name}')
        
        session_info = user_experiment_info.loc[IndexSlice[:, session_name], :]
        result_dict: dict[str, DataFrame] = {}
        result_dict.update(
            {"sleep": session_data.loc[session_info['actual_bed_time'].iloc[0] : session_info['wake_up_time'].iloc[0]]}
        )
        result_dict.update(
            {"awake_1": session_data.loc[:session_info['actual_bed_time'].iloc[0]]}
        )
        result_dict.update(
            {"awake_2": session_data.loc[session_info['wake_up_time'].iloc[0]:]}
        )
        return result_dict
    else:
        raise ValueError(f"Mode must be 1 or 2. Got {mode}")

    ...


def separate_raw_signal_by_label(
    data: dict[str, defaultdict[str, defaultdict[str, DataFrame]]],
    experiment_info: ExperimentInfo,
):

    data_with_labels: dict[str, dict[str, dict[str, DataFrame]]] = {
        side: {
            user: {session: segmentation_over_label(session_data=session_data, 
                                                    user_experiment_info=experiment_info.to_dict()[user], 
                                                    mode=experiment_info.get_mode(),
                                                    session_name=session)
                   for session, session_data in user_data.items()}
            for user, user_data in side_data.items()
        }
        for side, side_data in data.items()
    }

    return data_with_labels
    ...
