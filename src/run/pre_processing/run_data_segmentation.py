from os.path import basename
from logging import basicConfig, getLogger, INFO
from numpy.random import seed as set_np_seed, choice as random_choice
from random import seed as set_seed
from os.path import basename, join as join_paths
from logging import DEBUG, INFO, WARNING, basicConfig, getLogger
from pandas import (
    DataFrame,
    read_parquet,
    read_excel,
    IndexSlice,
    Timestamp,
    Timedelta,
    Series,
    MultiIndex,
)
from numpy import datetime64, array, ndarray, concatenate
from numpy import isnan
from numpy import sqrt
from scipy.stats import ttest_ind_from_stats
from tqdm import tqdm

from src.utils.io import load_config
from pandas import read_csv, DataFrame



_filename: str = basename(__file__).split(".")[0][4:]
basicConfig(filename=f"logs/run/statistical_analysis/{_filename}.log", level=INFO)
logger = getLogger(_filename)



def perform_data_segmentation(
    all_data: DataFrame,
    experimento_info_w_laugh: DataFrame,
    event_of_interest: str = "laughter",
    baseline_event: str = "baseline_2",
) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    if "laugh" in event_of_interest:
        logger.debug(f"Segmenting data for laughter -> changin to include all sessions")
        event_of_interest = "f0t"

    # TODO: make more efficient
    left_data_list = list()
    right_data_list = list()
    labels_list = list()
    groups_list = list()
    for user in experimento_info_w_laugh.index.get_level_values(0).unique():
        repeat_time: int = 0
        for event in (
            experimento_info_w_laugh.loc[IndexSlice[user, :], :]
            .index.get_level_values(1)
            .unique()
        ):
            if event[:3] == event_of_interest:
                start: Timestamp | datetime64 = experimento_info_w_laugh.loc[
                    IndexSlice[user, event], "start"
                ].unique()[0]
                end = start + Timedelta(seconds=2)
                left_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "left",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                right_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "right",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                left_data_list.append(left_data.values)
                right_data_list.append(right_data.values)
                labels_list.append(1)
                groups_list.append(user)

                start: Timestamp | datetime64 = experimento_info_w_laugh.loc[
                    IndexSlice[user, baseline_event], "start"
                ].unique()[0]
                start = start + Timedelta(seconds=2 * repeat_time)
                end = start + Timedelta(seconds=2)
                left_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "left",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                right_data: DataFrame = all_data.loc[
                    IndexSlice[user, start:end],
                    IndexSlice[
                        "right",
                        ["ACC_filt", "BVP_filt", "EDA_filt_phasic", "EDA_filt_stand"],
                    ],
                ]
                left_data_list.append(left_data.values)
                right_data_list.append(right_data.values)
                labels_list.append(0)
                groups_list.append(user)

                repeat_time += 1
            else:
                continue

    return (
        array(left_data_list),
        array(right_data_list),
        array(labels_list),
        array(groups_list),
    )

def main(seed: int):
    set_np_seed(seed)

    path_to_config: str = f"src/run/machine_learning_task/config_{_filename}.yml"

    logger.info("Starting model training")
    configs = load_config(path=path_to_config)
    logger.debug("Configs loaded")
    
    data_type: str = configs["data_type"]
    path_to_data: str = configs["path_to_data"]
    experiment_info_path: str = configs["experiment_info_path"]
    experiment_info: DataFrame = read_csv(experiment_info_path, index_col=[0,1])
    
    
    (
        left_data_list,
        right_data_list,
        labels_list,
        groups_list,
    ) = perform_data_segmentation(
        all_data=all_data,
        experimento_info_w_laugh=experimento_info_w_laugh,
        event_of_interest=event_of_interest,
        baseline_event=baseline_event,
    )

    (
        hand_crafted_features_left,
        hand_crafted_features_right,
        hand_crafted_features_random,
    ) = perform_feature_extraction_over_segments(
        left_data_list=left_data_list, right_data_list=right_data_list
    )    
