from json import dump as json_dump
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any

from numpy import ndarray, savez

path.append(".")
from src.utils.experiment_info import ExperimentInfo
from src.utils.io import filter_sleep_nights, load_config, load_processed_data
from src.utils.segmentation import segment
from src.utils import filter_user

basicConfig(filename="logs/run_segmentation.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/feature_extraction/config_segmentation.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_experiment_info: str = configs["path_to_experiment_info"]
    eda_data_base_path: str = configs["eda_data_base_path"]
    eda_data_format: str = configs["eda_data_format"]
    path_to_save_file: str = configs["path_to_save_file"]
    mode: int = configs["mode"]
    segment_size_in_secs: int = configs["segment_size_in_secs"]
    eda_sample_rate: int = configs["eda_sample_rate"]
    artifacts: bool = configs["artifacts"]
    components: list[str] = configs["components"]
    users_to_remove: list[str] = configs["users_to_remove"]
    bileteral_fusion: bool = configs["bileteral_fusion"]

    experiment_info = ExperimentInfo(path=path_to_experiment_info, mode=mode)
    experiment_info.filter_correct_times(inplace=True)
    eda_data = load_processed_data(path=eda_data_base_path, file_format=eda_data_format)
    if mode == 2:
        eda_data = filter_sleep_nights(
            data=eda_data, experiment_info=experiment_info.to_df()
        )

    users_in_common: set | None = None
    for side in eda_data.keys():
        eda_data[side] = filter_user(
        users_to_filter=users_to_remove, data=eda_data[side]
    )
        if users_in_common is None:
            users_in_common = set(eda_data[side].keys())
        else:
            users_in_common = set(eda_data[side].keys()).intersection(users_in_common)
            
    logger.debug(
        f"Number of users with both left and right hand data: {len(users_in_common)}"
    )

    segment_size_in_sampling_rate: int = segment_size_in_secs * eda_sample_rate

    if not artifacts:
        if not bileteral_fusion:
            (
                values_left,
                values_right,
                labels_left,
                labels_right,
                groups_left,
                groups_right,
            ) = segment(
                data=eda_data,
                experiment_info_as_dict=experiment_info.to_dict(),
                segment_size_in_sampling_rate=segment_size_in_sampling_rate,
                segment_size_in_secs=segment_size_in_secs,
                data_sample_rate=eda_sample_rate,
                mode=experiment_info.get_mode(),
                components=components,
            )
            logger.info("Saving data")
            savez(
                file=path_to_save_file,
                values_left=values_left,
                values_right=values_right,
                labels_left=labels_left,
                labels_right=labels_right,
                groups_left=groups_left,
                groups_right=groups_right,
            )
            logger.info("Data saved successfully")
        else:
            (
                values_left,
                values_right,
                values_diff,
                labels_left,
                labels_right,
                labels_diff,
                groups_left,
                groups_right,
                groups_diff,
            ) = segment(
                data=eda_data,
                experiment_info_as_dict=experiment_info.to_dict(),
                segment_size_in_sampling_rate=segment_size_in_sampling_rate,
                segment_size_in_secs=segment_size_in_secs,
                data_sample_rate=eda_sample_rate,
                mode=experiment_info.get_mode(),
                components=components,
                bilateral=True,
            )
            logger.info("Saving data")
            savez(
                file=path_to_save_file,
                values_left=values_left,
                values_right=values_right,
                values_diff=values_diff,
                labels_left=labels_left,
                labels_right=labels_right,
                labels_diff=labels_diff,
                groups_left=groups_left,
                groups_right=groups_right,
                groups_diff=groups_diff,
            )
            logger.info("Data saved successfully")
    else:
        values_left: list[ndarray]
        values_right: list[ndarray]
        labels_left: list[int]
        labels_right: list[int]
        groups_left: list[str]
        groups_right: list[str]
        artefacts_left: list[int | bool]
        artefacts_right: list[int | bool]
        (
            values_left,
            values_right,
            labels_left,
            labels_right,
            groups_left,
            groups_right,
            artefacts_left,
            artefacts_right,
        ) = segment(
            data=eda_data,
            experiment_info_as_dict=experiment_info.to_dict(),
            segment_size_in_sampling_rate=segment_size_in_sampling_rate,
            segment_size_in_secs=segment_size_in_secs,
            data_sample_rate=eda_sample_rate,
            mode=experiment_info.get_mode(),
            artefact=True,
            components=components,
        )

        logger.info("Saving data")
        savez(
            file=path_to_save_file,
            values_left=values_left,
            values_right=values_right,
            labels_left=labels_left,
            labels_right=labels_right,
            groups_left=groups_left,
            groups_right=groups_right,
            artefacts_left=artefacts_left,
            artefacts_right=artefacts_right,
        )
        logger.info("Data saved successfully")


if __name__ == "__main__":
    logger.info("**START OF PROGRAM**")
    main()
    logger.info("**END OF PROGRAM**")
