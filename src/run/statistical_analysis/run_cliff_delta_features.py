from typing import Any
from sys import path
from logging import DEBUG, INFO, basicConfig, getLogger

path.append(".")
from src.utils.io import load_and_prepare_data, load_config, load_processed_data
from src.utils.experiment_info import ExperimentInfo

basicConfig(filename="logs/run_cliff_delta_features.log", level=DEBUG)

logger = getLogger("main")

def segment_extract_featurea():
    (
    values_left,
    values_right,
    labels_left,
    labels_right,
    groups_left,
    groups_right,
) = segment(
    data=eda_data,
    experiment_info_as_dict=experiment_info_as_dict,
    segment_size_in_sampling_rate=segment_size_in_sampling_rate,
    segment_size_in_secs=segment_size_in_secs,
    data_sample_rate=eda_sample_rate,
)



def main():
    path_to_config: str = (
        "src/run/statistical_analysis/config_cliff_delta_feautures.yml"
    )

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    path_to_experiment_info: str = configs["path_to_experiment_info"]
    eda_data_base_path: str = configs["eda_data_base_path"]
    eda_data_format: str = configs["eda_data_format"]
    mode: int = configs["mode"]
    segment_size_in_secs = configs['segment_size_in_secs']
    eda_sample_rate = configs['eda_sample_rate']

    experiment_info = ExperimentInfo(path=path_to_experiment_info, mode=mode)
    eda_data = load_processed_data(path=eda_data_base_path, file_format=eda_data_format)

    users_in_left_side = set(eda_data["left"].keys())
    users_in_right_side = set(eda_data["right"].keys())
    logger.debug(
        f"Number of users with both left and right hand data: {len(users_in_left_side & users_in_right_side)}"
    )
    
    segment_size_in_sampling_rate: int = segment_size_in_secs * eda_sample_rate
    experiment_info_as_dict = experiment_info.to_dict()


if __name__ == "__main__":
    main()
