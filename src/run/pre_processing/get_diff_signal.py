from sys import path
from typing import Any, Callable
from logging import DEBUG, INFO, basicConfig, getLogger

path.append(".")
from src.utils.io import load_config, load_processed_data
from src.utils.pre_processing import trim_to_shortest

basicConfig(filename="logs/get_diff_signal.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/statistical_analysis/config_stationarity_tests.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    logger.debug("Configs loaded")

    # n_workers: int = configs["n_workers"]
    dataset: str = configs["dataset"]
    path_to_data: str = configs["path_to_data"]
    data_format: str = configs["data_format"]
    signal_component: str = configs["signal_component"]
    methods: list[str] | None = configs["methods"]
    path_to_save_data: str = configs["path_to_save_data"]
    
    physiological_data = load_processed_data(path=path_to_data, file_format=data_format)
    physiological_data = trim_to_shortest(signal_data=physiological_data)
    
    


if __name__ == "__main__":
    main()