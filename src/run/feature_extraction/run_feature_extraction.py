from json import load as json_load
from logging import DEBUG, INFO, basicConfig, getLogger
from sys import path
from typing import Any

from numpy import ndarray, stack, asarray, savez, load
from joblib import Parallel, delayed

path.append(".")
from src.utils.experiment_info import ExperimentInfo
from src.utils.io import load_config
from src.feature_extraction.eda import get_eda_features

basicConfig(filename="logs/run_feature_extraction.log", level=DEBUG)

logger = getLogger("main")


def main():
    path_to_config: str = "src/run/feature_extraction/config_feature_extraction.yml"

    logger.info("Starting model training")
    configs: dict[str, Any] = load_config(path=path_to_config)
    n_jobs: int = configs["n_jobs"]
    logger.debug("Configs loaded")
    
    artifacts: bool = configs["artifacts"]
    path_to_segmented_data: str = configs["path_to_segmented_data"]
    path_to_save_file: str = configs["path_to_save_file"]
    
    segmented_data: dict[str, Any] = load(path_to_segmented_data)
        
    values_left: ndarray[float] = segmented_data["values_left"]
    values_right: ndarray[float] = segmented_data["values_right"]
    labels_left: ndarray[int] = segmented_data["labels_left"]
    labels_right: ndarray[int] = segmented_data["labels_right"]
    groups_left: ndarray[str] = segmented_data["groups_left"]
    groups_right: ndarray[str] = segmented_data["groups_right"]
    
    logger.info('Starting feature extraction')
    features_left = Parallel(n_jobs=n_jobs)(
    delayed(get_eda_features)(value) for value in (values_left)
)
    logger.info('Extracted for left side data')
    features_right = Parallel(n_jobs=n_jobs)(
        delayed(get_eda_features)(value) for value in (values_right)
    )
    logger.info('Extracted for right side data')
    logger.info('Complete feature extraction. Reorganizing the data before saving.')

    features_left: ndarray = stack(features_left)
    features_right: ndarray = stack(features_right)

    labels_left: ndarray = stack(labels_left)
    labels_right: ndarray = stack(labels_right)

    groups_left: ndarray = stack(groups_left)
    groups_right: ndarray = stack(groups_right)
    
    if artifacts:
        logger.info("Loading artifacts data as well")
        artefacts_left = segmented_data["artefacts_left"]
        artefacts_right = segmented_data["artefacts_right"]
        
        artefacts_left: ndarray = stack(artefacts_left)
        artefacts_right: ndarray = stack(artefacts_right)

        logger.info("Saving the data")
        savez(
            file=path_to_save_file,
            features_left=features_left,
            features_right=features_right,
            labels_left=labels_left,
            labels_right=labels_right,
            groups_left=groups_left,
            groups_right=groups_right,
            artefacts_left=artefacts_left,
            artefacts_right=artefacts_right)
    else:
        logger.info("Saving the data")
        savez(
            file=path_to_save_file,
            features_left=features_left,
            features_right=features_right,
            labels_left=labels_left,
            labels_right=labels_right,
            groups_left=groups_left,
            groups_right=groups_right)
    logger.info("Data saved successfully")
        
        


if __name__ == "__main__":
    logger.info("**START OF PROGRAM**")
    main()
    logger.info("**END OF PROGRAM**")