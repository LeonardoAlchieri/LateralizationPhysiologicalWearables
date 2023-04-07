from warnings import warn

from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from imblearn.under_sampling.base import BaseUnderSampler
from pandas import DataFrame


def resampling(
    df: DataFrame,
    resampling_method: BaseUnderSampler | None = None,
    random_state: int = 42,
) -> DataFrame:
    if resampling_method is None:
        warn(
            f"No resampling method provided, using {RandomUnderSampler.__name__} instead.",
            RuntimeWarning,
        )
        resampling_method = RandomUnderSampler

    x = df.drop(columns=["label"], inplace=False).values
    y = df["label"].values
    cc = resampling_method(random_state=random_state)
    x_resampled, y_resampled = cc.fit_resample(x, y)
    result = DataFrame(x_resampled)
    result["label"] = y_resampled
    return result


def local_resampling(x, y, groups, resampling_method: BaseUnderSampler = RandomUnderSampler):
    data = DataFrame(x, index=groups)
    data["label"] = y
    data_resampled = data.groupby(axis=0, level=0).apply(resampling, resampling_method=resampling_method)
    x_resampled = data_resampled.drop(columns=["label"], inplace=False).values
    y_resampled = data_resampled["label"].values
    groups = data_resampled.index.get_level_values(0).values
    return x_resampled, y_resampled, groups