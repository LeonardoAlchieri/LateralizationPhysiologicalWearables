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
from numpy import empty, hstack, ndarray, random, unique, vstack, where
from pandas import DataFrame
from scipy.ndimage import gaussian_filter


def resampling(
    df: DataFrame = None,
    x: ndarray| None = None,
    y: ndarray | None = None,
    labels: list[str] | None = None,
    resampling_method: BaseUnderSampler | None = None,
    random_state: int = 42,
) -> DataFrame:
    if resampling_method is None:
        warn(
            f"No resampling method provided, using {RandomUnderSampler.__name__} instead.",
            RuntimeWarning,
        )
        resampling_method = RandomUnderSampler

    if df is not None:
        x = df.drop(columns=["label"], inplace=False).values
        if len(df["label"].unique()) == 1:
            warn(
                f"Only one class in the dataset. Removing current user {df.name}",
                RuntimeWarning,
            )
            print(
                f"Only one class in the dataset. Removing current user {df.name}",
                RuntimeWarning,
            )
            return None
        y = df["label"].values
    else:
        if x is None or y is None or labels is None:
            raise ValueError(
                "Either a DataFrame or x, y and labels must be provided."
            )
        else:
            if len(set(labels)) == 1:
                warn(
                    f"Only one class in the dataset. Removing current user {df.name}",
                    RuntimeWarning,
                )
                print(
                    f"Only one class in the dataset. Removing current user {df.name}",
                    RuntimeWarning,
                )
                return None
        
    cc = resampling_method(random_state=random_state)
    x_resampled, y_resampled = cc.fit_resample(x, y)
    result = DataFrame(x_resampled)
    result["label"] = y_resampled
    return result


def local_resampling(
    x: ndarray,
    y: ndarray,
    groups: ndarray,
    resampling_method: BaseUnderSampler = RandomUnderSampler,
    seed: int = 42,
):  
    x = x.reshape((x.shape[0], -1))
    data = DataFrame(x, index=groups)
    data["label"] = y
    data_resampled = data.groupby(axis=0, level=0).apply(
        resampling, resampling_method=resampling_method, random_state=seed
    )
    x_resampled = data_resampled.drop(columns=["label"], inplace=False).values
    y_resampled = data_resampled["label"].values
    groups = data_resampled.index.get_level_values(0).values
    return x_resampled, y_resampled, groups


def data_augmentation(
    x: ndarray,
    y: ndarray,
    methods: list[str] = ["jitter", "scaling", "flipping", "warping"],
    aug_percent: float = 0.2,
    seed: int | None = None,
) -> tuple[ndarray, ndarray]:
    """
    Augment the provided data array by applying a series of augmentation techniques.

    Parameters
    ----------
    x : ndarray
        Input data of shape (n_samples, n_features).
    y : ndarray
        Labels corresponding to the input data.
    methods : List[str], optional
        List of augmentation techniques to be applied. The options are 'jitter', 'scaling', 'flipping' and 'warping'.
        By default, all these methods are applied.
    aug_percent : float, optional
        Percentage of data to be augmented, by default 0.2.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    (ndarray, ndarray)
        Tuple of the augmented data and corresponding labels.

    """
    if seed:
        random.seed(seed)
    n_samples = x.shape[0]
    n_augment = int(n_samples * aug_percent)
    classes, counts = unique(y, return_counts=True)

    x_augmented = empty((0, x.shape[1]))
    y_augmented = empty((0,))

    for cls in classes:
        idx = where(y == cls)[0]
        for method in methods:
            random.shuffle(idx)
            x_temp = x[idx[:n_augment]]
            y_temp = y[idx[:n_augment]]

            if method == "jitter":
                noise = random.normal(loc=0, scale=0.05, size=x_temp.shape)
                x_temp = x_temp + noise

            elif method == "scaling":
                scale = random.uniform(low=0.5, high=2.0)
                x_temp = x_temp * scale

            elif method == "flipping":
                x_temp = -x_temp

            elif method == "warping":
                for i in range(x_temp.shape[0]):
                    x_temp[i] = gaussian_filter(x_temp[i], sigma=1)

            x_augmented = vstack((x_augmented, x_temp))
            y_augmented = hstack((y_augmented, y_temp))

    return x_augmented, y_augmented
