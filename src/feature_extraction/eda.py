from logging import getLogger
from typing import Any
from warnings import warn

from neurokit2.eda import eda_peaks
from numpy import (
    apply_along_axis,
    array,
    gradient,
    isnan,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    ndarray,
)
from scipy.stats import linregress

logger = getLogger(__name__)


def get_eda_features(data: ndarray, sampling_rate: int = 4) -> ndarray:
    """This method performs the feature extraction for an EDA signal (be it mixed or phasic).
    The features extracted are: statistical features (minimum, maximum, mean, standard deviation,
    difference between maximum and minimum value or dynamic change, slope, absolute value
    of the slope, mean and standard deviation of the first derivative), number of peaks,
    peaks’ amplitude.
    The features extracted follow what done by Di Lascio et al. (2019).

    Parameters
    ----------
    data : ndarray
        eda data to extract features from.
    sampling_rate : int, optional
        sampling rate of the eda features, in Hz, by default 4.

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description, i.e.,
        `[min, max, mean, std, diff_max_min, slope, absolute_slope, mean_derivative,
        std_derivative,number_peaks,peaks_amplitude]`
    """

    data: ndarray = data[~isnan(data).any(axis=1)]
    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        min_feat: float = nanmin(data, axis=0)
        max_feat: float = nanmax(data, axis=0)
        mean_feat: float = nanmean(data, axis=0)
        std_feat: float = nanstd(data, axis=0)
        dynamic_range_feat: float = max_feat - min_feat

        def get_slop_linregress(arr: ndarray) -> float:
            slope, intercept, r_value, p_value, std_err = linregress(
                range(len(arr)), arr
            )
            return slope

        slope_feat = apply_along_axis(get_slop_linregress, axis=0, arr=data)
        absolute_slope_feat: float = abs(slope_feat)

        def get_gradient_for_each_dimension(arr: ndarray) -> ndarray:
            return gradient(arr)

        first_derivative_data: ndarray = apply_along_axis(
            get_gradient_for_each_dimension, axis=0, arr=data
        )
        first_derivetive_mean_feat: float = nanmean(first_derivative_data, axis=0)
        first_derivative_std_feat: float = nanstd(first_derivative_data, axis=0)

        def get_eda_peaks_info(arr: ndarray) -> dict[str, Any]:
            try:
                eda_peaks_result = eda_peaks(
                    arr,
                    sampling_rate=sampling_rate,
                )
                logger.debug(f"Calculated eda peaks for input {arr}")
            except ValueError as e:
                # NOTE: sometimes, when no peaks are detected, as ValueError is thrown by the
                # neurokit2 method. We solve this in a very simplistic way
                logger.warning(f"Could not extract EDA peaks. Reason: {e}")
                eda_peaks_result: tuple[None, dict[str, Any]] = (
                    None,
                    dict(SCR_Peaks=[], SCR_Amplitude=[0]),
                )

            return len(eda_peaks_result[1]["SCR_Peaks"]), sum(
                eda_peaks_result[1]["SCR_Amplitude"]
            )

        number_of_peaks_feat, peaks_amplitude_feat = apply_along_axis(
            get_eda_peaks_info, axis=0, arr=data
        )

        # eda_peaks_result: dict[str, Any] = eda_peaks(
        #     data,
        #     sampling_rate=sampling_rate,
        # )

        # number_of_peaks_feat: int = len(eda_peaks_result[1]["SCR_Peaks"])
        # # NOTE: I am not sure that the sum of the amplitudes is the correct feature to be
        # # extracted
        # peaks_amplitude_feat: float = sum(eda_peaks_result[1]["SCR_Amplitude"])

        return array(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                absolute_slope_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                number_of_peaks_feat,
                peaks_amplitude_feat,
            ]
        )
