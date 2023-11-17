from ast import literal_eval
from json import load
from sys import path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, MultiIndex, Series, merge
from scipy.stats import norm, ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests

path.append("../")

from src.utils.plots import cliff_delta_plot, make_errorplot


def custom_welch_ztest(
    negative_val: float,
    positive_val: float,
    negative_std: float,
    positive_std: float,
    return_zscore: bool = False,
) -> tuple[float, float] | float:
    z = (negative_val - positive_val) / ((negative_std**2 + positive_std**2) ** 0.5)

    if return_zscore:
        return z, norm.sf(abs(z))
    else:
        # NOTE: I am using the normal distribution because the sample size is large enough (like 40 billion are the degrees of freedom)
        return norm.sf(abs(z))
        # return ttest_ind_from_stats(
        #     mean1=negative_val,
        #     std1=negative_std,
        #     nobs1=n,
        #     mean2=positive_val,
        #     std2=positive_std,
        #     nobs2=n,
        #     equal_var=False,
        # )[1]


def compute_cliff_delta_pvals(
    values_w_intervals: DataFrame, threshold: float = 0.05
) -> DataFrame:
    difference_cliff_delta: Series = values_w_intervals.apply(
        lambda x: custom_welch_ztest(
            x[0],
            x[1],
            (x[0] - literal_eval(x[2])[1]) / 1.96,
            (x[1] - literal_eval(x[3])[1]) / 1.96,
            return_zscore=False,
        ),
        axis=1,
    )
    difference_cliff_delta = DataFrame(difference_cliff_delta, columns=["p-value"])
    # correct for Bonferroni-Holm method
    multi_hypothesis_result = multipletests(
        difference_cliff_delta.values.reshape(
            -1,
        ),
        alpha=threshold,
        method="holm",
    )
    difference_cliff_delta["above_threshold"] = multi_hypothesis_result[0]
    difference_cliff_delta["p-value"] = multi_hypothesis_result[1]
    difference_cliff_delta["negative_bigger?"] = (
        values_w_intervals[("negative", "value")]
        > values_w_intervals[("positive", "value")]
    )
    return difference_cliff_delta


def compute_statistics(
    values_w_intervals, events: list[str] = ["positive", "negative"]
):
    avg_neg = values_w_intervals[("negative", "value")].mean()
    avg_pos = values_w_intervals[("positive", "value")].mean()
    print(f"Average cliff δ for {events[0]}: {avg_pos:.2f}")
    print(f"Average cliff δ for {events[1]}: {avg_neg:.2f}")

    # val_pos_bigger = 0
    # val_neg_bigger = 0
    # val_no_dif = 0
    # for i in range(len(values_w_intervals)):
    #     negative_int = literal_eval(values_w_intervals[('negative', 'confidence interval')][i])
    #     positive_int = literal_eval(values_w_intervals[('positive', 'confidence interval')][i])
    #     if negative_int[0] > positive_int[1]:
    #         val_neg_bigger += 1
    #     elif positive_int[0] > negative_int[1]:
    #         val_pos_bigger += 1
    #     else:
    #         val_no_dif += 1
    cliff_delta_pvals = compute_cliff_delta_pvals(values_w_intervals, threshold=0.05)
    cliff_delta_pvals_significant = cliff_delta_pvals[
        cliff_delta_pvals["above_threshold"] == True
    ]
    mapping_negatives = cliff_delta_pvals_significant["negative_bigger?"].value_counts()
    if len(mapping_negatives) == 2:
        val_pos_bigger = mapping_negatives[
            False
        ]
        val_neg_bigger = mapping_negatives[
            True
        ]
    else:
        if mapping_negatives.index[0] == True:
            val_pos_bigger = 0
            val_neg_bigger = mapping_negatives[True]
        else:
            val_pos_bigger = mapping_negatives[True]
            val_neg_bigger = 0
            
    val_no_dif = len(cliff_delta_pvals[cliff_delta_pvals["above_threshold"] == False])

    val_pos_bigger = val_pos_bigger / len(values_w_intervals)
    val_neg_bigger = val_neg_bigger / len(values_w_intervals)
    val_no_dif = val_no_dif / len(values_w_intervals)

    print(
        f"Percentage of intervals where {events[0]} has larger cliff δ values: {val_pos_bigger*100:.2f}%"
    )
    print(
        f"Percentage of intervals where {events[1]} has larger cliff δ values: {val_neg_bigger*100:.2f}%"
    )
    print(
        f"Percentage of intervals where there is no difference: {val_no_dif*100:.2f}%"
    )
    return cliff_delta_pvals


def prepare_dataframe_with_class_labels_columns(
    input_dict: dict[tuple[str, str, str], Any]
):
    df = DataFrame(input_dict, index=[0])

    df = df.unstack()
    # First, let's reset your DataFrame index
    df_reset = df.reset_index()
    df_reset = df_reset.drop(columns=["level_3"])
    # Then, rename your columns for clarity
    df_reset.columns = ["Class", "Component", "Feature", "cliff delta"]

    # Now, pivot your DataFrame
    df_pivot = df_reset.pivot(
        index="Class", columns=["Component", "Feature"], values="cliff delta"
    )

    # Display the result
    return df_pivot.T


def make_plot(
    values_w_intervals: DataFrame,
    path_to_save: str,
    feature_names: list[str],
    measure_name: str = "",
    plot_title: str = "",
    classes_list: list[str] = ["positive", "negative"],
    ylim: tuple[float, float] = (-0.6, 0.6),
    make_thresholds: bool = True,
    marker: str | None = None,
) -> None:
    # set the figure size using the golden ratio
    golden_ratio = (5**0.5 - 1) / 2
    figsize = 3.2

    # set seaborn style
    sns.set_style("darkgrid")

    # # set latex font
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"

    # increase font size
    plt.rcParams.update({"font.size": 14})

    fig, axs = plt.subplots(1, 3, figsize=(figsize / golden_ratio * 2, figsize))

    colors = sns.color_palette("colorblind", n_colors=2)

    for ax, component in zip(axs, ["mixed-EDA", "tonic-EDA", "phasic-EDA"]):
        for i, (event_type, event_actual_name) in enumerate(
            zip(["positive", "negative"], classes_list)
        ):
            make_errorplot(
                cliff_deltas=values_w_intervals,
                type_event=event_type,
                color=colors[i],
                ax=ax,
                custom_label=event_actual_name,
                elinewidth=10,
                markersize=4,
                component=component,
                marker=marker,
            )
        ax.set_title(component)

    axs[0]
    axs[1].set_ylim(ylim)
    axs[2].set_ylim(ylim)

    
    # set a horizontal line at 0.1 and -0.1
    for i, ax in enumerate(axs):
        ax.set_ylim(ylim)
        if make_thresholds:
            ax.axhline(0.1, color="grey", linestyle="--")
            ax.axhline(-0.1, color="grey", linestyle="--")

            ax.axhline(0.3, color="black", linestyle="--")
            ax.axhline(-0.3, color="black", linestyle="--")
        if i == 0:
            ax.set_ylabel(measure_name)
        else:
            ax.get_yaxis().set_visible(False)

        # set the x label to a 30º angle
        ax.set_xticklabels(
            feature_names, rotation=30, ha="right", fontsize=10, rotation_mode="anchor"
        )
        ax.tick_params(axis="x", which="major", pad=-3)

    cliff_delta_pvals = compute_cliff_delta_pvals(values_w_intervals, threshold=0.05)
    cliff_delta_pvals = cliff_delta_pvals.reindex(
        ["mixed-EDA", "tonic-EDA", "phasic-EDA"], level=0
    )

    # display(cliff_delta_pvals)
    # iterate over the labels, and make them bold if they are significant
    i = 0
    for ax in axs:
        for _, label in enumerate(ax.get_xticklabels()):
            if cliff_delta_pvals["above_threshold"][i]:
                label.set_weight("bold")
                label.set_color("red")
            else:
                label.set_color("black")
            i += 1

    # move the vertical grid lines to the right

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=None
    )
    plt.legend()
    plt.suptitle(
        plot_title, fontsize=16, y=1.05
    )
    plt.savefig(path_to_save, bbox_inches="tight")
    plt.show()


def load_and_prepare_data(path_to_data: str) -> tuple[DataFrame, list[str]]:
    with open(path_to_data, "r") as f:
        cliff_delta_results = load(f)

    values = {
        (key1, key2, key3): val3[0]
        for key1, val1 in cliff_delta_results.items()
        for key2, val2 in val1.items()
        for key3, val3 in val2.items()
    }
    confidence_intervals = {
        (key1, key2, key3): str(val3[1])
        for key1, val1 in cliff_delta_results.items()
        for key2, val2 in val1.items()
        for key3, val3 in val2.items()
    }
    values = prepare_dataframe_with_class_labels_columns(values)
    confidence_intervals = prepare_dataframe_with_class_labels_columns(
        confidence_intervals
    )

    values_w_intervals = merge(
        left=values, right=confidence_intervals, left_index=True, right_index=True
    )
    values_w_intervals.columns = MultiIndex.from_tuples(
        [
            ("negative", "value"),
            ("positive", "value"),
            ("negative", "confidence interval"),
            ("positive", "confidence interval"),
        ]
    )
    values_w_intervals = values_w_intervals.dropna(how="any", inplace=False, axis=0)
    feature_names = values_w_intervals.index.get_level_values(1).unique().tolist()
    return values_w_intervals, feature_names