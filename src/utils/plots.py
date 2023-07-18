from collections import defaultdict
from itertools import cycle
from os.path import join as join_paths
from pathlib import Path
from logging import getLogger
from ast import literal_eval

from matplotlib.axes import SubplotBase
from matplotlib.patches import Patch
from matplotlib.pyplot import (
    close,
    cm,
    figure,
    legend as make_legend,
    plot,
    rcParams,
    savefig,
    show,
    style,
    subplots,
    title as figtitle,
    xlabel,
    xticks,
    ylabel,
    errorbar,
    axhline,
    tick_params,
    ylim,
    margins
)
from numpy import asarray, mean, ndarray, std, unique
from pandas import DataFrame, IndexSlice, Series, Timestamp, to_datetime
from seaborn import (
    barplot,
    boxplot,
    heatmap,
    set as set_seaborn,
    set_context,
    set_style,
    set_theme,
    color_palette,
)

logger = getLogger("plots")


def bland_altman_plot(
    data: DataFrame, ax: SubplotBase, *args, **kwargs
) -> tuple[SubplotBase, dict[str, float]]:
    """Simple script to evaluate the Bland-Atlman plot between the left and right signal.

    Parameters
    ----------
    data : DataFrame
        data to be plotted, must have 'left' and 'right' columns
    ax : SubplotBase
        axis to plot the plot

    Returns
    -------
    SubplotBase
        the method returns the input axis
    """
    data1: Series = data.loc[:, "left"]
    data2: Series = data.loc[:, "right"]
    data1: ndarray = asarray(data1)
    data2: ndarray = asarray(data2)
    avg = mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = mean(diff)  # Mean of the difference
    sd = std(diff, axis=0)  # Standard deviation of the difference

    ax.scatter(avg, diff, *args, **kwargs)
    ax.axhline(md, color="gray", linestyle="--")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    return ax, dict(mean_diff=md, std_diff=sd)


def make_lineplot(
    data: ndarray | Series | DataFrame,
    which: str,
    savename: str,
    title: str | None = None,
    style_label: str = "seaborn",
):
    """Simple script to make lineplot for time series data. Different methods shall be
    implemented for different types of data in the project, e.g. EDA, BVP etc.

    Parameters
    ----------
    data : ndarray | Series | DataFrame
        data to be plotted; according to the type of data, different methods shall be
        implemented
    which : str
        defines the type of data; current accepted values: 'EDA'
    savename : str
        filename to save the plot (not the whole path)
    title : str | None, optional
        title for the plot, by default None
    style_label : str, optional
        defines the styl for matplot, by default 'seaborn'

    Raises
    ------
    NotImplementedError
        the type of data accepted for 'EDA' is only ndarray, pandas Series or pandas DataFrame.
        If another is given, the method will fail.
    NotImplementedError
        The function will fail if a `which` value different than the accepted ones is given.
    """
    linestyles: cycle[list[str]] = cycle(["-", "-.", "--", ":"])

    with style.context(style_label):
        match which:
            case ("EDA" | "BVP" | "ACC"):
                figure(figsize=(10, 5))
                if isinstance(data, Series):
                    plot(data.index, data.values, label=which)
                elif isinstance(data, ndarray):
                    plot(data, label=which)
                elif isinstance(data, DataFrame):
                    for i, column in enumerate(data.columns):
                        if isinstance(column, int):
                            column = which
                        if which in column[-1]:
                            data_to_plot = data[column].dropna()
                            plot(
                                data_to_plot.index.get_level_values("timestamp"),
                                data_to_plot,
                                label=column[-1],
                                linewidth=2,
                                alpha=0.7,
                                linestyle=next(linestyles),
                            )
                        make_legend()
                else:
                    raise NotImplementedError(
                        f"Still have not implemented plot for type {type(data)}."
                    )
                if title:
                    figtitle(title)
                xlabel("Time")
                ylabel(f"{which} value (mV)")
                # TODO: imlement using path joining, and not string concatenation
                path_to_save: str = f"./visualizations/{which}/"
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                savefig(join_paths(path_to_save, f"{savename}.pdf"))
            case _:
                raise NotImplementedError(
                    f'Unknown plot type "{which}". Currently implemented: "EDA"'
                )


def statistical_test_plot(
    test_results: DataFrame,
    signal_name: str,
    path_to_save: str = "./visualizations/",
    test_name: str = "Wilcoxon",
    threshold: float = 0.05,
) -> None:
    set_seaborn(font_scale=1.8)
    df_to_save = test_results.iloc[:, 2].unstack(level=1)
    cmap = cm.coolwarm
    figure(figsize=(24, 5))
    heatmap(
        df_to_save,
        xticklabels=df_to_save.columns,
        # vmax=0.2, vmin=-.2, center=0,
        # annot=df_to_save.replace({True: 'Significant', False: 'Non Significant'}, inplace=False).values,
        annot=test_results.iloc[:, 1].round(decimals=3).unstack(level=1).values,
        cmap=cmap,
        fmt="",
        cbar=False,
        yticklabels=df_to_save.index,
    )

    xticks(rotation=30, ha="right")
    xlabel("Feature")
    ylabel("Event")
    figtitle(f"P values of {test_name} test for {signal_name} features")
    custom_handles = [
        Patch(facecolor=cmap(0.0), edgecolor=cmap(0.0), label="Non Significant"),
        Patch(facecolor=cmap(1.0), edgecolor=cmap(1.0), label="Significant"),
    ]
    make_legend(
        handles=custom_handles,
        bbox_to_anchor=(1.253, 1.05),
        title=f"P value significance ({threshold} threshold)",
    )
    savefig(
        join_paths(path_to_save, f"{test_name}_statistical_heatmap_{signal_name}.pdf"),
        bbox_inches="tight",
    )


def cliff_delta_plot(
    cliff_delta_bins: DataFrame,
    cliff_delta_results_vals: DataFrame,
    signal_name: str,
    path_to_save: str = "./visualizations/",
) -> None:
    cmap = cm.coolwarm
    set_seaborn(font_scale=2.89)
    figure(figsize=(13, 9))
    ax = heatmap(
        cliff_delta_bins,
        xticklabels=cliff_delta_results_vals.columns,
        vmax=3,
        vmin=-3,
        center=0,
        annot=cliff_delta_results_vals.round(decimals=3),
        cmap=cmap,
        fmt="",
        cbar=False,
        yticklabels=cliff_delta_results_vals.index.str.replace("_", " "),
    )
    ax.tick_params(left=True, bottom=True)
    xticks(rotation=30, ha="right")
    xlabel("Feature")
    ylabel("Event")
    figtitle(
        f"Cliff Delta values ({signal_name.replace('_filt', '').replace('_', ' ')})",
        fontsize=40,
    )

    def make_custom_handles(bins: DataFrame, cmap) -> list[Patch]:
        labels: list[Patch] = list()
        for el in unique(bins.values):
            if el == 0:
                labels.append(
                    Patch(facecolor=cmap(0.5), edgecolor=cmap(0.5), label="negligible")
                )
            elif el == 1:
                labels.append(
                    Patch(
                        facecolor=cmap(0.6), edgecolor=cmap(0.6), label="small (left)"
                    )
                )
            elif el == 2:
                labels.append(
                    Patch(
                        facecolor=cmap(0.8), edgecolor=cmap(0.8), label="medium (left)"
                    )
                )
            elif el == 3:
                # NOTE: use 1. and not 1, otherwise something goes wronf with cmap
                labels.append(
                    Patch(
                        facecolor=cmap(1.0), edgecolor=cmap(1.0), label="large (left)"
                    )
                )
            elif el == -1:
                labels.append(
                    Patch(
                        facecolor=cmap(0.4), edgecolor=cmap(0.4), label="small (right)"
                    )
                )
            elif el == -2:
                labels.append(
                    Patch(
                        facecolor=cmap(0.2), edgecolor=cmap(0.2), label="medium (right)"
                    )
                )
            elif el == -3:
                labels.append(
                    Patch(
                        facecolor=cmap(0.0), edgecolor=cmap(0.0), label="large (right)"
                    )
                )

        return labels

    custom_handles = make_custom_handles(bins=cliff_delta_bins, cmap=cmap)
    make_legend(
        handles=custom_handles,
        bbox_to_anchor=(1.75, 1.05),
        title="Cliff Delta effect (dominant side)",
    )
    savefig(
        join_paths(path_to_save, f"cliff_delta_{signal_name}.pdf"),
        bbox_inches="tight",
    )


def correlation_heatmap_plot(
    data: DataFrame, signal_name: str, path_to_save: str = "./visualizations/"
) -> None:
    set_seaborn(font_scale=1.6)
    figure(figsize=(7, 5))
    ax = heatmap(
        data,
        xticklabels=data.columns,
        vmax=1,
        vmin=-1,
        center=0,
        cmap="coolwarm",
        yticklabels=data.index.str.replace("_", " "),
        annot=True,
    )
    ax.tick_params(left=True, bottom=True)
    ax.set_ylabel("Event")
    figtitle(
        f"Correlation coefficient per event ({signal_name.replace('_filt', '').replace('_', ' ')})",
        fontsize=20,
    )
    savefig(
        join_paths(path_to_save, f"correlation_heatmap_{signal_name}.pdf"),
        bbox_inches="tight",
    )


def plot_heatmap_boxplot(
    data: dict,
    signal: str,
    data_name: str = "",
    measure_name: str = "",
    nested: bool = False,
    **kwargs,
):
    if nested:
        reform = {
            (outerKey, innerKey): values
            for outerKey, innerDict in data.items()
            for innerKey, values in innerDict.items()
        }
        df_to_save = DataFrame.from_dict(reform).stack(level=1, dropna=False).T
    else:
        df_to_save: Series = Series(data)
        df_to_save: DataFrame = (
            DataFrame(df_to_save, columns=[measure_name]).sort_index().T
        )

    figure(figsize=(len(df_to_save.columns), 1))
    heatmap(
        df_to_save,
        xticklabels=df_to_save.columns,
        vmax=kwargs.get("vmax", 1),
        vmin=kwargs.get("vmin", -1),
        center=kwargs.get("center", 0),
        cmap=kwargs.get("cmap", "coolwarm"),
        yticklabels=df_to_save.index,
        annot=True,
    )
    figtitle(f"{measure_name} per user (signal)")
    savefig(
        f"../visualizations/user_{measure_name}_{signal}_{data_name}.pdf",
        bbox_inches="tight",
    )
    show()

    boxplot(x=df_to_save.iloc[0, :])
    figtitle(f"{measure_name} per user ({signal})")
    savefig(
        f"../visualizations/user_{measure_name}_boxplot_{signal}_{data_name}.pdf",
        bbox_inches="tight",
    )
    show()


def make_biometrics_plots_together_matplotlib(
    data: defaultdict[str, defaultdict[str, defaultdict[str, Series]]],
    user_id: str,
    session_id: str,
    dataset: str,
    experiment_info: DataFrame,
    subset_data: bool = False,
    output_folder: str = "./visualizations/",
    **kwargs,
) -> None:
    set_context("paper")

    data = {key: val for key, val in data.items() if val is not None}
    fig, axs = subplots(
        len(data.keys()), 1, figsize=(14, 11 * len(data.keys())), sharex=True
    )
    if len(data.keys()) == 1:
        axs = [axs]

    for n, (data_type, physiological_data) in enumerate(data.items()):
        if physiological_data is None:
            continue

        for side, specific_side_data in physiological_data.items():
            user_data: DataFrame = specific_side_data[user_id]

            if data_type == "EDA":
                eda_type: str = kwargs["eda_type"]
                data_to_plot = user_data[eda_type]
            else:
                data_to_plot = user_data.iloc[:, 0]

            data_to_plot = data_to_plot.loc[IndexSlice[session_id, :]]
            if subset_data:
                data_to_plot = data_to_plot[:1000]

            data_to_plot.index = to_datetime(data_to_plot.index)
            axs[n].plot(
                data_to_plot.index,
                data_to_plot.values,
                label=side,
                linestyle="-",
            )
            axs[n].set_title(data_type)

        if dataset == "mwc2022":
            start_exp = Timestamp(
                experiment_info.loc[IndexSlice[user_id, session_id], "actual_bed_time"]
            )
            end_exp = Timestamp(
                experiment_info.loc[IndexSlice[user_id, session_id], "wake_up_time"]
            )
            axs[n].axvspan(
                xmin=start_exp,
                xmax=end_exp,
                color="#828282",
                alpha=0.3,
                label="sleep time",
            )
            axs[n].set_ylabel(f"Time")
            axs[n].set_ylabel(f"{data_type}")
        elif dataset == "usi_laughs":
            events = set(
                ["_".join(col.split("_")[1:]) for col in experiment_info.columns]
            )
            for i, event in enumerate(events):
                start_exp = Timestamp(experiment_info.loc[user_id, f"start_{event}"])
                end_exp = Timestamp(experiment_info.loc[user_id, f"end_{event}"])
                if "baseline" in event and i == 0:
                    label = "baseline"
                elif "baseline" not in event:
                    label = event
                else:
                    label = None

                axs[n].axvspan(
                    xmin=start_exp,
                    xmax=end_exp,
                    color="#828282" if "baseline" in event else "#548572",
                    alpha=0.3,
                    label=label,
                )
                axs[n].set_ylabel(f"Time")
                axs[n].set_ylabel(f"{data_type}")
        else:
            raise ValueError(
                f"Received as dataset {dataset}, but only mwc2022 and usi_laughs are supported"
            )

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, fontsize="x-large", title="Side", title_fontsize="xx-large"
    )
    fig.suptitle(
        f"Plots for all singals for user {user_id}, session {session_id}",
        fontsize=30,
        y=0.97,
    )
    output_path = join_paths(
        output_folder,
        f"{kwargs.get('signal_type', None)}_{dataset}_{user_id}_{session_id}.pdf",
    )
    savefig(output_path)
    close()

    return start_exp, end_exp


set_theme()
set_context("paper")


def with_hue(ax, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in ax.patches]
    patch = [p for p in ax.patches]
    k = 0
    for i in range(Number_of_categories):
        for j in range(hue_categories):
            x = (
                patch[(j * Number_of_categories + i)].get_x()
                + patch[(j * Number_of_categories + i)].get_width() / 2
            )
            y = (
                patch[(j * Number_of_categories + i)].get_y()
                + patch[(j * Number_of_categories + i)].get_height()
            )
            ax.annotate("%.2f%%" % (feature.iloc[k]), (x, y), ha="center")
            k += 1


def plot_binary_labels(
    counts: DataFrame,
    title: str,
    dataset_name: str,
    figsize: int,
    output_folder: str = "./visualizations/",
):
    # set_seaborn(font_scale=1)
    
    # set the figure size using the golden ratio
    golden_ratio = (5**0.5 - 1) / 2

    # set seaborn style
    set_style("darkgrid")

    # # set latex font
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.family"] = "STIXGeneral"
    
    # increase font size
    rcParams.update({"font.size": 14})

    fig, ax = subplots(figsize=(figsize, figsize * golden_ratio))
    ax1 = barplot(data=counts, x="side", y="count", hue="label")

    percentages = counts.groupby("side", group_keys=True)["count"].apply(
        lambda x: x / x.sum() * 100
    )
    with_hue(ax1, percentages, 2, 2)

    figtitle(title, fontsize=18)
    make_legend(bbox_to_anchor=(0.9, 1), loc="upper left", borderaxespad=0)
    output_path = join_paths(output_folder, f"label_distribution_{dataset_name}.pdf")
    savefig(output_path, bbox_inches="tight")
    show()


def make_errorplot(
    cliff_deltas: DataFrame,
    type_event: str,
    color: tuple[float],
    ax,
    custom_label: str | None = None,
    elinewidth: int = 30,
    markersize: int = 10,
    component: str = "mixed-EDA",
) -> None:
    bounds = [
        literal_eval(el)
        for el in cliff_deltas.loc[
            IndexSlice[component, :], IndexSlice[type_event, "confidence interval"]
        ].values
    ]
    values = cliff_deltas.loc[IndexSlice[component, :], IndexSlice[type_event, "value"]].astype(float).values
    lower_bounds = [abs(val - el[0]) for el, val in zip(bounds, values)]
    upper_bounds = [abs(el[1] - val) for el, val in zip(bounds, values)]

    if custom_label is None:
        custom_label = type_event

    ax.errorbar(
        x=["_".join(el) for el in cliff_deltas.loc[IndexSlice[component, :], :].index],
        y=values,
        yerr=(lower_bounds, upper_bounds),
        label=custom_label,
        elinewidth=elinewidth,
        linestyle="none",
        markersize=markersize,
        marker=".",
        color=color,
        ecolor=(*color, 0.3),
    )


def cliff_delta_plot(cliff_deltas: DataFrame, dataset: str, figsize: int = 7, path_to_save_figure: str = "./final_visualizations") -> None:
    # set the figure size using the golden ratio
    golden_ratio = (5**0.5 - 1) / 2

    # set seaborn style
    set_style("darkgrid")

    # # set latex font
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.family"] = "STIXGeneral"

    # increase font size
    rcParams.update({"font.size": 14})

    fig, ax = subplots(figsize=(figsize, figsize * golden_ratio))

    colors = color_palette("colorblind", n_colors=2)

    make_errorplot(cliff_deltas, "Cognitive Load", colors[0], "Sleep")
    make_errorplot(cliff_deltas, "Baseline", colors[1], "Awake")

    ylim(-0.6, 0.6)
    margins(x=0.1)

    # set a horizontal line at 0.1 and -0.1
    axhline(0.1, color="grey", linestyle="--")
    axhline(-0.1, color="grey", linestyle="--")

    axhline(0.3, color="black", linestyle="--")
    axhline(-0.3, color="black", linestyle="--")
    ylabel("Cliff's Delta")

    # set the x label to a 30º angle
    xticks(rotation=30, ha="right")
    # put the x label ticks closer
    tick_params(axis="x", which="major", pad=0.01)
    # move the vertical grid lines to the right

    make_legend()
    figtitle("Cliff's delta between left and right hand features")
    savefig(f"../final_visualizations/cliff_delta-{dataset}.pdf", bbox_inches="tight")
    show()