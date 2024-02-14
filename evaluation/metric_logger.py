import time
import warnings
from abc import abstractmethod
from typing import Dict, List, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "figure.titlesize": 35,
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "axes.titlesize": 27,
    "axes.labelsize": 30,
    "xtick.labelsize": 27,
    "ytick.labelsize": 27,
    "legend.fontsize": 30,
    "legend.labelspacing": 0.0,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.1,
    "legend.columnspacing": 0.5,
    "text.usetex": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    #"font.serif": ["Computer Modern Serif"],
}
plt.rcParams.update(params)

markers_colors = [
    ("o", "tab:blue"),
    ("^", "tab:orange"),
    ("s", "tab:green"),
    ("p", "tab:red"),
    ("h", "tab:brown"),
    ("+", "tab:pink"),
    ("x", "tab:gray"),
    ("D", "tab:olive"),
    ("d", "tab:cyan"),
]

bar_markers_colors = [
    ("s", "tab:green"),
    ("^", "tab:orange"),
    ("o", "tab:blue"),
    ("p", "tab:red"),
    ("h", "tab:brown"),
    ("+", "tab:pink"),
    ("x", "tab:gray"),
    ("D", "tab:olive"),
    ("d", "tab:cyan"),
]


class MetricLogger:
    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, state, reward, done, truncated, info) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def aggregate_ep_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def aggregate_all_metrics(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        raise NotImplementedError

    def plot(
        self,
        all_stats,
        outdir=None,
        x_key: str = "gamma_name",
        gbys=[],
        margin_percent=0.05,
        barplot=False,
        plot_type: str = "default",
        legend=True,
    ):
        linestyles = ["-"]

        metrics = self.metrics

        if plot_type == "bar":
            for i, gby in enumerate(gbys):
                alls = set(all_stats[gby])
                nrows, ncols = 1, len(alls)
                fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))

                # add margin on bottom for legend, top for title
                fig.subplots_adjust(bottom=0.2, top=0.85)

                plot_bars(
                    all_stats=all_stats,
                    logger=self,
                    x_key=x_key,
                    gby=gby,
                    axes=axes,
                    margin_percent=margin_percent,
                )

                if outdir:
                    time_id = int(time.time())
                    plt.savefig(
                        outdir / f"bars_{x_key}_gby{gby}_{time_id}.pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        outdir / f"bars_{x_key}_gby{gby}_{time_id}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                else:
                    plt.show()

        elif plot_type == "default":
            # prepare axes
            nrows, ncols = 1, len(metrics)

            for i, gby in enumerate(gbys):
                fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
                axes = np.array([axes]) if len(metrics) == 1 else axes
                axes = axes.reshape(nrows, ncols)

                # add margin on top for legend
                fig.subplots_adjust(top=0.85)

                ls = linestyles[i % len(linestyles)]
                plot_gby(
                    all_stats=all_stats,
                    logger=self,
                    x_key=x_key,
                    gby=gby,
                    axes=axes,
                    linestyle=ls,
                    margin_percent=margin_percent,
                    barplot=barplot,
                    legend=legend,
                )

                if outdir:
                    time_id = int(time.time())
                    plt.savefig(
                        outdir / f"all_stats_{x_key}_gby{gby}_{time_id}.pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        outdir / f"all_stats_{x_key}_gby{gby}_{time_id}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                else:
                    plt.show()


def plot_gby(
    all_stats: Dict[str, List[float]],
    logger: MetricLogger,
    x_key: str,
    gby: str,
    axes,
    linestyle="-",
    margin_percent=0.05,
    barplot: bool = False,
    legend=True,
):
    nrows, ncols = axes.shape
    metrics = logger.metrics
    titles = [logger.metrics_meta[m]["title"] for m in metrics]

    # gradient from blue to red, 8 colors
    barcolor = [
        "#%02x%02x%02x" % (i * 255 // 8, 0, 255 - i * 255 // 8) for i in range(8)
    ]

    alls = sorted(set(all_stats[gby]))
    for i, val in enumerate(alls):
        stat_ids = np.where(np.array(all_stats[gby]) == val)[0]
        marker = markers_colors[i % len(markers_colors)][0]
        color = markers_colors[i % len(markers_colors)][1]

        for j, (metric, label) in enumerate(zip(metrics, titles)):
            r, c = j // ncols, j % ncols

            try:
                gammas = [all_stats[x_key][i] for i in stat_ids]
                values = [all_stats[metric][i] for i in stat_ids]
            except KeyError:
                warnings.warn(f"Key {metric} not found in all_stats")
                continue

            fn = logger.metrics_meta[metric]["best_fn"]

            # average grouped by gamma
            gs, vs = [], []
            for g in sorted(set(gammas)):
                gs.append(g)
                vs.append(np.mean([v for v, gg in zip(values, gammas) if gg == g]))

            # sort by gamma
            sorted_idx = np.argsort(gs)
            gammas = np.array(gs)[sorted_idx]
            values = np.array(vs)[sorted_idx]

            # find best gamma
            best_value = fn(values)
            best_gammas = gammas[values == best_value]
            best_values = values[values == best_value]

            min_y, max_y = logger.metrics_meta[metric]["all_range"]
            y_range = max_y - min_y

            ax = axes[r, c]
            # ax.set_title(f"{label}")
            ax.set_ylabel(f"{label}")

            if gby in logger.params_meta:
                label = logger.params_meta[gby]["label"]
                val_fn = logger.params_meta[gby]["val_fn"]
                line_label = f"{label}={val_fn(val)}"
            else:
                line_label = f"{gby}={val:.2f}"

            xx = gammas
            yy = values

            if barplot:
                bar_width = 0.1

                if x_key in logger.params_meta:
                    bar_width = logger.params_meta[x_key]["bar_width"]
                    color = "blue"
                    ax_distr = ax.twinx()

                    if "train_distr" in logger.params_meta[x_key]:
                        train_distr = logger.params_meta[x_key]["train_distr"]
                        distr_type, distr_params = train_distr[0], train_distr[1:]

                        if distr_type == "normal":
                            mu, stddev = distr_params
                            x = np.linspace(mu - 3 * stddev, mu + 3 * stddev, 100)
                            ax_distr.plot(
                                x,
                                stats.norm.pdf(x, mu, stddev),
                                color=color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.75,
                            )
                            ax_distr.fill_between(
                                x,
                                stats.norm.pdf(x, mu, stddev),
                                color=color,
                                alpha=0.025,
                            )
                        elif distr_type == "uniform":
                            a, b = distr_params
                            x = np.linspace(a, b, 100)
                            ax_distr.plot(
                                x,
                                stats.uniform.pdf(x, a, b - a),
                                color=color,
                                linestyle="--",
                                linewidth=1,
                                alpha=0.75,
                            )
                            ax_distr.fill_between(
                                x,
                                stats.uniform.pdf(x, a, b - a),
                                color=color,
                                alpha=0.025,
                            )
                        elif distr_type == "categorical":
                            distr_params = np.array(distr_params[0])
                            # visualize categorical distribution as a histogram
                            ax_distr.hist(
                                np.arange(len(distr_params)),
                                weights=distr_params,
                                bins=100,
                                linestyle="--",
                                edgecolor=color,
                                histtype="stepfilled",
                                linewidth=1,
                                alpha=0.75,
                            )

                        ax_distr.spines["top"].set_visible(False)
                        ax_distr.spines["right"].set_visible(False)
                        ax_distr.set_yticks([])
                        ax_distr.set_ylim(0, 50)

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    # linewidth 2 for other spines
                    ax.spines["left"].set_linewidth(2)
                    ax.spines["bottom"].set_linewidth(2)

                    if "ticks" in logger.params_meta[x_key]:
                        xticks = logger.params_meta[x_key]["ticks"]
                        if isinstance(xticks, tuple):
                            ax.set_xticks(*xticks)
                        else:
                            ax.set_xticks(xticks)

                    if "xlims" in logger.params_meta[x_key]:
                        xlims = logger.params_meta[x_key]["xlims"]
                        ax.set_xlim(xlims)

                    if "ticks" in logger.metrics_meta[metric]:
                        yticks = logger.metrics_meta[metric]["ticks"]
                        ax.set_yticks(yticks)

                color = barcolor[i % len(barcolor)]
                ax.bar(
                    xx, yy, color=color, edgecolor="black", width=bar_width, linewidth=3
                )
            else:
                ax.plot(
                    xx,
                    yy,
                    color=color,
                    linestyle="--",
                )

                ax.scatter(
                    gammas,
                    values,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    label=line_label,
                )

                ax.scatter(
                    best_gammas,
                    best_values,
                    marker="*",
                    color=color,
                    s=250,
                )
            ax.set_ylim(
                min_y - margin_percent * y_range, max_y + margin_percent * y_range
            )

            if x_key in logger.params_meta:
                ax.set_xlabel(logger.params_meta[x_key]["label"])
            else:
                ax.set_xlabel(x_key)

            # find index of gamma_name Adaptive
            # color plot background of those x values of green
            adaptive_idx = np.where(xx == "Ada")[0]
            if len(adaptive_idx) > 0:
                ticks = ax.get_xticks()
                delta_ticks = ticks[1] - ticks[0] if len(ticks) > 1 else 0.5
                begin = ticks[adaptive_idx[0]] - delta_ticks / 2
                end = ticks[-1] + delta_ticks / 2
                ax.axvspan(
                    begin,
                    end,
                    alpha=0.05,
                    color="green",
                )
                ax.set_xlim(None, end)

            # printout metric report
            print(f"Group-by:  {gby}={val}, metric={metric}")
            for g, v in zip(gammas, values):
                print(f"{g}: {v:.2f}")
            print()

    # get legend handles from last plot
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if not barplot:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="*",
                    color="k",
                    linestyle="None",
                    label="Best Value",
                    markerfacecolor="k",
                    markersize=20,
                )
            )

        # add train distribution to the legend
        if barplot:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="blue",
                    linestyle="--",
                    label="Train Distribution",
                )
            )

        # add legend to the figure, using the handles from the last plot
        plt.gcf().legend(
            handles=handles,
            ncols=len(handles),
            loc="upper center",
        )


def plot_pareto(
    all_stats: Dict[str, List[float]],
    logger: MetricLogger,
    x_key: str,
    gby: str,
    axes: List[plt.Axes],
    linestyle="-",
    margin_percent=0.05,
):
    all_types_gamma = ["S-CBF", "OD-CBF", "Adaptive", "Other"]

    def get_type_gamma(gamma):
        if isinstance(gamma, float) or (
            isinstance(gamma, str) and gamma.replace(".", "", 1).isdigit()
        ):
            # if gamma can be cast to float
            return "S-CBF"
        if isinstance(gamma, str) and gamma.startswith("OD"):
            # if gamma is a string and starts with OD
            return "OD-CBF"
        if isinstance(gamma, str) and gamma.startswith("Ada"):
            # if gamma is a string and starts with Ada
            return "Adaptive"
        return "Other"

    metrics = logger.metrics
    assert len(metrics) == 2, "Pareto plot only supports 2 metrics"

    alls = set(all_stats[gby])
    titles = [logger.metrics_meta[m]["title"] for m in metrics]

    for i, (val, ax) in enumerate(zip(alls, axes)):
        gby_name = (
            logger.params_meta[gby]["label"] if gby in logger.params_meta else gby
        )
        ax.set_title(f"{gby_name}={val}")
        ax.set_xlabel(titles[0])
        ax.set_ylabel(titles[1])

        stat_ids = np.where(np.array(all_stats[gby]) == val)[0]

        gammas = [all_stats[x_key][i] for i in stat_ids]
        gammas_types = [get_type_gamma(g) for g in gammas]

        for type_gamma in all_types_gamma:
            k = all_types_gamma.index(type_gamma)
            xs = [all_stats[metrics[0]][i] for i in stat_ids]
            ys = [all_stats[metrics[1]][i] for i in stat_ids]

            # filter by gamma type
            xs = [x for x, g in zip(xs, gammas_types) if g == type_gamma]
            ys = [y for y, g in zip(ys, gammas_types) if g == type_gamma]

            # skip empty
            if len(xs) == 0:
                continue

            # mark dominated and non-dominated points (maximize both metrics)
            """
            dominated = []
            for i1, (x, y) in enumerate(zip(xs, ys)):
                dominated.append(False)
                for j1, (x2, y2) in enumerate(zip(xs, ys)):
                    if i1 == j1:
                        continue
                    if x2 > x and y2 > y:
                        dominated[-1] = True
                        break
            """

            marker = markers_colors[k % len(markers_colors)][0]
            color = markers_colors[k % len(markers_colors)][1]

            ax.scatter(xs, ys, color=color, marker=marker, label=type_gamma)

            """
            # plot dominated points with alpha=0.5
            ax.scatter(
                [x for i, x in enumerate(xs) if dominated[i]],
                [y for i, y in enumerate(ys) if dominated[i]],
                color=color,
                marker=marker,
                s=10,
            )

            # plot non-dominated points with alpha=1
            ax.scatter(
                [x for i, x in enumerate(xs) if not dominated[i]],
                [y for i, y in enumerate(ys) if not dominated[i]],
                color=color,
                marker=marker,
                s=100,
            )
            """

        # based on best_fn, sort the axis (e.g., best is min, then sort descending)
        best_fn_x = logger.metrics_meta[metrics[0]]["best_fn"]
        best_fn_y = logger.metrics_meta[metrics[1]]["best_fn"]

        if best_fn_x == np.min:
            ax.invert_xaxis()
        if best_fn_y == np.min:
            ax.invert_yaxis()

    # legend
    handles, labels = ax.get_legend_handles_labels()
    plt.gcf().legend(
        handles=handles,
        ncols=len(handles),
        loc="lower center",
    )


def plot_bars(
    all_stats: Dict[str, List[float]],
    logger: MetricLogger,
    x_key: str,
    gby: str,
    axes: List[plt.Axes],
    linestyle="-",
    margin_percent=0.05,
):
    metrics = logger.metrics
    barwidth = 0.9 / len(metrics)
    all_types_gamma = ["S-CBF", "OD-CBF", "Adaptive", "Other"]

    def get_type_gamma(gamma):
        if isinstance(gamma, float) or (
            isinstance(gamma, str) and gamma.replace(".", "", 1).isdigit()
        ):
            # if gamma can be cast to float
            return "S-CBF"
        if isinstance(gamma, str) and gamma.startswith("OD"):
            # if gamma is a string and starts with OD
            return "OD-CBF"
        if isinstance(gamma, str) and "Ada" in gamma:
            # if gamma is a string and starts with Ada
            return "Adaptive"
        return None

    alls = sorted(set(all_stats[gby]))
    colors = ["tab:green", "tab:red", "tab:blue", "tab:orange"]
    linestyles = ["-", "--", "-.", ":"]

    for i, (val, ax) in enumerate(zip(alls, axes)):
        label = logger.params_meta[gby]["label"]
        val_fn = logger.params_meta[gby]["val_fn"]
        title = f"{label}={val_fn(val)}"
        ax.set_title(title)

        stat_ids = np.where(np.array(all_stats[gby]) == val)[0]

        for im, metric in enumerate(metrics):
            gammas = [all_stats[x_key][i] for i in stat_ids]
            xs = [get_type_gamma(g) for g in gammas]
            ys = [all_stats[metric][i] for i in stat_ids]

            # normalize metric "length" to 0..1
            if metric == "length":
                _, maxlen = logger.metrics_meta["length"]["all_range"]
                ys = [y / maxlen for y in ys]

            # aggregate by type_gamma
            unique_xs = list(set(xs))
            ys_mean = [
                np.mean([y for x, y in zip(xs, ys) if x == type_gamma])
                for type_gamma in unique_xs
            ]
            ys_min = [
                np.min([y for x, y in zip(xs, ys) if x == type_gamma])
                for type_gamma in unique_xs
            ]
            ys_max = [
                np.max([y for x, y in zip(xs, ys) if x == type_gamma])
                for type_gamma in unique_xs
            ]

            # relative errors
            ys_min = [y - y_min for y, y_min in zip(ys_mean, ys_min)]
            ys_max = [y_max - y for y, y_max in zip(ys_mean, ys_max)]

            # skip empty
            if len(unique_xs) == 0:
                continue

            # plot error bar
            xticks = [
                float(all_types_gamma.index(x)) + im * barwidth for x in unique_xs
            ]

            for ip in range(len(xticks)):
                label_metric = logger.metrics_meta[metric]["title"] if ip == 0 else None

                if logger.metrics_meta[metric]["best_fn"] == np.min:
                    marker = "v"
                elif logger.metrics_meta[metric]["best_fn"] == np.max:
                    marker = "^"
                else:
                    marker = "o"

                ax.vlines(
                    xticks[ip],
                    0.0,
                    ys_mean[ip],
                    color=colors[im % len(linestyles)],
                    linestyle=linestyle,
                    linewidth=barwidth * 100,
                    alpha=0.2,
                    label=label_metric,
                )
                ax.scatter(
                    xticks[ip],
                    ys_mean[ip],
                    color=colors[im % len(linestyles)],
                    marker=marker,
                    s=250,
                    alpha=0.6,
                )

                # add error bar
                error_alpha = 0.4
                ax.vlines(
                    xticks[ip],
                    ys_mean[ip] - ys_min[ip],
                    ys_mean[ip] + ys_max[ip],
                    color=colors[im % len(linestyles)],
                    linestyle=linestyle,
                    alpha=error_alpha,
                )
                ax.scatter(
                    xticks[ip],
                    ys_mean[ip] - ys_min[ip],
                    color=colors[im % len(linestyles)],
                    marker="_",
                    s=100,
                    alpha=error_alpha,
                )
                ax.scatter(
                    xticks[ip],
                    ys_mean[ip] + ys_max[ip],
                    color=colors[im % len(linestyles)],
                    marker="_",
                    s=100,
                    alpha=error_alpha,
                )

        # axis limits
        ax.set_xlim(-barwidth, len(unique_xs) - 1 + barwidth * len(metrics))
        ax.set_ylim(0.0, 1.0)

        # axis ticks
        # mark adaptive with bold
        ada_label = r"\textbf{Adaptive}" if params["text.usetex"] else "Adaptive"
        xtickslabels = [
            ada_label if x == "Adaptive" else x for x in unique_xs
        ]
        ax.set_xticks(
            [
                float(all_types_gamma.index(x)) + (len(metrics) - 1) * barwidth / 2
                for x in unique_xs
            ],
            xtickslabels,
        )

        # axis labels
        # ax.set_ylabel("Rate (\%)")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    # add two triangle markers (facing up and down) for "higher is better" and "lower is better"
    marker_size, color, facecolor = 20, "k", "w"
    handles += [
        Line2D(
            [],
            [],
            marker="v",
            color=color,
            markerfacecolor=facecolor,
            markersize=marker_size,
            linestyle="None",
            label="Lower is better",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color=color,
            markerfacecolor=facecolor,
            markersize=marker_size,
            linestyle="None",
            label="Higher is better",
        ),
    ]
    # add delimiter of errors to legend to denote min/max
    marker_size = 10
    handles += [
        Line2D(
            [],
            [],
            color="k",
            linestyle="-",
            marker=2,
            markersize=marker_size,
            label="Min",
        ),
        Line2D(
            [],
            [],
            color="k",
            linestyle="-",
            marker=3,
            markersize=marker_size,
            label="Max",
        ),
    ]

    plt.gcf().legend(
        handles=handles,
        ncols=len(handles),
        loc="lower center",
    )

    # use env as title
    plt.gcf().suptitle(logger.env_name)


class ParticleEnvMetricLogger(MetricLogger):
    env_name = "Multi-Robot Navigation"

    params_meta = {
        "horizonset": {
            "label": "$D_{s, others}$",
            "val_fn": lambda x: f"{x:.2f}",
        },
        "nagents": {
            "label": "Nr Agents",
            "val_fn": lambda x: f"{x:.0f}",
        },
        "gamma": {
            "label": "CBF $\gamma$",
            "val_fn": lambda x: f"{x:.2f}",
        },
        "gamma_name": {
            "label": "CBF $\gamma$",
            "val_fn": lambda x: f"{x:.2f}",
        },
    }

    metrics_meta = {
        "success": {
            "title": "Success Rate",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [0, 1],
        },
        "collision": {
            "title": "Collision Rate",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.min,
            "all_range": [0, 1],
        },
        "starvation": {
            "title": "Starvation Rate (\%)",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.min,
            "all_range": [0, 1],
        },
        "min_dist": {
            "title": "Avg DtC (m)",
            "ep_aggregator": np.min,
            "all_aggregator": lambda xx: np.mean(
                [x - 5.0 for x in xx if x - 5.0 > 0]
            ),  # ignore unsuccessful episodes, where min_dist <= 0
            "best_fn": np.max,
            "all_range": [0, 10],
        },
        "unfeasible": {
            "title": "Unfeasible Rate",
            "ep_aggregator": lambda x: np.sum(x) > 0,
            "all_aggregator": np.mean,
            "best_fn": np.min,
            "all_range": [0, 1],
        },
        "feasible": {
            "title": "Feasible Rate",
            "ep_aggregator": lambda x: np.sum(x) == 0,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [0, 1],
        },
        "length": {
            "title": "Norm. Episode Length",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [0, 50],
        },
    }

    def __init__(self, metrics=None):
        if metrics is None:
            metrics = [
                "success",
                "collision",
                "starvation",
                "min_dist",
                "unfeasible",
                "feasible",
                "length",
            ]
        self._metrics = metrics

    @property
    def metrics(self) -> List[str]:
        return self._metrics

    def get_metrics(self, state, reward, done, truncated, info) -> Dict[str, float]:
        all_metrics = {
            "success": info["success"],
            "collision": info["collision"],
            "starvation": done and not info["success"] and not info["collision"],
            "min_dist": info["min_dist"],
            "unfeasible": info["unfeasible"],
            "feasible": info["unfeasible"],
            "length": 1,
        }
        return {k: float(v) for k, v in all_metrics.items() if k in self.metrics}

    def aggregate_ep_metrics(
        self, metrics: Dict[str, Iterable[float]]
    ) -> Dict[str, float]:
        assert set(metrics.keys()) == set(self.metrics)
        return {k: self.metrics_meta[k]["ep_aggregator"](v) for k, v in metrics.items()}

    def aggregate_all_metrics(
        self, metrics: Dict[str, Iterable[float]]
    ) -> Dict[str, float]:
        all_metrics = {k: v for k, v in metrics.items() if k in self.metrics}

        # warn if any metric is missing
        missing_metrics = set(self.metrics) - set(all_metrics.keys())
        if len(missing_metrics) > 0:
            warnings.warn(f"Missing metrics: {missing_metrics}")

        return {
            k: self.metrics_meta[k]["all_aggregator"](v) for k, v in all_metrics.items()
        }


class ParticleEnvStateSuccessLogger(MetricLogger):
    @property
    def metrics(self) -> List[str]:
        return ["state", "gamma", "success", "collision"]

    def get_metrics(self, state, reward, done, truncated, info) -> Dict[str, float]:
        return {
            "state": state,
            "gamma": info["gamma"],
            "success": info["success"],
            "collision": info["collision"],
        }

    def aggregate_ep_metrics(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, Union[float, np.ndarray]]:
        all_states = np.array(metrics["state"])
        trajectory_dict = {
            f"agent_{idx}": all_states[:, idx] for idx in range(all_states.shape[1])
        }
        gamma_seq = np.array(metrics["gamma"])
        return {
            "trajectory": trajectory_dict,
            "gamma": gamma_seq,
            "success": np.sum(metrics["success"]),
            "collision": np.sum(metrics["collision"]),
        }

    def aggregate_all_metrics(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "Not intended to be implemented for this state logger."
        )


class F110MetricLogger(MetricLogger):
    env_name = "Multi-Agent Racing"

    params_meta = {
        "vgain": {
            "label": "$v_{opp}$",
            "val_fn": lambda x: f"{x:.2f}",
            "train_distr": ("normal", 0.60, 0.05),
            "bar_width": 0.03,
            "ticks": [0.5, 0.6, 0.7, 0.8],
            "xlims": [0.4, 0.9],
        },
        "nagents": {
            "label": "Nr Agents",
            "val_fn": lambda x: f"{x:.0f}",
            "train_distr": (
                "categorical",
                [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            "bar_width": 0.5,
            "ticks": [3, 4, 5, 6, 7, 8, 9],
            "xlims": [2, 10],
        },
        "npctype": {
            "label": "Opponents' Planner",
            "val_fn": lambda x: ["PP", "Lattice", "FTG"][
                int(np.floor(x))
            ],  # assume int denotes type, decimal speed
            "train_distr": ("normal", 0.60, 0.05),
            "bar_width": 0.05,
            "ticks": (
                [0.65, 1.65, 2.65],
                [
                    "PP",
                    "Lattice",
                    "FTG",
                ],
            ),
            "xlims": [0, 3],
        },
        "gamma": {
            "label": "CBF $\gamma$",
            "val_fn": lambda x: f"{x:.2f}",
        },
        "gamma_name": {
            "label": "CBF $\gamma$",
            "val_fn": lambda x: f"{x:.2f}",
        },
    }

    metrics_meta = {
        "success": {
            "title": "Success Rate",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [0, 1],
        },
        "collision": {
            "title": "Collision Rate",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.min,
            "all_range": [0, 1],
        },
        "reward": {
            "title": "Avg Reward",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [-2, +2],
        },
        "rank": {
            "title": "Avg Rank",
            "ep_aggregator": lambda xx: xx[-1],
            "all_aggregator": np.mean,
            "best_fn": np.min,
            "all_range": [1, 4],
            "ticks": [1, 2, 3, 4, 5],
        },
        "length": {
            "title": "Norm. Episode Length",
            "ep_aggregator": np.sum,
            "all_aggregator": np.mean,
            "best_fn": np.max,
            "all_range": [0, 150],
        },
    }

    def __init__(self, metrics=None):
        super().__init__()
        if metrics is None:
            metrics = ["success", "collision", "reward", "rank", "length"]
        self._metrics = metrics

    @property
    def metrics(self) -> List[str]:
        return self._metrics

    def get_metrics(self, state, reward, done, truncated, info) -> Dict[str, float]:
        ranking = [(k, state[k]["frenet_coords"][0]) for k in state]
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        rank = 1 + ranking.index(
            ("ego", state["ego"]["frenet_coords"][0])
        )  # start at 1

        all_metrics = {
            "success": float(done and rank == 1),
            "reward": reward,
            "rank": rank,
            "collision": state["ego"]["collision"],
            "length": 1,
        }
        return {k: float(v) for k, v in all_metrics.items() if k in self.metrics}

    def aggregate_ep_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: self.metrics_meta[k]["ep_aggregator"](v) for k, v in metrics.items()}

    def aggregate_all_metrics(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        all_metrics = {k: v for k, v in metrics.items() if k in self.metrics}

        # warn if any metric is missing
        missing_metrics = set(self.metrics) - set(all_metrics.keys())
        if len(missing_metrics) > 0:
            warnings.warn(f"Missing metrics: {missing_metrics}")

        return {
            k: self.metrics_meta[k]["all_aggregator"](v) for k, v in all_metrics.items()
        }


def logger_factory(env_id: str, **kwargs) -> MetricLogger:
    if "particle-env" in env_id:
        return ParticleEnvMetricLogger(**kwargs)
    elif "f110-multi-agent" in env_id:
        return F110MetricLogger(**kwargs)
    else:
        raise NotImplementedError()
