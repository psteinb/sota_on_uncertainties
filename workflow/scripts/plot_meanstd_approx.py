from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import norm


def main(incsvfiles, outplot, legend=True, legend_title=True, errorbars=True):

    dfs = []
    for fname in incsvfiles:
        descr = "unknown"
        if "folds_and_seeds" in fname:
            descr = "allseeds"

        df = pd.read_csv(fname)

        if df.columns.str.contains("description").sum() == 0:
            df["description"] = descr

        df["estimate"] = "fold samples"
        df["estimate"] = df["estimate"].astype("category")
        dfs.append(df)

        # appending a binomial approximation
        dfapprox = df.copy()
        # dfapprox["description"] = len(df) * [str(df["description"][0] + ", binomial")]
        # dfapprox["description"] = dfapprox["description"].astype("category")
        dfapprox["estimate"] = "approximated"
        dfapprox["estimate"] = dfapprox["estimate"].astype("category")
        #
        ## compute normal approximation based correction factor
        cdf_2sigma = norm.cdf([-2, 2])
        integral_2sigma = cdf_2sigma[-1] - cdf_2sigma[0]

        z_std_2sigma = norm.ppf(
            1 - ((1 - integral_2sigma) / 2.0)
        )  # 95% confidence interval = 1.959, equal 2 sigma
        print(
            f"95% confidence interval: integral_range = {integral_2sigma}, correction {z_std_2sigma}"
        )
        cdf_1sigma = norm.cdf([-1, 1])
        integral_1sigma = cdf_1sigma[-1] - cdf_1sigma[0]  # should be 1.!

        z_std = norm.ppf(
            1 - ((1 - integral_1sigma) / 2.0)
        )  # 1 sigma confidence interval
        print(
            f"68.2% confidence interval: integral_range = {integral_1sigma}, correction {z_std}"
        )

        #
        inv_samples = 1.0 / df["nsamples"]
        dfapprox["val_accuracy_std"] = z_std * np.sqrt(
            inv_samples
            * dfapprox["val_accuracy_mean"]
            * (1 - dfapprox["val_accuracy_mean"])
        )

        dfs.append(dfapprox)

    fulldf = pd.concat(dfs).reset_index()
    fulldf["val_accuracy_mean_max"] = (
        fulldf["val_accuracy_mean"] + fulldf["val_accuracy_std"]
    )
    fulldf["val_accuracy_mean_min"] = (
        fulldf["val_accuracy_mean"] - fulldf["val_accuracy_std"]
    )

    mask42 = fulldf["description"].str.contains("42")
    fulldf.loc[mask42, "description"] = "42"
    fulldf.loc[~mask42, "description"] = "any"
    fulldf["description"] = fulldf["description"].astype("category")

    df = fulldf.copy()
    print(
        f"using dataframe of {df.shape} with these counts:\n{df.description.value_counts()}"
    )

    print(df.head())

    maxy = df["val_accuracy_mean_max"].max()
    miny = df["val_accuracy_mean_min"].min()

    minytick = np.round(np.floor(100.0 * miny) / 100.0, decimals=1)
    maxytick = np.round(np.ceil(100.0 * maxy) / 100.0, decimals=1)
    yticks = np.arange(minytick - 0.025, maxytick + 0.026, step=0.025)
    print(f"y: {miny,maxy} -> {minytick,maxytick}")
    print(yticks)
    plt = (
        ggplot(
            df,
            aes(x="arch", y="val_accuracy_mean", color="description", shape="estimate"),
        )
        + geom_point(position=position_dodge(width=0.75), size=3)
        + geom_errorbar(
            aes(ymin="val_accuracy_mean_min", ymax="val_accuracy_mean_max"),
            position=position_dodge(width=0.75),
            width=0.2,
        )
        # + xlab("architecture")
        + scale_x_discrete(name="")
        + ylab("accuracy")
        # + ylim(0.98 * miny, 1.02 * maxy)
        + scale_y_continuous(breaks=yticks)
        + scale_color_discrete(breaks=["any", "42"])
        + scale_shape_discrete(breaks=["fold samples", "approximated"])
        + theme_light()
        + theme(legend_key=element_rect(fill="None", colour="None"))
        + guides(  # theme(legend_position="top", legend_direction="horizontal")
            # nrow=1,
            color=guide_legend(
                # nrow=1,
                # direction="horizontal",
                # label_position="right",
                title="none"
                if not legend_title
                else "seed"
                # title_vjust=0.5
                # title_separation=0.3,
                # label_separation=0.2,
            ),
            shape=guide_legend(
                # nrow=1,
                # direction="horizontal",
                # label_position="right",
                title="none"
                if not legend_title
                else "uncertainty from"
                # title_vjust=0.5
                # title_separation=0.3,
                # label_separation=0.2,
            ),
        )
        + coord_flip()
    )

    if not legend:
        print("removing legend")
        # http://www.cookbook-r.com/Graphs/Legends_(ggplot2)/
        plt += theme(legend_position="none")

    ggsave(plt, str(outplot), width=6, height=3)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        filter_by_seed_ = -42  # default is to NOT filter by seed
        if hasattr(snakemake, "wildcards") and hasattr(snakemake.wildcards, "seedval"):
            filter_by_seed_ = snakemake.wildcards.seedval

        show_legend = True
        show_legend_title = True
        if hasattr(snakemake, "params") and hasattr(snakemake.params, "show_legend"):
            show_legend = snakemake.params.show_legend
        if hasattr(snakemake, "params") and hasattr(
            snakemake.params, "show_legend_title"
        ):
            show_legend_title = snakemake.params.show_legend_title

        value = main(
            incsvfiles=snakemake.input,
            outplot=snakemake.output,
            legend=show_legend,
            legend_title=show_legend_title,
        )
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 2
        ), f"usage: python plot_seed42_histo.py in.csv to.csv <last|best>"
        inputs = sys.argv[1:3] if len(sys.argv) > 1 else None
        output = sys.argv[3]
        value = main(
            datacsv=inputs,
            outplot=output,
        )

    sys.exit(value)
