from pathlib import Path

import pandas as pd
from plotnine import *


def main(incsvfiles, outplot, filter_by_seed=42, legend=True):

    dfs = []
    for fname in incsvfiles:
        cat = "best"
        if "last" in fname:
            cat = "last"
        df = pd.read_csv(fname)
        df["checkpoint"] = cat
        dfs.append(df)

    fulldf = pd.concat(dfs)
    fulldf["val_accuracy_mean_max"] = (
        fulldf["val_accuracy_mean"] + fulldf["val_accuracy_std"] / 2.0
    )
    fulldf["val_accuracy_mean_min"] = (
        fulldf["val_accuracy_mean"] - fulldf["val_accuracy_std"] / 2.0
    )

    df = fulldf.copy()
    if int(filter_by_seed) > -1:
        mask = fulldf.seed == int(filter_by_seed)
        df = fulldf[mask]
        print(
            f"reduced shape {fulldf.shape} to {df.shape} (based on seed {filter_by_seed})"
        )
    print(f"using dataframe of shape {fulldf.shape} from {incsvfiles}")
    plt = (
        ggplot(
            df,
            aes(x="arch", y="val_accuracy_mean", color="checkpoint"),
        )
        + geom_point(position=position_dodge(width=0.3))
        + geom_errorbar(
            aes(ymin="val_accuracy_mean_min", ymax="val_accuracy_mean_max"),
            position=position_dodge(width=0.3),
            width=0.2,
        )
        + xlab("architecture")
        + ylab("accuracy")
        + theme_light()
        + theme(panel_spacing_x=0.3, legend_position="top")
        + guides(
            color=guide_legend(
                nrow=1,
                direction="horizontal",
                label_position="right",
                # title_vjust=0.5
                # title_separation=0.3,
                # label_separation=0.2,
            )
        )
        + coord_flip()
    )

    if not legend:
        print("removing legend")
        # http://www.cookbook-r.com/Graphs/Legends_(ggplot2)/
        plt += theme(legend_position="none")

    ggsave(plt, str(outplot), width=6, height=2)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        filter_by_seed_ = -42  # default is to NOT filter by seed
        if hasattr(snakemake, "wildcards") and hasattr(snakemake.wildcards, "seedval"):
            filter_by_seed_ = snakemake.wildcards.seedval

        show_legend = True
        if hasattr(snakemake, "params") and hasattr(snakemake.params, "show_legend"):
            show_legend = snakemake.params.show_legend
        value = main(
            incsvfiles=snakemake.input,
            outplot=snakemake.output,
            filter_by_seed=filter_by_seed_,
            legend=show_legend,
        )
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 2
        ), f"usage: python plot_seed42_histo.py in.csv to.csv <last|best>"
        inputs = sys.argv[1:3] if len(sys.argv) > 1 else None
        output = sys.argv[3]
        seedval = sys.argv[4] if len(sys.argv) > 2 else 42
        value = main(
            datacsv=inputs,
            outplot=output,
            filter_by_seed=seedval,
        )

    sys.exit(value)
