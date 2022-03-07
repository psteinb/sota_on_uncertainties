from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *


def series_to_percentiles(series, percs=np.linspace(0, 1.0, 21)):

    print("series_to_percentiles: input ", type(series), series.shape)

    value = series.quantile(percs)
    midx = pd.Index(data=percs, name="percs")
    value = value.reindex(midx)
    print("series_to_percentiles: output ", type(value), value.shape, value)
    # need to change index name to percs
    return value


def sample_normal_to_percentiles(series, percs=np.linspace(0, 1.0, 21)):

    mn = series.mean()
    std = series.std()
    nsamples = len(series)

    # seed?
    samples = np.random.normal(mn, std, nsamples)
    print(series.index, series.index.values)
    midx = pd.Index(data=percs, name="percs")
    value = pd.Series(np.percentile(samples, percs), index=midx)
    print("sample_normal_to_percentiles: output ", type(value), value.shape, value)
    return value


def main(datacsv, destination, filter_by_cat="last", filter_by_seed=42):

    fulldf = pd.read_csv(datacsv)

    rmask1 = fulldf.srccategory.str.contains(filter_by_cat)
    print(f"{filter_by_cat}? {len(fulldf)} -> {rmask1.sum()}")

    rmask2 = rmask1
    if int(filter_by_seed) >= 0:
        rmask2 = fulldf["seed"] == int(filter_by_seed)
        print(f"{filter_by_seed}? {len(fulldf)} -> {rmask2.sum()}")

    mask = rmask1 & rmask2
    print(f"total? {len(fulldf)} -> {mask.sum()} ({mask.shape})")

    df = fulldf[mask]
    assert (
        len(df) > 0
    ), f"filtering by {filter_by_cat} & {filter_by_seed} removed all rows"
    df.arch_ = df.arch.astype("category")
    print(df.head())

    print(f"reduced shape {fulldf.shape} to {df.shape}")

    nbins = 15 if df.shape[0] <= 60 else 25

    percentiles = np.linspace(0, 1.0, 21)
    adf = df.groupby("arch")
    ssf = adf["val_accuracy"].agg(["mean", "std"])
    print(ssf)

    prcf = (
        adf["val_accuracy"]
        .apply(lambda x: series_to_percentiles(x))
        .rename("accuracy_percentiles")
    )
    nprcf = (
        adf["val_accuracy"]
        .apply(lambda x: sample_normal_to_percentiles(x))
        .rename("standard_percentiles")
    )

    fdf = pd.merge(prcf, nprcf, on=prcf.index.names, how="inner")

    df = fdf.reset_index()

    plt = (
        ggplot(df, aes(x="accuracy_percentiles", y="standard_percentiles"))
        + geom_point()
        # + ylim(0.7, 1.0)
        # + xlim(0.7, 1.0)
        + facet_wrap("arch", scales="free")
        + geom_abline(intercept=0, slope=1.0, color="red", linetype="dashed")
        + xlab("sample percentile")
        + ylab("theoretical percentiles")
        + theme_light()
        + theme(panel_spacing_x=0.7)
    )

    ggsave(plt, str(destination), width=10, height=2)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        value = main(
            datacsv=Path(snakemake.input[0]),
            destination=snakemake.output,
            filter_by_cat=snakemake.wildcards.chkpt,
            filter_by_seed=snakemake.wildcards.seedval,
        )
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 2
        ), f"usage: python plot_seed42_histo.py in.csv to.csv <last|best>"
        inputs = sys.argv[1] if len(sys.argv) > 1 else None
        output = sys.argv[2]
        chkpt = sys.argv[3] if len(sys.argv) > 2 else "last"
        seedval = sys.argv[4] if len(sys.argv) > 3 else 42
        value = main(
            datacsv=inputs,
            destination=output,
            filter_by_cat=chkpt,
            filter_by_seed=seedval,
        )

    sys.exit(value)
