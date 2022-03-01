from pathlib import Path

import pandas as pd
from plotnine import *


def main(datacsv, destination, filter_by_cat="last", filter_by_seed=42):

    fulldf = pd.read_csv(datacsv)

    rmask1 = fulldf.srccategory.str.contains(filter_by_cat)
    print(f"{filter_by_cat}? {len(fulldf)} -> {rmask1.sum()}")

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

    plt = (
        ggplot(df, aes(x="val_accuracy"))
        + geom_histogram(bins=15)
        + facet_wrap("arch", scales="free_y")
        + xlab("accuracy")
        + theme_light()
        + theme(panel_spacing_x=0.3)
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
