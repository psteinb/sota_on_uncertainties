from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *


def main(incsv, outcsv, groupbycols=["arch", "seed"]):

    fulldf = pd.read_csv(incsv)

    gdf = fulldf.groupby(groupbycols)  # should collapse to singular values

    print("aggregating mean/std val_accuracy based on col groups", groupbycols)
    print(f"grouped df {len(gdf)} compared to full df {len(fulldf)}")
    value_accuracy = (
        gdf["val_accuracy"]
        .agg(["mean", "std", "size"])
        .rename(
            columns={
                "size": "foldcnt",
                "mean": "val_accuracy_mean",
                "std": "val_accuracy_std",
            }
        )
        .reset_index()
    )

    print("aggregating mean/std val_f1score based on col groups", groupbycols)
    value_f1score = (
        gdf["val_f1score"]
        .agg(["mean", "std", "size"])
        .rename(
            columns={
                "size": "foldcnt",
                "mean": "val_f1score_mean",
                "std": "val_f1score_std",
            }
        )
        .reset_index()
    )

    value = pd.merge(
        value_accuracy,
        value_f1score,
        on=groupbycols,
        how="inner",
        validate="one_to_one",
    )
    # during the analysis, 4 folds yielded 669 samples whereas the remaining 16 670
    # we choose the majority vote here
    value["nsamples"] = fulldf.nsamples.max()

    print(value.head())
    value.to_csv(str(outcsv), index=False)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        groupbycols = ["arch", "seed"]
        if hasattr(snakemake, "params"):
            groupbycols = snakemake.params.groupbycols
            print("discovered params in snakemake: ", snakemake.params.groupbycols)
        value = main(
            incsv=Path(snakemake.input[0]),
            outcsv=snakemake.output,
            groupbycols=groupbycols,
        )
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 2
        ), f"usage: python mean_std_across_folds.py in.csv to.csv <colon seperated list of columns to groubpy>"
        inputs = sys.argv[1] if len(sys.argv) > 1 else None
        output = sys.argv[2] if len(sys.argv) > 2 else "to.csv"
        groupbycols = sys.argv[3].split(":") if len(sys.argv) > 3 else ["arch", "seed"]
        value = main(incsv=inputs, outcsv=output, groupbycols=groupbycols)

    sys.exit(value)
