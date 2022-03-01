from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *


def main(incsv, outcsv):

    fulldf = pd.read_csv(incsv)

    gdf = fulldf.groupby(["arch", "seed"])  # should collapse to singular values

    print("grouped", gdf.size())
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
        on=["arch", "seed"],
        how="inner",
        validate="one_to_one",
    )
    # during the analysis, 4 folds yielded 669 samples whereas the remaining 16 670
    # we choose the majority vote here
    value["nsamples"] = 670

    print(value)
    value.to_csv(str(outcsv), index=False)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        value = main(incsv=Path(snakemake.input[0]), outcsv=snakemake.output)
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 2
        ), f"usage: python plot_seed42_histo.py in.csv to.csv <last|best>"
        inputs = sys.argv[1] if len(sys.argv) > 1 else None
        output = sys.argv[2]

        value = main(incsv=inputs, outcsv=output)

    sys.exit(value)
