import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import norm


def main(
    incsv,
    outplot,
    add_approximation=True,
    extend_limits_on_errors=True,
    confidence_scale=2,
):

    df = pd.read_csv(incsv)

    print(df.columns)
    print(df)
    df["accuracy"] = df["accuracy_percent"] / 100.0
    df["label"] = df["label"].astype("category")

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

    z_std = norm.ppf(1 - ((1 - integral_1sigma) / 2.0))  # 1 sigma confidence interval
    print(
        f"68.2% confidence interval: integral_range = {integral_1sigma}, correction {z_std}"
    )

    ## apply formula for approximate uncertainty
    inv_samples = 1.0 / 10_000  # for accuracy CIFAR100 was used
    df["accuracy_std"] = z_std * np.sqrt(
        inv_samples * df["accuracy"] * (1 - df["accuracy"])
    )

    df["accuracy_2std"] = z_std_2sigma * np.sqrt(
        inv_samples * df["accuracy"] * (1 - df["accuracy"])
    )

    df["accuracy_max"] = df["accuracy"] + df["accuracy_std"]
    df["accuracy_min"] = df["accuracy"] - df["accuracy_std"]
    df["accuracy_2max"] = df["accuracy"] + df["accuracy_2std"]
    df["accuracy_2min"] = df["accuracy"] - df["accuracy_2std"]
    print(f"using dataframe of {df.shape}")
    print(df.head())

    plt = (
        ggplot(
            df,
            aes(y="accuracy", x="msa_number", color=df["color"], label="label"),
        )
        # thanks to https://stackoverflow.com/questions/43129280/color-points-with-the-color-as-a-column-in-ggplot2
        + scale_color_identity()
        + ylab("accuracy")
        + xlab("MSA")
        + theme_light()
    )

    darkbluedf = df[df["linecolor"].str.contains("darkblue")]
    plt += geom_line(
        data=darkbluedf, mapping=aes(y="accuracy", x="msa_number"), color="darkblue"
    )

    purpledf = df[df["linecolor"].str.contains("purple")]
    plt += geom_line(
        data=purpledf, mapping=aes(y="accuracy", x="msa_number"), color="purple"
    )

    reddf = df[df["linecolor"].str.contains("red")]
    plt += geom_line(data=reddf, mapping=aes(y="accuracy", x="msa_number"), color="red")

    orangedf = df[df["linecolor"].str.contains("orange")]
    plt += geom_line(
        data=orangedf, mapping=aes(y="accuracy", x="msa_number"), color="orange"
    )

    if add_approximation:
        if confidence_scale > 1:
            plt += geom_errorbar(
                aes(ymin="accuracy_2min", ymax="accuracy_2max"),
                # position=position_dodge(width=0.75),
                width=0.0,
                size=1.5,
                color="lightgrey",
            )
        plt += geom_errorbar(
            aes(ymin="accuracy_min", ymax="accuracy_max"),
            # position=position_dodge(width=0.75),
            width=0.05,
        )

    # do the actual plotting last
    plt = plt + geom_point(size=3) + geom_text(nudge_x=0.25, nudge_y=0.001)

    if extend_limits_on_errors:
        maxy = (
            df["accuracy_2max"].max()
            if confidence_scale > 1
            else df["accuracy_max"].max()
        )
        miny = (
            df["accuracy_2min"].min()
            if confidence_scale > 1
            else df["accuracy_min"].min()
        )
        print(df.describe())
        print(f"plot limits: {0.9995 * miny, 1.0005 * maxy}")
        plt += ylim(0.9995 * miny, 1.0005 * maxy)

    ggsave(plt, str(outplot), width=5, height=4)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        ipath = Path(snakemake.input[0])
        opath = Path(snakemake.output[0])

        show_approx = True
        if hasattr(snakemake, "params") and hasattr(
            snakemake.params, "add_approximation"
        ):
            show_approx = snakemake.params.add_approximation

        value = main(
            incsv=ipath,
            outplot=opath,
            add_approximation=show_approx,
        )
        sys.exit(value)
    else:
        assert len(sys.argv) > 2, f"usage: python thisfilename.py in.csv to.csv"
        inputs = sys.argv[1] if len(sys.argv) > 1 else None
        output = sys.argv[2]
        value = main(
            incsv=inputs,
            outplot=output,
        )

    sys.exit(value)
