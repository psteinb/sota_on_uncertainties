import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import norm


def main(incsv, outplot, add_approximation=False):

    df = pd.read_csv(incsv)

    print(df.columns)
    print(df)
    df["accuracy"] = df["accuracy_percent"] / 100.0
    df["label"] = df["label"].astype("category")

    z_std95 = norm.ppf(
        1 - ((1 - 0.95) / 2.0)
    )  # 95% confidence interval = 1.959, almost 2 sigma
    z_std = norm.ppf(1 - ((1 - 0.68) / 2.0))  # 1 sigma confidence interval
    inv_samples = 1.0 / 10_000  # for accuracy CIFAR100 was used
    df["accuracy_std"] = z_std * np.sqrt(
        inv_samples * df["accuracy"] * (1 - df["accuracy"])
    )

    df["accuracy_max"] = df["accuracy"] + df["accuracy_std"]
    df["accuracy_min"] = df["accuracy"] - df["accuracy_std"]
    print(f"using dataframe of {df.shape}")
    print(df.head())

    plt = (
        ggplot(
            df,
            aes(y="accuracy", x="msa_number", color=df["color"], label="label"),
        )
        # thanks to https://stackoverflow.com/questions/43129280/color-points-with-the-color-as-a-column-in-ggplot2
        + scale_color_identity()
        + geom_point(size=3)  # , position=position_dodge(width=0.75)
        + geom_text(nudge_x=0.25, nudge_y=0.001)
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
        plt += geom_errorbar(
            aes(ymin="accuracy_min", ymax="accuracy_max"),
            # position=position_dodge(width=0.75),
            width=0.05,
        )

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
