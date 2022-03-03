from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *
from scipy.stats import norm


def main(incsv, outplot, add_approximation=False):

    df = pd.read_csv(fname)
    df["accuracy"] = df["accuracy_percent"] / 100.0
    df["robustness"] = df["robustness_percent"] / 100.0

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

    rdf = df[df.label.str.contains("ResNet")]
    print("resnets only")
    print(rdf)

    plt = (
        ggplot(
            df,
            aes(y="accuracy", x="robustness"),
        )
        # + geom_errorbar(
        #     aes(ymin="accuracy_min", ymax="accuracy_max"),
        #     position=position_dodge(width=0.75),
        #     width=0.2,
        # )
        + geom_point(
            data=rdf, mapping=aes(y="accuracy", x="robustness"), color="violet"
        )
        + ylab("accuracy")
        + xlab("robustness")
        + theme_light()
        # + theme(legend_position="top")
        + coord_flip()
    )

    ggsave(plt, str(outplot), width=6, height=2)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])

        show_approx = True
        if hasattr(snakemake, "params") and hasattr(snakemake.params, "show_approx"):
            show_approx = snakemake.params.show_approx

        value = main(
            incsv=snakemake.input,
            outplot=snakemake.output,
            add_approximation=show_approx,
        )
        sys.exit(value)
    else:
        assert len(sys.argv) > 2, f"usage: python thisfilename.py in.csv to.csv"
        inputs = sys.argv[1:3] if len(sys.argv) > 1 else None
        output = sys.argv[3]
        value = main(
            datacsv=inputs,
            outplot=output,
        )

    sys.exit(value)
