from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import *


def main(incsv, outcsv, colname="seed", colvalue=42, adddescription=True):

    fulldf = pd.read_csv(incsv)

    value = fulldf[fulldf[colname] == pd.to_numeric(colvalue)]
    if adddescription:
        value["description"] = f"{colname}={colvalue}"
    print(value.head())
    value.to_csv(str(outcsv), index=False)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        colname, colvalue = "seed", 42
        if hasattr(snakemake, "wildcards"):
            colname = snakemake.wildcards.filtercol
            colvalue = snakemake.wildcards.filterval
            print("discovered wildcards in snakemake: ", snakemake.wildcards)
        value = main(
            incsv=Path(snakemake.input[0]),
            outcsv=snakemake.output,
            colname=colname,
            colvalue=colvalue,
        )
        sys.exit(value)
    else:
        assert len(sys.argv) > 2, f"usage: python thisscript.py in.csv to.csv "
        inputs = sys.argv[1] if len(sys.argv) > 1 else None
        output = sys.argv[2] if len(sys.argv) > 2 else "to.csv"
        value = main(incsv=inputs, outcsv=output, groupbycols=groupbycols)

    sys.exit(value)
