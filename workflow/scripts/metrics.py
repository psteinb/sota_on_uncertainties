import sys
from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metric(tablefile, nrows=-1, metric_fn=accuracy_score, verbose=False):

    tfile = Path(tablefile)
    assert tfile.exists(), f"tablefile {tfile} does not exist"
    assert tfile.is_file()

    df = pd.read_csv(tfile)
    if verbose:
        print(df.describe())

    if nrows > 0:
        df = df.sample(n=nrows)

    y_true = df.gtlabel
    y_pred = df.topk0

    value = metric_fn(y_true, y_pred)

    return value, len(y_true)


def main(tablefile, output, metadata={}, write_header=False, verbose=False):

    ipath = Path(tablefile)
    keys = sorted(list(metadata.keys()))
    opath = Path(output)

    acc, nrows = compute_metric(ipath)
    f1_score_ = partial(f1_score, average="weighted")
    f1, _ = compute_metric(ipath, metric_fn=f1_score_)

    vals = [metadata[k] for k in keys]
    lines = []
    if write_header:
        header = ""
        if keys:
            header = ",".join(keys)
            header += ","
        header += "val_accuracy"
        header += ","
        header += "val_f1score"
        header += ","
        header += "nsamples"
        lines.append(header)

    line = ""
    if keys:
        line = ",".join(vals)
        line += ","
    line += str(acc)
    line += ","
    line += str(f1)
    line += ","
    line += str(nrows)
    lines.append(line)

    payload = "\n".join(lines)
    opath.write_text(payload)

    if verbose:
        print(payload)


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        metadata = {}
        opath = Path(snakemake.output[0])
        value = main(snakemake.input[0], opath, metadata, write_header=True)
        sys.exit(value)
    else:
        inpath = sys.argv[1] if len(sys.argv) > 1 else None
        outpath = sys.argv[2] if len(sys.argv) > 2 else "."
        md = {}
        for item in sys.argv[3:]:
            key, val = item.split(":")
            md[key] = val
        value = main(inpath, outpath, metadata=md, write_header=True)

    sys.exit(value)
