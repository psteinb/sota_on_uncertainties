from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def compute_metric(tablefile, nrows=-1, metric_fn=accuracy_score):

    tfile = Path(tablefile)
    assert tfile.exists(), f"tablefile {tfile} does not exist"
    assert tfile.is_file()

    df = pd.read_csv(tfile)
    print(df.describe())

    if nrows > 0:
        df = df.sample(n=nrows)

    y_true = df.gtlabel
    y_pred = df.topk0

    value = metric_fn(y_true, y_pred)

    return value, len(y_true)


def main(tablefile, output, metadata={}, write_header=False):

    ipath = Path(tablefile)
    keys = sorted(list(metadata.keys()))
    opath = Path(output)

    acc, nrows = compute_metric(ipath)
    f1, _ = compute_metric(ipath, metric_fn=f1_score)

    vals = [metadata[k] for k in keys]

    if write_header:
        header = ",".join(keys)
        header += ","
        header += "val_accuracy"
        header += ","
        header += "val_f1score"
        header += ","
        header += "nsamples"
        opath.write_text(header)

    line = ",".join(fnamvals)
    line += ","
    line += str(acc)
    line += ","
    line += str(f1)
    line += ","
    line += str(nrows)
    opath.write_text(line)
    print(line)


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
