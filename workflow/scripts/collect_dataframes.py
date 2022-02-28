from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def concat_frames(listoffilenames):

    frames = []

    for fname in listoffilenames:
        frame = pd.read_csv(fname)

        fname_ = Path(fname)
        parts = list(fname_.parts)

        ## add filename parts 

        frames.append(frame)

    value = pd.concat(frames)

    return value


def main(output, write_header=False, filenames):

    


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        value = main(filenames=snakemake.input, output=opath, write_header=True)
        sys.exit(value)
    else:
        assert len(sys.argv)>3, f"usage: python collect_dataframe.py dst.csv in1.csv in2.csv ..."
        output = sys.argv[1] if len(sys.argv) > 1 else None
        inputs = sys.argv[2:] 
        value = main(filenames=inputs, output=outpath, write_header=True)

    sys.exit(value)
