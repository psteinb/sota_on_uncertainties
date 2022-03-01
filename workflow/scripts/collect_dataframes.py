from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def concat_frames(listoffilenames, listofmd=[]):

    frames = []
    if not listofmd:
        for idx, fname in enumerate(listoffilenames):
            fname_ = Path(fname)
            parts = list(fname_.parts)
            partkeys = [f"part{cnt:02.0f}" for cnt in range(len(parts))]
            metadict = {k: [v] for (k, v) in zip(partkeys, parts)}
            listofmd.append(metadict)

    for idx, fname in enumerate(listoffilenames):
        frame = pd.read_csv(fname)
        metadict = listofmd[idx]

        metadf = pd.DataFrame.from_dict(metadict)

        frame_nrows = len(frame)
        frame = pd.concat([frame, metadf], axis=1)
        ## add filename parts
        assert frame_nrows == len(frame)
        frames.append(frame)

    value = pd.concat(frames)

    return value


def main(output, filenames, write_header=False):

    return 1


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        value = main(filenames=snakemake.input, output=opath, write_header=True)
        sys.exit(value)
    else:
        assert (
            len(sys.argv) > 3
        ), f"usage: python collect_dataframe.py dst.csv in1.csv in2.csv ..."
        output = sys.argv[1] if len(sys.argv) > 1 else None
        inputs = sys.argv[2:]
        value = main(filenames=inputs, output=outpath, write_header=True)

    sys.exit(value)
