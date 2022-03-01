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
        assert isinstance(
            metadict, dict
        ), f"metadata for {fname} is not a dictionary\n{metadict}"
        metadf = pd.DataFrame.from_dict(metadict)

        frame_nrows = len(frame)
        frame = pd.concat([frame, metadf], axis=1)
        ## add filename parts
        assert frame_nrows == len(frame)
        frames.append(frame)

    value = pd.concat(frames)

    return value


def obtain_metadata(filenames):

    md = []
    for (idx, fname) in enumerate(filenames):
        fname_ = Path(fname)
        all_parts = list(fname_.parts)
        # cut off all parts before results
        # assumes /some/irrelevant/path/results/resnet50/seed42/fold-03/after80/last_metrics.csv
        resultsi = all_parts.index("results")
        parts = all_parts[resultsi + 1 :]
        mddict = {"arch": [parts[0]]}
        if "seed" in parts[1].lower():
            mddict["seed"] = [int(parts[1].lower().lstrip("seed"))]
        if "fold-" in parts[2].lower():
            mddict["fold"] = [int(parts[2].lower().lstrip("fold-"))]
        if "after" in parts[3].lower():
            mddict["epochs"] = [int(parts[3].lower().lstrip("after"))]
        if "_" in parts[-1]:
            mddict["srccategory"] = [parts[-1].split("_")]
        mddict["srcfile"] = [str(parts[-1]).lower()]

        md.append(mddict)
    return md


def main(output, filenames, write_header=False, metadata_from_filenames=False):

    md = []
    if metadata_from_filenames:
        md = obtain_metadata(filenames)

    value = concat_frames(filenames, md)

    value.to_csv(output, index=False)

    if Path(output).stat().st_size > 0:
        print(f"wrote dataframe of shape {value.shape} to {output}")
        return 0
    else:
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
        value = main(
            filenames=inputs,
            output=outpath,
            write_header=True,
            metadata_from_filenames=False,
        )

    sys.exit(value)
