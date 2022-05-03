from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from collections import Counter
import numpy as np


def X_y(path, suffix="JPEG"):
    """generate X and y from walking a <path_>, the path is expected
    to contain images that end on <suffix>, each parent folder of a file is
    considered a unique class identifyer
    """

    path_ = Path(
        path
    ).resolve()  # to prevent having symlinks through slow filesystems in the src path
    candidatefiles = sorted(list(path_.glob("**/*" + suffix)))

    parents = [f.parent.parts[-1] for f in candidatefiles]

    lb = LabelEncoder()
    y = lb.fit_transform(parents)

    X = np.arange(0, len(candidatefiles)).reshape((len(candidatefiles), 1))

    return X, y, candidatefiles, lb


def write_tables(inpath, outputdir=".", kfolds=20, seed=42):

    X, y, candidatefiles, lb = X_y(inpath)
    skf = StratifiedKFold(n_splits=kfolds)  # , random_state=seed
    opath = Path(outputdir)
    foldcnt = 0
    written = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]

        assert len(X_train.shape) >= 2
        assert len(X_test.shape) >= 2

        y_train, y_test = y[train_index], y[test_index]
        # create from-to filelist
        foldstem = f"fold-{foldcnt:02.0f}"
        folddir = Path(outputdir) / foldstem
        foldtable = Path(outputdir) / f"{foldstem}.table"

        lines = []

        for index in X_train[:, 0]:
            src = candidatefiles[index]
            dst = folddir / "train" / src.parts[-2] / src.parts[-1]
            lines.append(f"{src} {dst}")

        for index in X_test[:, 0]:
            src = candidatefiles[index]
            dst = folddir / "val" / src.parts[-2] / src.parts[-1]
            lines.append(f"{src} {dst}")

        foldtable.write_text("\n".join(lines))
        written.append(str(foldtable))
        foldcnt += 1

    return written


def main(inpath, outputdir=".", kfolds=20, seed=42):

    assert Path(outputdir).is_dir()
    assert Path(inpath).is_dir()

    tablefiles = write_tables(inpath, outputdir, kfolds, seed)

    if len(tablefiles) == kfolds:
        return 0  # success
    else:
        return 1


if __name__ == "__main__":
    value = 1

    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        ipath_ = Path(snakemake.input[0])
        ipath = ipath_ if not ipath_.is_file() else ipath_.parent
        # this if-clause is a workaround for snakemake
        if ".table" == opath.suffix:
            value = main(ipath, opath.parent)
        else:
            value = main(ipath, opath)
        sys.exit(value)
    else:
        inpath = sys.argv[1] if len(sys.argv) > 1 else None
        outpath = sys.argv[2] if len(sys.argv) > 2 else "."
        kfolds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

        value = main(inpath, outpath, kfolds, seed)

    sys.exit(value)
