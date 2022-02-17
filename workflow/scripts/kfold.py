from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from collections import Counter
import numpy as np


def X_y(path, suffix="JPEG"):
    """generate X and y from walking a path, the path is expected
    to contain images that end of suffix, each parent folder of a file is
    considered a unique class identifyer
    """

    candidatefiles = sorted(list(Path(path).glob("**/*" + suffix)))

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
        folddir = opath / f"fold-{foldcnt:02.0f}"
        foldtable = opath / f"fold-{foldcnt:02.0f}.table"
        ofile = open(foldtable, "w")
        for item in X_train:
            index = item[0]
            src = candidatefiles[index]
            dst = folddir / "train" / src.parts[-2] / src.parts[-1]
            ofile.write(f"{src} {dst}\n")

        for item in X_test:
            index = item[0]
            src = candidatefiles[index]
            dst = folddir / "val" / src.parts[-2] / src.parts[-1]
            ofile.write(f"{src} {dst}\n")

        ofile.flush()
        ofile.close()
        written.append(str(foldtable))
        foldcnt += 1

    return written


def main(inpath, outputdir=".", kfolds=20, seed=42):

    tablefiles = write_tables(inpath, outputdir, kfolds, seed)

    if len(tablefiles) == kfolds:
        return 0
    else:
        return 1


if __name__ == "__main__":
    main()
