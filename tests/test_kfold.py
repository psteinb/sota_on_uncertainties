from pathlib import Path
import pytest
from tempfile import mkdtemp
from scripts.kfold import X_y, write_tables, main
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter


@pytest.fixture
def tmpdir():

    temppath = Path(mkdtemp())
    sdir1 = temppath / "n01440764"
    sdir1.mkdir()
    sdir2 = temppath / "n03888257"
    sdir2.mkdir()

    files = [
        (sdir1 / "foo.JPEG"),
        (sdir1 / "baz.JPEG"),
        (sdir2 / "boo.JPEG"),
        (sdir2 / "bar.JPEG"),
    ]
    [f.touch() for f in files]

    yield temppath

    [f.unlink() for f in files]
    sdir1.rmdir()
    sdir2.rmdir()
    temppath.rmdir()


def test_fixture_exists(tmpdir):

    assert tmpdir.exists()
    assert (tmpdir / "n03888257").exists()
    assert (tmpdir / "n03888257" / "bar.JPEG").exists()


def test_uniques(tmpdir):

    assert tmpdir.exists()
    candidatefiles = Path(tmpdir).glob("**/*JPEG")
    parents = [f.parent.parts[-1] for f in candidatefiles]
    uparents = [c for c in Counter(parents)]
    assert len(uparents) == 2
    assert "n03888257" in uparents
    assert "n01440764" in uparents


def test_X_y_nonnil(tmpdir):

    obsX, obsy, obsflist, obslb = X_y(tmpdir)

    assert isinstance(obslb, LabelEncoder)
    assert isinstance(obsy, np.ndarray)
    assert obsy.dtype == np.int64
    assert (obsy == np.array([0, 0, 1, 1], dtype=np.int64)).all()

    expy = np.ones((3,), dtype=obsy.dtype)
    expy[1] = 0
    expy_ = obslb.inverse_transform(expy)

    assert expy_[0] != expy_[1]
    assert expy_[0] == expy_[-1]
    assert expy_[0] == "n03888257"
    assert expy_[1] == "n01440764"

    assert type(obsX) != type(None)
    assert obsX.shape == (4, 1)
    assert obsX[2, 0] == 2


def test_write_tables(tmpdir):

    opath = Path(mkdtemp())

    rvalue = write_tables(tmpdir, outputdir=opath, kfolds=2)

    assert len(rvalue) == 2

    tfiles = list(opath.glob("*.table"))
    for rf in rvalue:
        assert Path(rf) in tfiles, f"{rf} not in tfiles"

    contents = opath.glob("*")
    [c.unlink() for c in contents]
    opath.rmdir()
