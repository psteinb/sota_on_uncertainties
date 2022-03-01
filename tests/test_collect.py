from pathlib import Path
from pprint import pprint
from tempfile import mkstemp

import numpy as np
import pandas as pd
import pytest
from scripts.collect_dataframes import concat_frames, main, obtain_metadata

TESTFOLDER = Path(__file__).parent
RESULTSPATH = Path(TESTFOLDER.absolute()) / "results"
RESULTSCSV = list(RESULTSPATH.glob("**/last_metrics.csv"))


def test_csv_present():
    assert len(RESULTSCSV) == 3
    assert sum([item.stat().st_size for item in RESULTSCSV]) > 0


def test_concat_csvs():

    nexp = pd.read_csv(RESULTSCSV[0])

    assert nexp.shape[-1] == 3, f"{nexp.shape}"  # columns
    assert nexp.shape[0] == 1, f"{nexp.shape}"  # rows

    obs = concat_frames(RESULTSCSV)

    assert isinstance(obs, pd.DataFrame)
    assert obs.shape[0] == 3, f"{obs.shape}\n{obs.describe()}"
    assert len(obs) == 3, f"{obs.shape}"
    assert obs.shape[-1] > nexp.shape[-1]


def test_concat_csvs_with_factors():

    obs = concat_frames(RESULTSCSV)
    # pprint(list(RESULTSCSV[0].parts))
    obs.part08_ = obs.part08.astype("category")

    assert len(obs.part08_.cat.categories) == 3, f"{obs.part08_}"


def test_metadata_from_filenames():

    obs = obtain_metadata(RESULTSCSV)

    assert len(obs) == 3
    assert isinstance(obs[0], dict)
    assert len(list(obs[0].keys())) == 6
    assert obs[0]["arch"] == ["vit_small"]
    assert obs[0]["seed"] == [42]
    assert obs[0]["fold"] == [3]


def test_main():

    tempo = mkstemp(suffix=".csv")
    outf = Path(tempo[-1])
    assert not isinstance(outf, tuple)

    obs = main(outf, RESULTSCSV, write_header=True, metadata_from_filenames=True)
    assert obs == 0

    rld = pd.read_csv(outf)
    assert "arch" in rld.columns
    assert rld.arch.str.contains("resnet50").any()
    assert "val_accuracy" in rld.columns
    assert "nsamples" in rld.columns
    assert (rld.nsamples > 0).all()

    outf.unlink()
