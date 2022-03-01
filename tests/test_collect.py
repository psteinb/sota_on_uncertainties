from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import pytest
from scripts.collect_dataframes import concat_frames

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
    pprint(list(RESULTSCSV[0].parts))
    obs.part08_ = obs.part08.astype("category")
    pprint(obs.part08_.cat.categories.values)
    assert len(obs.part08_.cat.categories) == 3, f"{obs.part08_}"
