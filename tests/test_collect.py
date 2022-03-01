from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.collect_dataframes import concat_frames

TESTFOLDER = Path(__file__).parent
RESULTSPATH = Path(TESTFOLDER.absolute()) / "results"
RESULTSCSV = list(RESULTSPATH.glob("**/*csv"))


def test_csv_present():
    assert len(RESULTSCSV) == 3
    assert sum([item.stat().st_size for item in RESULTSCSV]) > 0


def test_concat_csvs():

    obs = concat_frames(RESULTSCSV)

    assert isinstance(obs, pd.DataFrame)
    assert obs.shape[-1] == 3, f"{obs.shape}"
