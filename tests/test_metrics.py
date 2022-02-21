from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from timm.utils import accuracy
import torch
from scripts.metrics import compute_metric

TESTFOLDER = Path(__file__).parent
CSVPATH = Path(TESTFOLDER.absolute()) / "accuracies_input.csv"
# results/vit_small/seed42/fold-02/after80/topk_ids.csv


@pytest.fixture
def testcsv():

    return CSVPATH


@pytest.fixture
def idf():

    value = pd.read_csv(CSVPATH)
    return value


def test_fixture_csv_exists(testcsv):

    assert testcsv.exists()
    assert testcsv.stat().st_size > 0


def test_fixture_df_exists(idf):

    assert hasattr(idf, "shape")
    assert len(idf.shape) >= 2
    assert idf.shape == (670, 12)


def test_sklearn_accuracy(idf):

    y_true = idf.gtlabel
    y_pred = idf.topk0

    assert y_true.dtype == np.int64
    assert y_pred.dtype == np.int64

    acc = accuracy_score(y_true, y_pred)

    assert acc > 0
    assert acc > 0.5
    assert acc < 0.8
    assert acc > 0.7
    assert np.allclose(acc, 0.7284, atol=1e-3)

    y_pred_ = idf.topk1
    acc_ = accuracy_score(y_true, y_pred_)

    assert acc_ > 0
    assert acc_ > 0.1
    assert acc_ < 0.5
    assert acc > acc_


def test_metrics(testcsv):

    acc, nrows = compute_metric(testcsv)

    assert acc > 0
    assert acc > 0.5
    assert acc < 0.8
    assert acc > 0.7
    assert np.allclose(acc, 0.7284, atol=1e-3)
