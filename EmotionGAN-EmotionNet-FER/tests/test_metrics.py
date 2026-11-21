import numpy as np
from emotionfer.evaluation import compute_metrics
from emotionfer.gan import synthetic_budget


def test_paper_illustrative_budget():
    # Paper illustrative 80/20 partition (NOT the corrected 72/8/20 training counts).
    counts = [768, 88, 816, 1460, 912, 640, 972]  # native FER-2013 class-ID order
    labels = np.concatenate([np.repeat(i, n) for i, n in enumerate(counts)])
    assert synthetic_budget(labels).tolist() == [692,1372,644,0,548,820,488]
    assert int(synthetic_budget(labels).sum()) == 4564
    assert len(labels) + int(synthetic_budget(labels).sum()) == 10220


def test_perfect_metrics_and_confusion_matrix():
    truth = np.array([0,1,2,3,4,5,6]*2)
    proba = np.eye(7)[truth]
    m = compute_metrics(truth, proba)
    assert m["accuracy"] == 1.
    assert m["f1_macro"] == 1.
    assert m["paper_harmonic_macro_f1"] == 1.
    assert np.trace(m["confusion_matrix"]) == len(truth)


def test_paper_macro_f1_not_necessarily_standard_macro_f1():
    truth = np.repeat(np.arange(7), 4)
    pred = truth.copy()
    pred[4:8] = 0
    proba = np.full((len(truth), 7), .001, np.float64)
    proba[np.arange(len(truth)), pred] = .994
    proba /= proba.sum(axis=1, keepdims=True)
    m = compute_metrics(truth, proba)
    assert m["accuracy"] < 1.
    assert abs(m["paper_harmonic_macro_f1"] - m["f1_macro"]) > 1e-5
    assert m["per_class"]["Disgust"]["recall"] == 0.


def test_metric_rejects_invalid_softmax():
    import pytest
    with pytest.raises(ValueError):
        compute_metrics(np.arange(7), np.zeros((7,7)))
