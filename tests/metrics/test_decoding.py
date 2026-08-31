import numpy as np

from qldpc_fno.metrics.decoding import score_observable_predictions


def test_block_error_counts_any_wrong_observable() -> None:
    actual = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    predicted = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.uint8)
    metrics = score_observable_predictions(actual, predicted)
    assert metrics["shots"] == 3
    assert metrics["block_errors"] == 2
    assert metrics["block_error_rate"] == 2 / 3
    assert 0 <= metrics["block_error_rate_95ci_low"] < 2 / 3
    assert 2 / 3 < metrics["block_error_rate_95ci_high"] <= 1


def test_score_requires_equal_binary_arrays() -> None:
    actual = np.array([[0], [1]], dtype=np.uint8)
    predicted = np.array([[0, 1]], dtype=np.uint8)
    try:
        score_observable_predictions(actual, predicted)
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("accepted mismatched observable arrays")
