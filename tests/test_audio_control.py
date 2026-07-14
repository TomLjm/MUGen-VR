import numpy as np
import pytest

from mugen.evaluation.audio_control import resample_series


def test_resample_series_preserves_endpoints():
    result = resample_series([0.0, 1.0], 5)
    assert result.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_resample_series_rejects_empty_input():
    with pytest.raises(ValueError):
        resample_series(np.asarray([]), 3)
