import warnings

import numpy as np
import pytest

from xrayrecon.gui.table import PandasModel
from xrayrecon.layerseg import layerInfoFromScores, templateMatchScores


@pytest.mark.unit
def test_table_accepts_slice_arrays_without_dtype_warnings(qapp):
    model = PandasModel(
        {"View": False, "#": -1, "Layer Name": "", "Slices": -1},
        indexCol="#",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", RuntimeWarning)
        model.addDfRows(
            {
                "Layer Name": ["Top", "Bottom"],
                "Slices": [np.arange(3), np.arange(3, 6)],
            }
        )
        model.addDfRows({"Layer Name": ["Extra"], "Slices": [np.arange(6, 8)]})

    assert model.dataDf.index.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(model.dataDf.iloc[2]["Slices"], np.arange(6, 8))


@pytest.mark.unit
def test_flat_layer_scores_produce_nonempty_finite_regions():
    scores = np.ones(8, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        info = layerInfoFromScores(scores)

    assert info["Layer Name"]
    assert all(len(slices) > 0 for slices in info["Slices"])
    assert all(np.all(np.isfinite(slices)) for slices in info["Slices"])


@pytest.mark.unit
def test_constant_volume_template_scores_are_finite():
    volume = np.ones((12, 12, 6), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scores = templateMatchScores(volume)

    assert np.all(np.isfinite(scores))
