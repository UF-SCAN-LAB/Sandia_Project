import numpy as np

from xrayrecon.gui.view3d import PCBViewer3D


def test_downsample_data_preserves_high_bit_depth_range():
    viewer = PCBViewer3D.__new__(PCBViewer3D)
    viewer.downsampFactor = 1
    data = np.arange(40 * 48 * 24, dtype=np.uint16).reshape(40, 48, 24) * 2

    downsampled, highlights = PCBViewer3D.downsampleData(
        viewer, data, np.array([0, 5, 23]), factor=3
    )

    assert downsampled.shape == (13, 16, 8)
    assert downsampled.dtype == np.float32
    assert downsampled.max() > np.iinfo(np.uint8).max
    assert highlights.tolist() == [0, 1, 7]
