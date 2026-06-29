import numpy as np
from tqdm import tqdm

from xrayrecon.cubecorrections.emptyslices import trimEmptySlices
from xrayrecon.cubecorrections.rotation import getOptimalRotation, rotateCube


def performAllCorrections(cube: np.ndarray):
    # -----
    # EXTRANEOUS EXTRA DIMENSIONS
    # -----
    if cube.ndim > 3:
        cube = cube[..., 0]

    # -----
    # ROTATION
    # -----
    for ax in tqdm(range(3)):
        angle = getOptimalRotation(cube, ax)
        cube = rotateCube(cube, angle, ax)

    # -----
    # EMPTY / REDUNDANT SLICES
    # -----
    cube = trimEmptySlices(cube)
    return cube
