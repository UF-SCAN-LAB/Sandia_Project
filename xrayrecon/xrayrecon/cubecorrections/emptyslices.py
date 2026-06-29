import numpy as np


def trimEmptySlices(cube: np.ndarray):
    trimmedCube = cube.copy()
    for axisorder in [(1, 2, 0), (1, 2, 0), (1, 2, 0)]:
        trimmedCube = trimmedCube.transpose(axisorder)
        toRemove = []
        for ii in range(trimmedCube.shape[0]):
            allEmpty = not np.any(trimmedCube[ii, :, :])
            if allEmpty:
                toRemove.append(ii)
        trimmedCube = np.delete(trimmedCube, toRemove, 0)
    return trimmedCube
