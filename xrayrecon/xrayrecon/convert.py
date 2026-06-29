import glob

import cv2
import numpy as np
from fire import Fire
from qtextras import multiprocessApply


def stackData(folder: str, ext="tif", out="data.npy"):
    files = list(glob.glob(f"{folder}\*.{ext}"))
    imgs = multiprocessApply(
        cv2.imread,
        files,
        "Reading files",
        extraArgs=(cv2.IMREAD_GRAYSCALE,),
        applyAsync=False,
    )

    cv_img = np.stack(imgs, axis=2)
    if out:
        np.save(out, cv_img)
    return cv_img


def stackData_script():
    Fire(stackData)


if __name__ == "__main__":
    stackData_script()
