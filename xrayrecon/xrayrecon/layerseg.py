from typing import Dict, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
import pyqtgraph as pg
from qtextras import ImageViewer, multiprocessApply
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import norm


def templateMatchScores(cube: np.ndarray):
    # if (cube > 0).sum()/cube.size < 0.1:
    #   return np.zeros(cube.shape[-1]-1)
    # Transpose the volume to orient according to the slice plane selection
    filCube = cv.morphologyEx(cube, cv.MORPH_CLOSE, np.ones((3, 3)))
    scores = []
    iterCube = filCube.transpose([2, 0, 1])
    for prevSlice, curSlice in zip(iterCube, iterCube[1:, ...]):
        prevSlice = prevSlice.astype(np.float32, copy=False)
        curSlice = curSlice.astype(np.float32, copy=False)
        score = cv.matchTemplate(prevSlice, curSlice, cv.TM_CCOEFF)
        scores.append(score[0, 0])
    scores = np.array(scores)
    if scores.max() != 0:
        scores /= scores.max()
    return scores


def layerInfoFromScores(scores: np.ndarray):
    diff2 = np.diff(scores, 2)
    # Pad so that indices match locations in `scores`
    diff2 = np.concatenate([[0], diff2, [0]])
    diff2 /= diff2.max()
    peaks, _ = find_peaks(diff2, height=0.2)
    # Since 'peaks' should represent pairs of start, stop idxs, make sure there are
    # an even number of them
    peaks = peaks[len(peaks) % 2 :]
    pairs = [peaks[start : start + 2] for start in range(0, len(peaks), 2)]
    # Add dummy first pair for easier loop logic
    pairs.insert(0, [-1, -1])
    layerNames = []
    layerSlices = []
    for ii, pair in enumerate(pairs[1:], 1):
        # Pairs should start right before the base of each correlation peak, not
        # at the exact start
        pair[0] -= 1
        pair[-1] += 1
        layerNames.append(f"Via {ii-1}-{ii}")
        layerSlices.append([pairs[ii - 1][1] + 1, pair[0] - 1])
        layerNames.append(f"Layer {ii}")
        layerSlices.append(pair.tolist())
    # Drop first bad via layer
    if layerNames:
        del layerNames[0]
        del layerSlices[0]
        
    if not layerNames:
        # Fallback for when no clear repeating structure is found (e.g. not
        # enough slices, or uniform synthetic material bands with no
        # via/trace texture for the template matcher to key on -- small
        # synthetic demo volumes hit this often). Split evenly into a
        # handful of regions instead of one meaningless whole-volume row,
        # so the layer selector still gives the user something to select
        # rather than collapsing to a single unusable blob.
        n_regions = min(4, max(1, len(scores) + 1))
        bounds = np.linspace(0, len(scores), n_regions + 1).astype(int)
        layerNames = [f"Region {i + 1}" for i in range(n_regions)]
        layerSlices = [[int(bounds[i]), int(bounds[i + 1])] for i in range(n_regions)]
    # Make sure layers start and stop at beginning and end of cube
    layerSlices[0][0] = 0
    layerSlices[-1][-1] = len(scores) - 1
    layerSlices = [np.clip(np.arange(s[0], s[1] + 1, dtype=int), 0, len(scores)) for s in layerSlices]
    ret = {"Layer Name": layerNames, "Slices": layerSlices}
    return ret


sz = Tuple[int, int]


def scoresLstForWindow(cube: np.ndarray, winSize: sz, spacing=1):
    """
    :param cube: 3d data cube to test layer scores on
    :param winSize: (w, h) of the analysis window
    """
    cubeH, cubeW = cube.shape[:2]
    winH, winW = winSize
    rrange = np.arange(0, cubeH - winH + 1, spacing)
    crange = np.arange(0, cubeW - winW + 1, spacing)

    def window_gen():
        for ii in rrange:
            for jj in crange:
                yield cube[ii : ii + winH, jj : jj + winW, :]

    tot = (cubeH - winH) * (cubeW - winW) / (spacing**2)

    scoresLst = np.asarray(
        multiprocessApply(
            templateMatchScores, window_gen(), "getting scores", total=tot, debug=True
        )
    )

    return scoresLst


def layerInfoForWindow(
    cube, winSizes: Union[sz, Sequence[sz]], spacing=1, returnMetric=False
):
    if isinstance(winSizes[0], int):
        winSizes = [winSizes]
    scoresLst = []
    for size in winSizes:
        scoresLst.append(scoresLstForWindow(cube, size, spacing))
    scoresLst = np.vstack(scoresLst)
    features = getFeatures(scoresLst)
    bestIdx = int(np.argmax(features))
    bestScores = scoresLst[bestIdx]
    layerInfo = layerInfoFromScores(bestScores)

    # iv = overlayFeatures_win_slice(cube, crange, rrange, features, winW, winH)
    if returnMetric:
        return layerInfo, layerSegMetricsPlot(bestScores, layerInfo)


def overlayFeatures_win_slice(cube, crange, rrange, features, winW, winH):
    nRowWindows = len(rrange)
    nColWindows = len(crange)
    feature_img = features.reshape(nRowWindows, nColWindows)
    bestIdx = int(np.argmax(features))
    bestR, bestC = np.unravel_index(bestIdx, feature_img.shape)
    windowImg = cube.mean(2)
    cv.rectangle(
        windowImg,
        (crange[bestC], rrange[bestR]),
        (crange[bestC] + winW, rrange[bestR] + winH),
        255,
        5,
    )
    iv = ImageViewer(windowImg)
    feature_img = cv.resize(feature_img, windowImg.shape[::-1])
    feature_img /= feature_img.max()
    lut = pg.colormap.get("inferno").map(np.linspace(0, 1, 256)) * 255
    newItem = pg.ImageItem(feature_img)
    newItem.setLookupTable(lut)
    newItem.setOpacity(0.5)
    iv.mainImg.addItem(newItem)
    return iv


def getFeatures(scoresLst: np.ndarray):
    scores = np.vstack(scoresLst)
    # Throw out peaks we are pretty sure don't hold up right away
    # maxNormedScore = scores.max(1)
    # scores = scores[maxNormedScore > 0.3]
    features = np.array([_peakFeatures(s) for s in scores])
    features[~np.isfinite(features)] = 0
    return features


def _peakFeatures(score):
    # Rule out generally bad scores
    if score.max() == 0:
        return 0
    peaks, features = find_peaks(score, height=0.2)
    widths = peak_widths(score, peaks)[0].astype(int)
    proms = peak_prominences(score, peaks)[0]
    widths[widths == 0] = 1

    numPeaks = len(peaks)
    avgHeight = features["peak_heights"].mean()
    avgDist = np.diff(peaks).mean()
    avgProm = proms.mean()

    errs = []
    for peak, width in zip(peaks, widths):
        gausPortion = score[peak - width : peak + width]
        if len(gausPortion) == 0:
            continue
        mean, std = norm.fit(gausPortion)
        x = np.linspace(gausPortion.min(), gausPortion.max(), 2 * width)
        y = norm.pdf(x, mean, std)
        errs.append(np.abs(gausPortion - y[: len(gausPortion)]).mean())
    peakFitErr = np.mean(errs)

    return numPeaks * avgHeight * avgDist * avgProm * numPeaks / peakFitErr


def layerSegMetricsPlot(scores: np.ndarray, layerInfo: Dict):
    sliceLocs = [s[0] for s in layerInfo["Slices"] if len(s) > 0]
    metricPlt = pg.PlotWidget()
    plot: pg.PlotItem = metricPlt.plotItem
    plot.plot(scores, pen=pg.mkPen("k", width=3))
    plot.setLabel("bottom", "Slice Number")
    plot.setLabel("left", "Corr. Coef. (a.u.)")
    plot.setTitle("Slice locations based on template match scores")

    scatter = pg.ScatterPlotItem(
        sliceLocs, scores[sliceLocs], symbol="x", size=10, pen="r", brush="r"
    )
    plot.addItem(scatter)
    metricPlt.setBackground("w")
    return metricPlt
