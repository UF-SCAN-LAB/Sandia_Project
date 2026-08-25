from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from qtextras import EasyWidget

from ..registration import (
    RigidScaleTransform,
    propose_initial_transform,
    registration_quality,
    resample_volume,
)
from .view3d import PCBViewer3D


class RegistrationWindow(QtWidgets.QMainWindow):
    """Reviewable 5-DOF (XYZ translate + Z-rotation + uniform scale) alignment
    tool: bounded initial suggestion plus a live red/green manual overlay of
    the Gerber-derived MOVING volume against the real-CT FIXED volume, following
    the non-modal preview pattern in `cubecorrections.rotation.RotateWindow`.

    The live preview resamples/normalizes a downsampled copy of both volumes
    (see `_downsample_for_preview`) and debounces slider input (see
    `_PREVIEW_DEBOUNCE_MS`) -- a full-resolution `affine_transform` +
    `np.percentile` scan on every drag tick would freeze the UI on
    realistically-sized CT volumes. The accepted transform itself (mm-based)
    is resolution-independent, so this doesn't affect registration accuracy.
    """

    sigTransformAccepted = QtCore.Signal(object)  # RigidScaleTransform

    _PREVIEW_MAX_DIM = 96
    _PREVIEW_DEBOUNCE_MS = 80

    def __init__(
        self,
        moving_hwz: np.ndarray,
        moving_spacing,
        fixed_hwz: np.ndarray,
        fixed_spacing,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Register Gerber to Real CT")
        self.moving_hwz = moving_hwz
        self.moving_spacing = tuple(moving_spacing)
        self.fixed_hwz = fixed_hwz
        self.fixed_spacing = tuple(fixed_spacing)

        self._preview_moving, self._preview_moving_spacing = self._downsample_for_preview(
            moving_hwz, self.moving_spacing
        )
        self._preview_fixed, self._preview_fixed_spacing = self._downsample_for_preview(
            fixed_hwz, self.fixed_spacing
        )
        self._fixed_norm = self._normalize(self._preview_fixed)
        self._proposalSummary = ""

        self._previewTimer = QtCore.QTimer(self)
        self._previewTimer.setSingleShot(True)
        self._previewTimer.timeout.connect(self._updatePreview)

        extent_mm = [s * d for s, d in zip(fixed_hwz.shape, self.fixed_spacing)]

        def _spinbox(value, bounds, step):
            sbox = pg.SpinBox(value=value)
            sbox.setOpts(bounds=bounds, step=step)
            return sbox

        self.txSbox = _spinbox(0.0, (-extent_mm[0], extent_mm[0]), 0.05)
        self.tySbox = _spinbox(0.0, (-extent_mm[1], extent_mm[1]), 0.05)
        self.tzSbox = _spinbox(0.0, (-extent_mm[2], extent_mm[2]), 0.02)
        self.rotSbox = _spinbox(0.0, (-180.0, 180.0), 0.5)
        self.scaleSbox = _spinbox(1.0, (0.2, 5.0), 0.01)

        self.overlayViewer = PCBViewer3D()

        self.qualityLabel = QtWidgets.QLabel(
            "Alignment score: calculating... (compare transforms; review the overlay)"
        )
        self.qualityLabel.setWordWrap(True)
        self.suggestBtn = QtWidgets.QPushButton("Suggest Initial Alignment")
        self.resetBtn = QtWidgets.QPushButton("Reset")
        self.acceptBtn = QtWidgets.QPushButton("Accept")

        EasyWidget.buildMainWindow(
            [
                self.overlayViewer,
                [
                    [QtWidgets.QLabel("Tx (mm)"), self.txSbox],
                    [QtWidgets.QLabel("Ty (mm)"), self.tySbox],
                    [QtWidgets.QLabel("Tz (mm)"), self.tzSbox],
                    [QtWidgets.QLabel("Rot Z (deg)"), self.rotSbox],
                    [QtWidgets.QLabel("Scale"), self.scaleSbox],
                ],
                self.qualityLabel,
                [self.suggestBtn, self.resetBtn, self.acceptBtn],
            ],
            window=self,
        )

        for sbox in (self.txSbox, self.tySbox, self.tzSbox, self.rotSbox, self.scaleSbox):
            sbox.sigValueChanging.connect(lambda *_: self._schedulePreviewUpdate())

        self.suggestBtn.clicked.connect(self._suggestInitialAlignment)
        self.resetBtn.clicked.connect(self._resetSliders)
        self.acceptBtn.clicked.connect(self._onAccept)

        self._updatePreview()
        # Frame the volume once at startup -- resetting on every slider
        # change would fight the user's own camera pan/zoom while they're
        # adjusting alignment.
        self.overlayViewer.resetCamera()

    def currentTransform(self) -> RigidScaleTransform:
        return RigidScaleTransform(
            tx=self.txSbox.value(),
            ty=self.tySbox.value(),
            tz=self.tzSbox.value(),
            rot_z_deg=self.rotSbox.value(),
            scale=self.scaleSbox.value(),
        )

    def _resetSliders(self):
        self._proposalSummary = ""
        self.txSbox.setValue(0.0)
        self.tySbox.setValue(0.0)
        self.tzSbox.setValue(0.0)
        self.rotSbox.setValue(0.0)
        self.scaleSbox.setValue(1.0)
        self._updatePreview()

    def _setTransform(self, transform: RigidScaleTransform):
        """Populate the manual controls without accepting the transform."""
        self.txSbox.setValue(transform.tx)
        self.tySbox.setValue(transform.ty)
        self.tzSbox.setValue(transform.tz)
        self.rotSbox.setValue(transform.rot_z_deg)
        self.scaleSbox.setValue(transform.scale)

    def _suggestInitialAlignment(self):
        """Run the bounded search and leave its result open for review."""
        self.suggestBtn.setEnabled(False)
        self.qualityLabel.setText("Searching for an initial alignment...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            proposal = propose_initial_transform(
                self._preview_moving,
                self._preview_moving_spacing,
                self._preview_fixed,
                self._preview_fixed_spacing,
            )
            self._proposalSummary = (
                f"suggestion {proposal.confidence} confidence, "
                f"search gain {proposal.improvement:+.3f}"
            )
            self._setTransform(proposal.transform)
            self._updatePreview()
        except Exception as error:
            self._proposalSummary = ""
            QtWidgets.QMessageBox.warning(
                self,
                "Initial Alignment Failed",
                f"The bounded search could not produce a suggestion:\n{error}",
            )
            self._updatePreview()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.suggestBtn.setEnabled(True)

    @classmethod
    def _downsample_for_preview(cls, vol: np.ndarray, spacing):
        """Stride-based downsample so the live-preview resample/normalize
        cost stays bounded regardless of the real volume's resolution.
        Coarsens the physical spacing to match (fewer, larger voxels), so
        `resample_volume`'s mm-based math stays correct on the downsampled
        copy."""
        factors = tuple(
            max(1, int(np.ceil(size / cls._PREVIEW_MAX_DIM))) for size in vol.shape
        )
        if factors == (1, 1, 1):
            return vol, tuple(spacing)
        small = vol[tuple(slice(None, None, factor) for factor in factors)]
        small_spacing = tuple(s * factor for s, factor in zip(spacing, factors))
        return small, small_spacing

    @staticmethod
    def _normalize(vol: np.ndarray) -> np.ndarray:
        """Percentile-based normalization to [0, 1] -- FBP-reconstructed
        volumes have a handful of extreme ringing-artifact voxels that would
        otherwise wash out the rest of the overlay under a naive min/max
        normalization."""
        vol = vol.astype(np.float32)
        lo, hi = np.percentile(vol, [1, 99])
        if hi <= lo:
            return np.zeros_like(vol)
        return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

    def _schedulePreviewUpdate(self):
        """Debounce rapid slider-drag ticks into a single recompute -- restarts
        the timer on every call rather than queuing one per tick."""
        self._previewTimer.start(self._PREVIEW_DEBOUNCE_MS)

    def _updatePreview(self):
        transform = self.currentTransform()
        resampled_moving = resample_volume(
            self._preview_moving,
            self._preview_moving_spacing,
            self._preview_fixed.shape,
            self._preview_fixed_spacing,
            transform,
            order=1,
        )
        moving_norm = self._normalize(resampled_moving)
        score = registration_quality(self._preview_fixed, resampled_moving)
        self._currentQualityScore = score
        suffix = f" | {self._proposalSummary}" if self._proposalSummary else ""
        self.qualityLabel.setText(
            f"Alignment score: {score:.3f}{suffix}. "
            "This is a comparison aid, not proof of a correct registration; review the overlay."
        )

        h, w, z = self._preview_fixed.shape
        rgba = np.zeros((h, w, z, 4), dtype=np.uint8)
        rgba[..., 0] = (moving_norm * 255).astype(np.uint8)  # moving = red
        rgba[..., 1] = (self._fixed_norm * 255).astype(np.uint8)  # fixed = green
        alpha = np.clip(np.maximum(moving_norm, self._fixed_norm), 0.0, 1.0)
        rgba[..., 3] = (alpha * 255).astype(np.uint8)

        self.overlayViewer.setRGBADataCube(rgba)

    def _onAccept(self):
        self.sigTransformAccepted.emit(self.currentTransform())
