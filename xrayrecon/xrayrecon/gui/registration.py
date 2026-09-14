from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..registration import (
    RegistrationSession,
    RigidScaleTransform,
    propose_initial_transform,
    registration_quality,
    resample_volume,
)
from .view3d import PCBViewer3D


class RegistrationPanel(QtWidgets.QWidget):
    """Embedded 5-DOF Gerber/CT registration workspace.

    The panel provides a bounded initial suggestion and live red/green manual
    overlay without owning a top-level window. ``RegistrationSession`` keeps
    the edit and accepted transform independent from widget visibility.

    The live preview resamples/normalizes a downsampled copy of both volumes
    (see `_downsample_for_preview`) and debounces slider input (see
    `_PREVIEW_DEBOUNCE_MS`) -- a full-resolution `affine_transform` +
    `np.percentile` scan on every drag tick would freeze the UI on
    realistically-sized CT volumes. The accepted transform itself (mm-based)
    is resolution-independent, so this doesn't affect registration accuracy.
    """

    sigTransformAccepted = QtCore.Signal(object)  # RigidScaleTransform
    sigCloseRequested = QtCore.Signal()

    _PREVIEW_MAX_DIM = 96
    _PREVIEW_DEBOUNCE_MS = 80

    def __init__(
        self,
        session: RegistrationSession | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session or RegistrationSession()
        self.moving_hwz: np.ndarray | None = None
        self.moving_spacing: tuple[float, float, float] | None = None
        self.fixed_hwz: np.ndarray | None = None
        self.fixed_spacing: tuple[float, float, float] | None = None
        self._preview_moving: np.ndarray | None = None
        self._preview_moving_spacing: tuple[float, float, float] | None = None
        self._preview_fixed: np.ndarray | None = None
        self._preview_fixed_spacing: tuple[float, float, float] | None = None
        self._fixed_norm: np.ndarray | None = None
        self._proposalSummary = ""

        self._previewTimer = QtCore.QTimer(self)
        self._previewTimer.setSingleShot(True)
        self._previewTimer.timeout.connect(self._updatePreview)

        def _spinbox(value, bounds, step):
            sbox = pg.SpinBox(value=value)
            sbox.setOpts(bounds=bounds, step=step)
            return sbox

        self.txSbox = _spinbox(0.0, (-1.0, 1.0), 0.05)
        self.tySbox = _spinbox(0.0, (-1.0, 1.0), 0.05)
        self.tzSbox = _spinbox(0.0, (-1.0, 1.0), 0.02)
        self.rotSbox = _spinbox(0.0, (-180.0, 180.0), 0.5)
        self.scaleSbox = _spinbox(1.0, (0.2, 5.0), 0.01)

        self.overlayViewer = PCBViewer3D()

        self.topViewBtn = QtWidgets.QPushButton("Top View")
        self.isoViewBtn = QtWidgets.QPushButton("Isometric View")
        self.overlayGuide = QtWidgets.QLabel(
            "Overlay guide: Gerber = red, CT = green, aligned structures = yellow. "
            "Use Top View to compare board landmarks and Isometric View to inspect depth."
        )
        self.overlayGuide.setWordWrap(True)

        self.qualityLabel = QtWidgets.QLabel(
            "Load CT and Gerber data, then activate registration."
        )
        self.qualityLabel.setWordWrap(True)
        self.backBtn = QtWidgets.QPushButton("Back to Main Workspace")
        self.suggestBtn = QtWidgets.QPushButton("Suggest Initial Alignment")
        self.resetBtn = QtWidgets.QPushButton("Reset")
        self.acceptBtn = QtWidgets.QPushButton("Accept")

        title = QtWidgets.QLabel("Gerber to Real CT Registration")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.backBtn)

        controls = QtWidgets.QGridLayout()
        for row, (label, spinbox) in enumerate(
            (
                ("Tx (mm)", self.txSbox),
                ("Ty (mm)", self.tySbox),
                ("Tz (mm)", self.tzSbox),
                ("Rot Z (deg)", self.rotSbox),
                ("Scale", self.scaleSbox),
            )
        ):
            controls.addWidget(QtWidgets.QLabel(label), row, 0)
            controls.addWidget(spinbox, row, 1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.suggestBtn)
        action_row.addWidget(self.resetBtn)
        action_row.addWidget(self.acceptBtn)
        action_row.addStretch()

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("Camera:"))
        view_row.addWidget(self.topViewBtn)
        view_row.addWidget(self.isoViewBtn)
        view_row.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(header)
        layout.addWidget(self.overlayGuide)
        layout.addLayout(view_row)
        layout.addWidget(self.overlayViewer, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self.qualityLabel)
        layout.addLayout(action_row)

        for sbox in (self.txSbox, self.tySbox, self.tzSbox, self.rotSbox, self.scaleSbox):
            sbox.sigValueChanging.connect(lambda *_: self._schedulePreviewUpdate())

        self.backBtn.clicked.connect(self.sigCloseRequested.emit)
        self.suggestBtn.clicked.connect(self._suggestInitialAlignment)
        self.resetBtn.clicked.connect(self._resetSliders)
        self.acceptBtn.clicked.connect(self._onAccept)
        self.topViewBtn.clicked.connect(self.overlayViewer.setCameraTop)
        self.isoViewBtn.clicked.connect(self.overlayViewer.resetCamera)

        self._setControlsEnabled(False)

    def _setControlsEnabled(self, enabled: bool) -> None:
        for widget in (
            self.txSbox,
            self.tySbox,
            self.tzSbox,
            self.rotSbox,
            self.scaleSbox,
            self.suggestBtn,
            self.resetBtn,
            self.acceptBtn,
            self.topViewBtn,
            self.isoViewBtn,
        ):
            widget.setEnabled(enabled)

    def setVolumes(
        self,
        moving_hwz: np.ndarray,
        moving_spacing,
        fixed_hwz: np.ndarray,
        fixed_spacing,
    ) -> bool:
        """Configure the panel, preserving edits when the inputs are unchanged.

        Returns True when a new input signature reset the session.
        """
        reset = self.session.configure(
            moving_hwz, moving_spacing, fixed_hwz, fixed_spacing
        )
        self.moving_hwz = moving_hwz
        self.moving_spacing = tuple(float(value) for value in moving_spacing)
        self.fixed_hwz = fixed_hwz
        self.fixed_spacing = tuple(float(value) for value in fixed_spacing)
        self._preview_moving, self._preview_moving_spacing = self._downsample_for_preview(
            moving_hwz, self.moving_spacing
        )
        self._preview_fixed, self._preview_fixed_spacing = self._downsample_for_preview(
            fixed_hwz, self.fixed_spacing
        )
        self._fixed_norm = self._normalize(self._preview_fixed)

        extent_mm = [
            size * spacing for size, spacing in zip(fixed_hwz.shape, self.fixed_spacing)
        ]
        self.txSbox.setOpts(bounds=(-extent_mm[0], extent_mm[0]))
        self.tySbox.setOpts(bounds=(-extent_mm[1], extent_mm[1]))
        self.tzSbox.setOpts(bounds=(-extent_mm[2], extent_mm[2]))
        self._proposalSummary = ""
        if self.session.proposal is not None:
            proposal = self.session.proposal
            self._proposalSummary = (
                f"suggestion {proposal.confidence} confidence, "
                f"search gain {proposal.improvement:+.3f}"
            )
        self._setTransform(self.session.current_transform)
        self._setControlsEnabled(True)
        self._updatePreview()
        self.overlayViewer.setCameraTop()
        return reset

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
        self.session.proposal = None
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
        if self._preview_moving is None or self._preview_fixed is None:
            return
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
            self.session.record_proposal(proposal)
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
        if (
            self._preview_moving is None
            or self._preview_moving_spacing is None
            or self._preview_fixed is None
            or self._preview_fixed_spacing is None
            or self._fixed_norm is None
        ):
            return
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
        self.session.update_current(transform, score)
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
        self._updatePreview()
        self.sigTransformAccepted.emit(self.session.accept())
