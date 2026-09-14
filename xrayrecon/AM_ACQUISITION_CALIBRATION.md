# Alvarez–Macovski Acquisition and Calibration

## Physical contract

`AMAcquisitionConfig` is the single source of truth for spectra, angular
sampling, pixel size, photon-count assumptions, LUT bounds, calibration ID,
electron-density units, and provenance. Configuration JSON uses schema version
1. Electron density is relative to liquid water.

Each configuration exposes an `am-physics-v1` fingerprint over the response-
critical settings: spectrum energies and weights, pixel size, angular span,
electron-density units, and LUT bounds. View count, photon noise, provenance,
spectrum labels, and calibration ID are excluded so a reference calibration can
be reused for compatible acquisitions with different sampling or noise.

Simulation accepts either the legacy arguments or one explicit configuration,
never both. `simulate_am_projections` produces validated paired low/high
negative-log projections. `reconstruct_am_projections` consumes the same type,
including measured data, without requiring synthetic ground truth.

Measured import currently accepts three `.npy` files: low-energy negative-log
projections, high-energy negative-log projections, and projection angles. The
caller must supply an explicit `source_kind="measured"` configuration and a
calibration identifier. Proprietary detector corrections and raw-format
conversion remain upstream responsibilities.

## Calibration contract

The synthetic reference phantom contains separated FR4, alumina, aluminum,
copper, tin, tungsten, and gold inserts. Calibration fits FR4, alumina, copper,
tin, and tungsten. Aluminum and gold remain held out for cross-validation.

The calibrated center uses:

- material-regularized Z-effective, because the raw low-Z photoelectric ratio
  can collapse beside high-Z material; and
- raw relative electron density, because the Compton channel is more stable.

Raw and regularized arrays remain present and unchanged. Calibration adds
separately named calibrated values and interval bounds.

`AMCalibrationModel` JSON bundles use schema version 1 and retain the training
materials, response coefficients, uncertainty scales, source, and acquisition-
physics fingerprint. `save_calibration_model` refuses to overwrite an existing
bundle unless the caller explicitly opts in; `load_calibration_model` validates
the complete artifact before returning it.

Calibration application is fail-closed. It rejects missing acquisition metadata,
a different physics fingerprint, or a conflicting explicitly declared
calibration ID before creating any calibrated arrays. A successful application
records `calibration_compatibility="verified"`, which the embedded analysis
workspace displays as **acquisition physics verified**. Fingerprint agreement
means that declared acquisition settings match; it does not independently prove
that scanner metadata or detector preprocessing is correct.

## Uncertainty and provenance

Nominal 90% interval scales combine leave-one-material-out response error with
voxel-level projection consistency and density disagreement. Regression tests
require held-out empirical coverage between 80% and 97% and rank correlation
of at least 0.60 between calibrated risk and actual synthetic error.

These intervals are conditional on the configured spectra and synthetic
reference model. They are not universal probabilities. A result is only
labeled measured when paired measured projections were supplied, and the
application never turns the synthetic showcase calibration into a measured
scanner claim.
