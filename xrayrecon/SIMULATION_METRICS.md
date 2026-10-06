# Simulation timing, AM fidelity, and CUDA backprojection

Delivery objective: make CT/AM performance runs identifiable and measurable,
report AM reconstruction fidelity against known references, and accelerate
filtered backprojection in the existing GPU workflow.

## Delivery goals

| ID / outcome | Dependencies | Acceptance criteria | Verification | Non-goals | Status |
| --- | --- | --- | --- | --- | --- |
| G1: identifiable terminal timing records | Existing simulation APIs | Detailed completed wall time, sample, spacing, shape, angles, hardware and backends; preserve return types | CT/AM timing, editor forwarding, clock/device and failure tests | Dashboard, automatic file writes, voxelization/display timing | Complete; original verbose terminal format superseded by G4 |
| G2: honest AM SNR and SSI | G1, synthetic ground truth | Raw/regularized volume metrics; selected slice/mode metrics including calibration; explicit unavailable cases; array-only AM mapping | Known-error formula, exact/noisy images, slice aggregation, small/constant/missing reference and Qt mode tests | New reconstruction physics, measured-scanner accuracy claims | Complete |
| G3: parallel GPU backprojection and updated study | G1, existing CUDA workflow | Actual CUDA kernel and GPU FFT; geometry/interpolation/units match CPU; GPU CT/AM and residual maps use it; explicit CPU override; numerical and benchmark gates | CPU/CUDA parity, dtypes/shapes/angles, streams/input integrity, end-to-end tests, packaged kernel and before/after benchmark | GPU LUT interpolation, new dependency/toolkit installation, alternate reconstruction methods | Complete |

G3 was added when the requested scope expanded to backprojection. No source
CT/Gerber data is modified and label transfer remains an explicit action.

The follow-up request simplifies terminal output without removing measurements:

| ID / outcome | Dependencies | Acceptance criteria | Verification | Non-goals | Status |
| --- | --- | --- | --- | --- | --- |
| G4: compact timing output | Existing G1 records | Comment out verbose JSON calls; show only action, CPU/GPU and total/step seconds truncated to two decimals; retain full precision internally | Updated timing-format, truncation and failure tests | Simulation algorithm changes, deleting detailed measurements | Complete |
| G5: identify actions and document the format | G4, existing UI callbacks | CT, AM plot/volume and demo actions label their timing summaries; documentation describes display/internal timing scope | Editor integration and existing signal tests | Benchmark regeneration, rendering/voxelization timing | Complete |

## Algorithm assessment and implementation

The installed scikit-image `iradon` implementation performs filtered
backprojection: zero-pad the detector axis to a power of two (minimum 64),
apply the discrete ramp filter in Fourier space, interpolate each filtered
projection at `column*cos(theta) - row*sin(theta)`, accumulate over angles,
then normalize by `pi / (2 * angle_count)`. The application uses linear
interpolation, `circle=False`, and a centered crop/pad for rectangular slices.
The projection-residual map uses the same backprojection without a ramp filter.

Every output pixel/channel is independent. The custom CUDA kernel assigns one
thread to each output pixel/channel and sums angles in their original order.
There are no atomics and no output-pixel-by-angle temporary grids. Geometry
and interpolation use double precision; float32 output casts after each angle
to follow CPU accumulation. Fused multiply-add is disabled during compilation.
Zero-exterior interpolation, detector center, default square output size, and
normalization follow the CPU reference. PyTorch runs the FFT/filter/IFFT on CUDA.

The kernel compiles through NVRTC supplied with CUDA-enabled PyTorch and loads
through the CUDA driver API. Compilation/modules are cached per device in the
process. No `nvcc`, C++ compiler, new framework, or network service is required.
The `.cu` source is included in built wheels. Missing CUDA/NVRTC or compilation
failure raises an error; it does not silently execute on CPU. This implementation
supports the application's ramp and unfiltered linear `circle=False` paths.

`backprojection_backend` defaults to `projection_backend` in CT, simulated AM,
and paired-projection AM reconstruction. The editor's GPU selection therefore
enables it automatically. Set `backprojection_backend="cpu"` to reproduce the
previous partial GPU workflow. AM LUT interpolation, regularization, photon
noise and other quality calculations remain on CPU.

## Timing records

At the user's request, the original verbose JSON print calls are commented
out. Each successful simulation instead prints one compact line containing
only the button/action, CPU/GPU, total time and step times. Seconds are truncated
to two decimal places, rather than rounded. CT lists projection/backprojection;
AM lists acquisition, reconstruction with basis/residual backprojection
subintervals, and metric calculation. Device names, sample/settings metadata,
and numerical fidelity results are omitted from this terminal summary.

The displayed total is `elapsed_s + metrics_elapsed_s` (the latter applies to
AM). Internal measurements retain full precision. The existing
`print_simulation_record` function remains available for detailed JSON output,
and simulations retain the disabled calls as comments.

Unknown sample/spacing metadata is `null`; direct callers may supply
`sample_file` and `voxel_spacing_mm` in X/Y/Z order. The editor supplies its
stackup path and actual voxel spacing. Larger spacing represents coarser voxels;
angle count separately describes angular sampling density. `action_name` can
identify a direct caller's action; otherwise the label is CT/AM simulation.
The GUI supplies labels for the CT, AM plot, AM volume and showcase/demo actions.

`elapsed_s` starts at simulation entry and ends after completed numerical
reconstruction, including backend initialization, compilation and transfers.
It excludes prior Gerber rasterization/voxelization, rendering, terminal output,
and fidelity calculation. CUDA device-to-host copies complete before returning.
`metrics_elapsed_s` separately measures AM SNR/SSI calculations. Compare warmed
runs separately from first runs, and compare identical samples/settings.

CT records `projection_s` and `backprojection_s`. AM records `acquisition_s`
(projection/spectra/noise), `reconstruction_s` (LUT/decomposition/backprojection/
quality/regularization), and `basis_backprojection_s`,
`quality_backprojection_s`, `backprojection_s`. The last three are subintervals
inside reconstruction, so do not add them to reconstruction again. Setup and
minor bookkeeping are included in total elapsed time rather than stage sums.
Hardware identity and projection/spectrum/backprojection backends are explicit.

AM retains timing as `result.run_metrics` and fidelity as
`result.reconstruction_metrics`; dictionary values remain exclusively arrays.
Calibration preserves timing provenance and computes a separate calibrated
fidelity entry. The detailed simulation record describes raw/regularized
outputs; calibration is a subsequent operation. Fidelity results remain visible
in the analysis panel and through the result attributes.

## Fidelity definitions and interpretation

The term SSI is interpreted here as structural similarity, implemented with
scikit-image's SSIM. Both metrics compare matching scalar Z-effective and
relative electron-density arrays in their original units, including all
background voxels and boundaries. No clipping, fitting, threshold masking or
independent image normalization is applied by the metric calculation.

`SNR_dB = 10 * log10(sum(reference**2) / sum((reconstruction-reference)**2))`.
This measures reconstruction-error signal-to-noise, including artifacts and
bias, rather than estimating photon noise from a homogeneous image region.
Higher is better; negative values are valid. Exact nonzero recovery is `inf`
(serialized as the string `"inf"` in strict JSON); zero reference energy is
unavailable (`null`).

SSI uses a uniform odd window up to 7 pixels, sample covariance, K1=0.01,
K2=0.03, and `data_range = reference.max() - reference.min()`. The range is
fixed for a field across the volume and all reconstruction modes. Volume SSI
is the arithmetic mean of 2-D slice SSIM values, while volume SNR uses total
signal/error energy across all slices. SSI equals 1 for an exact image and can
be negative. Constant references or spatial dimensions smaller than 3 pixels
report unavailable SSI with an explicit reason. No reference means no fidelity
claim; measured projections still retain their data-consistency/confidence maps.

The analysis panel shows full-slice metrics for the selected reconstruction
mode. These complement the existing foreground errors and confidence maps.
Synthetic-reference metrics are decision support, not measured-scanner or
material-correctness validation. In particular, raw Z-effective reconstruction
may be poor in air or weakly absorbing material even when material-regularized
output looks convincing; both modes are reported separately.

## Timing study (2026-10-06)

Command: `python -B benchmarks/benchmark_cuda_backprojection.py --repeats 5
--output benchmarks/backprojection_study.json`.
The workspace JSON artifact contains all samples, first-call times,
medians/spread, numerical errors, workflow gates and AM fidelity/stage records.
No unrelated GPU workload was run concurrently with this study.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU; Intel64 Family 6 Model 186;
Python 3.12, PyTorch 2.14.0+cu130. Values below are warmed host-to-host medians
including transfers; standalone backprojection also includes constructor/setup
with the already compiled kernel. CUDA event compute times are in the JSON.

| Backprojection slice / angles | dtype | CPU ms | CUDA ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 96 x 128 / 64 | float32 | 7.194 | 0.910 | 7.91x |
| 96 x 128 / 64 | float64 | 6.642 | 0.964 | 6.89x |
| 256 x 384 / 180 | float32 | 323.958 | 6.097 | 53.13x |
| 256 x 384 / 180 | float64 | 317.319 | 5.837 | 54.36x |
| 512 x 512 / 180 | float32 | 580.497 | 47.857 | 12.13x |
| 512 x 512 / 180 | float64 | 568.980 | 66.286 | 8.58x |

All FBP numerical gates passed. Maximum error divided by maximum CPU reference
amplitude was <= 9.12e-6 for float32 and <= 5.10e-14 for float64. The float32
paths include different FFT rounding between CPU SciPy and GPU cuFFT.

Workflows used the same synthetic FR4/copper/tin volume, 2 x 96 x 128 voxels,
0.1/0.1/0.05 mm spacing and 64 angles. AM was noiseless. Workflow host timers
include fidelity calculations and terminal-record output routed to stderr.

| Workflow | CPU ms | Previous partial GPU ms | GPU with backprojection ms |
| --- | ---: | ---: | ---: |
| CT | 84.646 | 40.549 | 27.675 |
| AM | 1075.619 | 724.716 | 861.216 |

CT improved 1.47x relative to the previous partial GPU workflow and 3.06x over
CPU. AM backprojection itself improved (last-run combined basis/residual time
40.96 ms partial GPU versus 10.00 ms with GPU backprojection), but **overall AM
was slower in this study**, with CPU LUT work and run variation dominating.
Do not infer an end-to-end AM speedup from standalone kernel speed. AM's
material-interior property drift and classification gates passed, and CT's
numerical gate passed. These synthetic laptop measurements are not a guarantee
for other boards, devices, resolutions, thermal states or workloads.

The original spectrum and projection benchmark scripts retain their isolated
stage comparisons. Projection studies explicitly hold backprojection on CPU;
simulation logs go to stderr so their stdout JSON remains parseable.

## Verification and repository state from the initial delivery

- Baseline CT/AM checks before edits: 26 passed, 1 CUDA test deselected.
- Final regression: `python -m pytest -q -m "not slow"`: 241 passed,
  1 slow test deselected, 181 existing dependency deprecation warnings.
- Full 24 x 96 x 128 showcase test (the deselected slow test): 1 passed in
  7.17 seconds; its existing reconstruction-duration acceptance gate passed.
- CUDA numerical and scientific gates in the five-repeat study: all passed.
- `pip wheel . --no-deps --no-build-isolation`: passed; inspected wheel
  contains `cuda_backprojection.cu` and its Python loader.
- `git diff --check`: passed.

Active branch: `gpu-test`. No commits or remote actions were created. All task
changes remain uncommitted for review. The initial working tree was clean;
there were no pre-existing modified/untracked files to include or alter.
Existing presentation assets, notebooks and datasets were untouched.

Files changed or added, grouped by delivered capability:

- Run timing and integration: `xrayrecon/xrayrecon/simulation_metrics.py`,
  `xrayrecon/xrayrecon/ct_simulation.py`,
  `xrayrecon/xrayrecon/alvarez_macovski.py`,
  `xrayrecon/xrayrecon/gui/app.py`.
- AM fidelity: `xrayrecon/xrayrecon/reconstruction_metrics.py`,
  `xrayrecon/xrayrecon/am_calibration.py`,
  `xrayrecon/xrayrecon/gui/am_plot_window.py`.
- CUDA backprojection: `xrayrecon/xrayrecon/backprojection.py`,
  `xrayrecon/xrayrecon/cuda_backprojection.py`,
  `xrayrecon/xrayrecon/cuda_backprojection.cu`,
  `xrayrecon/xrayrecon/gui/stackup_editor.py`, `pyproject.toml`.
- Timing study: `benchmarks/benchmark_cuda_backprojection.py`,
  `benchmarks/benchmark_cuda_projection.py`,
  `benchmarks/benchmark_cuda_spectrum.py`, `benchmarks/backprojection_study.json`.
- Tests: `xrayrecon/xrayrecon/tests/test_simulation_metrics.py`,
  `xrayrecon/xrayrecon/tests/test_reconstruction_metrics.py`,
  `xrayrecon/xrayrecon/tests/test_cuda_backprojection.py`,
  `xrayrecon/xrayrecon/tests/test_embedded_registration.py`.
- Documentation: `README.md`, `xrayrecon/SIMULATION_METRICS.md`.

The remaining performance limitation is CPU AM LUT work. NVRTC/library discovery
was exercised on Windows with the available CUDA-enabled PyTorch; the Linux
loader path is implemented but not hardware-tested in this workspace.

## Compact-output follow-up verification

The compact-output and action-label goals (G4/G5) are complete. The verbose
simulation print calls are commented out; terminal output now contains only
action/processor and times. Detailed records and fidelity arrays are retained.

The timing, embedded signal and CT test modules passed together: 22 tests,
including the existing CUDA CT comparison. After extending the existing editor
test to exercise both AM plot and AM volume actions, its timing module passed
again: 6 tests. The precision check verifies truncation (1.239 -> 1.23), exact
two-decimal formatting, inclusion of metric time in the displayed total, and
omission of device names and sample metadata. `git diff --check` passed.

Files changed for this follow-up: `README.md`, `xrayrecon/SIMULATION_METRICS.md`,
`xrayrecon/xrayrecon/simulation_metrics.py`,
`xrayrecon/xrayrecon/ct_simulation.py`,
`xrayrecon/xrayrecon/alvarez_macovski.py`,
`xrayrecon/xrayrecon/gui/app.py`,
`xrayrecon/xrayrecon/tests/test_simulation_metrics.py`, and
`xrayrecon/xrayrecon/tests/test_embedded_registration.py`.

Work remains uncommitted on `gpu-test`; no commits or remote actions were
created. Earlier uncommitted implementation and benchmark results were retained;
unrelated files were untouched. Printed sub-centisecond steps appear as `0.00 s`
at the requested precision; internal timings retain their original precision.
