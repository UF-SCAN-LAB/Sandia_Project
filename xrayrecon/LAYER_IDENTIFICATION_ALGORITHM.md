# Layer Identification Algorithm

This document describes the mathematics behind the layer-segmentation algorithm
implemented in [`xrayrecon/layerseg.py`](xrayrecon/layerseg.py). The algorithm
finds PCB layer and via boundaries along the slice (z) axis of a 3D X-ray CT
volume by correlating adjacent slices and locating discontinuities in that
correlation signal.

## 1. Problem setup

Let the CT volume be a data cube

$$
C \in \mathbb{R}^{H \times W \times N}
$$

where $H \times W$ is the in-plane slice resolution and $N$ is the number of
slices along the stacking axis. Write $C_k \in \mathbb{R}^{H \times W}$ for the
$k$-th slice, $k = 1, \dots, N$.

Within a bounded region (window) of the volume, adjacent slices that belong to
the *same* physical layer (e.g. copper foil, dielectric) look nearly
identical, since the material composition does not change from slice to
slice. At a **layer boundary**, the slice content changes abruptly (new
material, new via, new trace pattern). The algorithm detects these boundaries
by measuring slice-to-slice self-similarity and finding where it drops
sharply.

## 2. Slice-to-slice template matching score

First, each slice is morphologically closed to suppress small noise pixels:

$$
\hat{C}_k = C_k \bullet S = (C_k \oplus S) \ominus S
$$

using a $3\times3$ structuring element $S$ (a "close" operation: dilation
followed by erosion), implemented via `cv.morphologyEx(..., cv.MORPH_CLOSE)`.

For each pair of consecutive slices $(\hat{C}_k, \hat{C}_{k+1})$, a
normalized cross-correlation coefficient is computed using OpenCV's
`TM_CCOEFF` template matching mode:

$$
R(k) = \sum_{x,y} \big(\hat{C}_k(x,y) - \bar{\hat{C}}_k\big)\,
                    \big(\hat{C}_{k+1}(x,y) - \bar{\hat{C}}_{k+1}\big)
$$

where $\bar{\hat{C}}_k$ is the mean pixel value of slice $k$ over the window.
Because the window size equals the template size, this reduces to a single
scalar per slice pair (no spatial sliding), giving a similarity time-series

$$
R = \big(R(1), R(2), \dots, R(N-1)\big).
$$

The score series is then normalized by its maximum magnitude:

$$
s(k) = \frac{R(k)}{\max_j R(j)}, \qquad s(k) \in [0, 1]
$$

(`templateMatchScores`). High $s(k)$ means slice $k$ and $k{+}1$ are
similar (same layer, no boundary between them); a drop in $s(k)$ signals a
transition.

## 3. Boundary detection via second-difference peaks

Layer/via boundaries appear as *rapid changes* in $s(k)$ — i.e., large
curvature — rather than in the raw value. The algorithm therefore works with
the **discrete second derivative**:

$$
s''(k) = s(k+1) - 2\,s(k) + s(k-1) = \Delta^2 s(k)
$$

computed via two applications of `np.diff`, then zero-padded at both ends so
indices line up with the original score array, and renormalized:

$$
\tilde{s}''(k) = \frac{s''(k)}{\max_j s''(j)}
$$

Peaks of $\tilde{s}''(k)$ mark the boundaries of high-curvature regions
(the "shoulders" of a correlation dip caused by a boundary). Peaks are
detected with a minimum height threshold:

$$
P = \{\, k : k \text{ is a local max of } \tilde{s}''(\cdot),\ \tilde{s}''(k) > 0.2 \,\}
$$

(`find_peaks(diff2, height=0.2)`).

## 4. Pairing peaks into layer/via slice ranges

The peak indices in $P = (p_1, p_2, \dots, p_M)$ are expected to occur in
**pairs** bracketing each transition region (the start and end of a
correlation dip). If $M$ is odd, the first (unpaired) peak is dropped so
peaks can be grouped as:

$$
\text{pairs} = \big\{\, (p_{2i-1}, p_{2i}) \;\big|\; i = 1, \dots, \lfloor M/2 \rfloor \,\big\}
$$

For each pair $i$ bounding a via/transition region $(p_{2i-1}-1,\ p_{2i}+1)$
(the boundary is nudged one slice outward since the pair marks the base of
the correlation peak, not its true edge), the algorithm defines two
alternating slice ranges walking along the stack:

$$
\text{Via}_i = \big[\, \text{pairs}_{i-1,\,\text{end}} + 1,\ \ \text{pairs}_{i,\,\text{start}} - 1 \,\big]
$$
$$
\text{Layer}_i = \big[\, \text{pairs}_{i,\,\text{start}},\ \ \text{pairs}_{i,\,\text{end}} \,\big]
$$

i.e., the transition band itself is labeled a via/junction, and the gap
between consecutive transition bands is labeled a stable layer. The first
(degenerate) via entry, corresponding to the artificial "pair 0" sentinel
used to bootstrap the loop, is discarded.

**Fallback.** If no peak pairs are found (e.g. too few slices, or a uniform
synthetic volume with no via/trace texture to key on), the volume is instead
split evenly into

$$
n = \min(4,\ \max(1,\ N+1))
$$

equal-width regions using `np.linspace(0, N, n+1)` as boundaries, labeled
`Region 1 .. Region n`, so the UI always has selectable regions.

Finally, the first and last ranges are clamped to the volume extent:

$$
\text{range}_1[\text{start}] = 0, \qquad \text{range}_{\text{last}}[\text{end}] = N-1
$$

and each range is expanded to the explicit list of slice indices it spans
(clipped to $[0, N]$).

## 5. Choosing the best analysis window (multi-window search)

Because template matching is done on a sub-window of each slice rather than
the whole image, the *placement and size* of that window strongly affects
signal quality (e.g. a window over empty substrate has poor contrast). The
algorithm searches over a grid of candidate windows and, optionally, over
multiple window sizes.

For a window size $(w_h, w_w)$ and stride (spacing) $\sigma$, candidate
window origins are

$$
i \in \{0, \sigma, 2\sigma, \dots\} \subseteq [0,\ H-w_h], \qquad
j \in \{0, \sigma, 2\sigma, \dots\} \subseteq [0,\ W-w_w]
$$

giving a set of sub-cubes $C^{(i,j)} = C[i:i+w_h,\ j:j+w_w,\ :]$. The score
series $s^{(i,j)}(k)$ (Section 2) is computed for every window
(`scoresLstForWindow`), and — if multiple window sizes are supplied — the
results are stacked across all sizes and positions into one matrix of
candidate score series.

### 5.1 Feature score for each candidate window

Each candidate's score series $s(k)$ is reduced to a single scalar
"quality" feature that rewards a strong, regular, well-separated,
well-shaped set of peaks and penalizes noisy/ambiguous ones
(`_peakFeatures`):

1. Detect peaks $k \in \text{Pk}$ of $s(\cdot)$ with height $> 0.2$; compute
   for each peak its **width** $w_{\text{peak}}$ (`peak_widths`) and
   **prominence** $\rho_{\text{peak}}$ (`peak_prominences`).
2. Compute summary statistics:
   $$
   n_{\text{pk}} = |\text{Pk}|, \quad
   \bar{h} = \operatorname{mean}(s(k){:}k\in\text{Pk}), \quad
   \bar{d} = \operatorname{mean}(\Delta \text{Pk}), \quad
   \bar{\rho} = \operatorname{mean}(\rho_{\text{peak}})
   $$
3. For each peak, fit a Gaussian $\mathcal{N}(\mu, \sigma^2)$ to the local
   window $s(k - w_{\text{peak}} : k + w_{\text{peak}})$ via
   `scipy.stats.norm.fit`, and measure the mean absolute fit residual
   $$
   e_{\text{peak}} = \operatorname{mean}\big| s(x) - \mathcal{N}(x;\mu,\sigma^2) \big|,
   \qquad \bar{e} = \operatorname{mean}(e_{\text{peak}})
   $$
   (a smaller $\bar e$ means the correlation dip looks like a clean,
   single Gaussian-shaped transition rather than noise).
4. Combine into the feature score:
   $$
   F = \frac{n_{\text{pk}}^2 \cdot \bar{h} \cdot \bar{d} \cdot \bar{\rho}}{\bar{e}}
   $$

Non-finite feature values (e.g. from empty peak sets or zero-height slices)
are set to $0$.

### 5.2 Window selection

The best window/size combination is the argmax over all candidates:

$$
(i^*, j^*, \text{size}^*) = \arg\max_{(i,j,\text{size})} F^{(i,j,\text{size})}
$$

Its score series $s^{(i^*,j^*)}(k)$ is passed to the peak-pairing procedure
of Sections 3–4 to produce the final layer/via slice map
(`layerInfoForWindow`).

## 6. Output

The algorithm returns a mapping

$$
\{\ \text{"Layer Name"}: [\ell_1, \ell_2, \dots],\ \ \text{"Slices"}: [\sigma_1, \sigma_2, \dots]\ \}
$$

where $\ell_m \in \{\text{"Layer } i\text{"},\ \text{"Via } i{-}(i{+}1)\text{"},\ \text{"Region } i\text{"}\}$
and $\sigma_m$ is the corresponding array of slice indices — consumed by
`LayerModel` (`xrayrecon/gui/layermodel.py`) to populate the layer table and
drive the 2D/3D viewers.
