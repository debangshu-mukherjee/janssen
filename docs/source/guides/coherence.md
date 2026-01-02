# Partial Coherence in Optical Microscopy

Janssen provides comprehensive support for partially coherent optical
simulations through coherent mode decomposition. This guide covers the
mathematical foundations and practical implementation.

## Why Partial Coherence Matters

Real optical sources are never perfectly coherent:

- **Lasers**: Finite linewidth (temporal) and possible mode structure (spatial)
- **LEDs**: Spatially incoherent (extended source) with broad spectra
- **Synchrotrons**: Anisotropic spatial coherence (different in H and V)
- **Thermal sources**: Fully incoherent both spatially and temporally

Ignoring partial coherence leads to artifacts like unrealistic speckle
contrast and incorrect resolution predictions.

## Spatial Coherence

### The Mutual Intensity

Spatial coherence is quantified by the **mutual intensity**:

$$
J(\mathbf{r}_1, \mathbf{r}_2) = \langle E^*(\mathbf{r}_1) E(\mathbf{r}_2) \rangle
$$

where $\langle \cdot \rangle$ denotes time averaging over optical cycles.

The **complex degree of coherence** normalizes this:

$$
\mu(\mathbf{r}_1, \mathbf{r}_2) = \frac{J(\mathbf{r}_1, \mathbf{r}_2)}{\sqrt{I(\mathbf{r}_1) I(\mathbf{r}_2)}}
$$

where $|\mu| = 1$ indicates full coherence and $|\mu| = 0$ incoherence.

```{figure} figures/spatial_coherence_kernel.svg
:alt: Spatial coherence kernel
:width: 80%

Spatial coherence kernel $|\mu(\Delta r)|$ for different source types.
A Gaussian kernel (solid) arises from Gaussian extended sources, while
a Jinc kernel (dashed) comes from circular incoherent sources via the
van Cittert-Zernike theorem.
```

### Van Cittert-Zernike Theorem

For an incoherent source of intensity $I_s(\mathbf{r}_s)$, the mutual
intensity at the observation plane is:

$$
J(\mathbf{r}_1, \mathbf{r}_2) \propto \mathcal{F}\{I_s\}
\left(\frac{\mathbf{r}_1 - \mathbf{r}_2}{\lambda z}\right)
$$

The spatial coherence width is approximately:

$$
\sigma_c \approx 0.44 \frac{\lambda z}{D}
$$

where $D$ is the source diameter and $z$ is the propagation distance.

## Temporal Coherence

### Coherence Length

Temporal coherence arises from finite spectral bandwidth. The coherence
length for a Gaussian spectrum is:

$$
L_c = \frac{2 \ln 2}{\pi} \frac{\lambda^2}{\Delta\lambda}
\approx 0.44 \frac{\lambda^2}{\Delta\lambda}
$$

where $\Delta\lambda$ is the FWHM spectral bandwidth.

```{figure} figures/temporal_coherence_spectrum.svg
:alt: Temporal coherence and spectra
:width: 80%

Relationship between spectral bandwidth and coherence length. Narrower
spectra (top) yield longer coherence lengths (bottom), while broader
spectra lead to rapid coherence decay.
```

### Spectral Distributions

Janssen supports several spectral models:

| Model | Shape | Application |
|-------|-------|-------------|
| Gaussian | $S(\lambda) \propto \exp[-(\lambda-\lambda_0)^2/2\sigma^2]$ | LED, filtered sources |
| Lorentzian | $S(\lambda) \propto 1/[1+(\lambda-\lambda_0)^2/\gamma^2]$ | Natural linewidth |
| Rectangular | $S(\lambda) \propto \mathrm{rect}[(\lambda-\lambda_0)/\Delta]$ | Bandpass filters |
| Blackbody | $S(\lambda) \propto B_\lambda(T)$ | Thermal sources |

## Coherent Mode Decomposition

### Mercer's Theorem

Any partially coherent field can be decomposed into orthogonal coherent
modes (Mercer's theorem):

$$
J(\mathbf{r}_1, \mathbf{r}_2) = \sum_n \lambda_n \phi_n^*(\mathbf{r}_1) \phi_n(\mathbf{r}_2)
$$

where $\lambda_n$ are eigenvalues and $\phi_n$ are orthonormal eigenmodes.

The total intensity is the **incoherent sum** of mode intensities:

$$
I(\mathbf{r}) = \sum_n \lambda_n |\phi_n(\mathbf{r})|^2
$$

```{figure} figures/coherent_mode_decomposition.svg
:alt: Coherent mode decomposition
:width: 90%

Coherent mode decomposition of a partially coherent beam. The field is
represented as a weighted sum of orthogonal modes $\phi_n$ with weights
$\lambda_n$. Intensities (not amplitudes) are summed.
```

### Effective Mode Count

The **participation ratio** quantifies partial coherence:

$$
N_{\text{eff}} = \frac{(\sum_n \lambda_n)^2}{\sum_n \lambda_n^2}
$$

- $N_{\text{eff}} = 1$: Fully coherent (single mode dominates)
- $N_{\text{eff}} > 1$: Partially coherent (multiple modes contribute)

### Memory Efficiency

Mode decomposition is memory-efficient compared to the full mutual
intensity:

| Representation | Memory | Grid Size Scaling |
|----------------|--------|-------------------|
| `CoherentModeSet` | $O(M \times N^2)$ | Linear in modes |
| `MutualIntensity` | $O(N^4)$ | Quartic |

For a 256x256 grid with 10 modes:
- Mode set: ~5 MB
- Mutual intensity: ~34 GB

## Source Models

### LED Source

LEDs combine spatial incoherence (extended die) with temporal incoherence
(broad spectrum):

```python
from janssen.coherence import led_source

mode_set, wavelengths, spectral_weights = led_source(
    center_wavelength=530e-9,     # Green LED
    bandwidth_fwhm=30e-9,         # 30 nm FWHM
    spatial_coherence_width=50e-6,  # 50 um coherence
    dx=1e-6,
    grid_size=(256, 256),
    num_spatial_modes=10,
    num_spectral_samples=11,
)
```

```{figure} figures/led_vs_laser_modes.svg
:alt: LED vs laser coherent modes
:width: 85%

Comparison of coherent mode structure for laser (left) and LED (right)
sources. The laser is dominated by the fundamental mode, while the LED
requires many modes to represent its partial coherence.
```

### Synchrotron Source

Synchrotron sources have anisotropic coherence (different in horizontal
and vertical due to different source sizes):

```python
from janssen.coherence import synchrotron_source

mode_set = synchrotron_source(
    center_wavelength=1e-10,      # 1 Angstrom X-rays
    horizontal_coherence=10e-6,   # 10 um horizontal
    vertical_coherence=100e-6,    # 100 um vertical
    dx=1e-7,
    grid_size=(256, 256),
)
```

## Propagation of Partially Coherent Fields

### Mode-by-Mode Propagation

Each coherent mode propagates independently. The total intensity is
computed by summing mode intensities at the output:

```python
from janssen.coherence import propagate_coherent_mode_set
from janssen.prop import angular_spectrum

# Propagate partially coherent field
output_modes = propagate_coherent_mode_set(
    mode_set,
    propagation_distance=1e-3,
    propagator_fn=angular_spectrum,
)

# Total output intensity
total_intensity = output_modes.intensity
```

### Polychromatic Propagation

For temporal coherence, propagate at each wavelength and sum:

```python
from janssen.coherence import propagate_polychromatic_wavefront

output = propagate_polychromatic_wavefront(
    polychromatic_wavefront,
    propagation_distance=1e-3,
    propagator_fn=angular_spectrum,
)
```

## References

1. Mandel, L. & Wolf, E. "Optical Coherence and Quantum Optics" (1995)
2. Goodman, J. W. "Statistical Optics" (2015)
3. Starikov, A. & Wolf, E. "Coherent-mode representation of Gaussian
   Schell-model sources" JOSA A (1982)
4. Thibault, P. & Menzel, A. "Reconstructing state mixtures from
   diffraction measurements" Nature (2013)
