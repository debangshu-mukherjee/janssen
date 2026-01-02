# Ptychographic Phase Retrieval

Ptychography is a computational imaging technique that recovers both
amplitude and phase of a sample from a series of diffraction patterns.
Janssen implements both iterative (ePIE) and gradient-based algorithms.

## The Ptychography Principle

### Scanning Geometry

A localized probe illuminates the sample at overlapping positions.
At each position $\mathbf{r}_j$, we measure the far-field diffraction
intensity:

$$
I_j(\mathbf{q}) = |\mathcal{F}\{P(\mathbf{r}) \cdot O(\mathbf{r} - \mathbf{r}_j)\}|^2
$$

where $P$ is the probe and $O$ is the sample (object) transmission.

```{figure} figures/ptychography_geometry.svg
:alt: Ptychography scanning geometry
:width: 85%

Ptychography scanning geometry. The probe (red) scans across the sample
with overlap between adjacent positions. Each position yields one
diffraction pattern.
```

### The Overlap Constraint

Overlapping illumination provides **redundant information**:

- Each sample region is measured multiple times
- Different probe positions "see" the same sample differently
- This redundancy enables unique phase retrieval

Typical overlap: 60-80% of probe diameter.

## The ePIE Algorithm

### Extended Ptychographic Iterative Engine

ePIE alternates between object and probe updates:

```{figure} figures/epie_algorithm.svg
:alt: ePIE algorithm diagram
:width: 90%

ePIE algorithm workflow. For each scan position: (1) form exit wave,
(2) propagate to detector, (3) apply magnitude constraint, (4) back-
propagate, (5) update object and probe.
```

### Update Equations

For each scan position $j$:

1. **Exit wave**: $\psi_j = P \cdot O_j$
2. **Propagate**: $\Psi_j = \mathcal{F}\{\psi_j\}$
3. **Constraint**: $\Psi'_j = \sqrt{I_j^{\text{meas}}} \cdot \frac{\Psi_j}{|\Psi_j|}$
4. **Back-propagate**: $\psi'_j = \mathcal{F}^{-1}\{\Psi'_j\}$
5. **Update difference**: $\Delta\psi_j = \psi'_j - \psi_j$

Object update:

$$
O \leftarrow O + \alpha \frac{P^*}{|P|^2_{\max}} \cdot \Delta\psi
$$

Probe update:

$$
P \leftarrow P + \beta \frac{O^*}{|O|^2_{\max}} \cdot \Delta\psi
$$

### Implementation

```python
from janssen.invert import epie_reconstruction

result = epie_reconstruction(
    diffraction_patterns=patterns,  # Shape: (N_positions, H, W)
    scan_positions=positions,       # Shape: (N_positions, 2)
    probe_guess=initial_probe,
    num_iterations=100,
    alpha=1.0,  # Object step size
    beta=1.0,   # Probe step size
)

# Access reconstructions
sample = result.object
probe = result.probe
```

## Gradient-Based Optimization

### Loss Function

Janssen supports gradient-based reconstruction using automatic
differentiation. The loss function is:

$$
\mathcal{L} = \sum_j \sum_{\mathbf{q}} \left(
\sqrt{I_j^{\text{meas}}(\mathbf{q})} -
\sqrt{I_j^{\text{model}}(\mathbf{q})}
\right)^2
$$

This **amplitude loss** is more stable than intensity loss.

```{figure} figures/gradient_descent_comparison.svg
:alt: ePIE vs gradient descent convergence
:width: 80%

Convergence comparison: ePIE (blue) vs gradient descent (orange).
Gradient descent can be slower per-iteration but allows flexible
regularization and multi-modal probes.
```

### Advantages of Gradient-Based Methods

| Feature | ePIE | Gradient-Based |
|---------|------|----------------|
| Speed per iteration | Fast | Slower |
| Regularization | Limited | Flexible |
| Multi-modal probes | Difficult | Natural |
| Automatic differentiation | No | Yes |
| GPU acceleration | Manual | Via JAX |

### Implementation

```python
from janssen.invert import gradient_ptychography
import optax

# Define optimizer
optimizer = optax.adam(learning_rate=1e-3)

result = gradient_ptychography(
    diffraction_patterns=patterns,
    scan_positions=positions,
    probe_guess=initial_probe,
    optimizer=optimizer,
    num_iterations=1000,
    loss_fn="amplitude",  # or "intensity", "poisson"
)
```

## Loss Functions

### Available Options

| Loss | Formula | Best For |
|------|---------|----------|
| Amplitude | $\|\sqrt{I_{\text{meas}}} - \sqrt{I_{\text{model}}}\|^2$ | General use |
| Intensity | $\|I_{\text{meas}} - I_{\text{model}}\|^2$ | High SNR |
| Poisson | $I_{\text{model}} - I_{\text{meas}} \log I_{\text{model}}$ | Low photon count |

```{figure} figures/reconstruction_convergence.svg
:alt: Loss function convergence curves
:width: 80%

Convergence curves for different loss functions on the same dataset.
Amplitude loss (blue) typically converges faster and more stably than
intensity loss (orange) or Poisson loss (green).
```

## Initialization Strategies

### Probe Initialization

Good probe initialization accelerates convergence:

- **Known probe**: Use measured or simulated probe
- **Gaussian**: Start with Gaussian of expected size
- **From data**: Estimate from central diffraction pattern

### Object Initialization

- **Unity**: Start with $O = 1$ (transparent sample)
- **Random phase**: $O = \exp(i\phi_{\text{rand}})$
- **Prior knowledge**: Use expected transmission

```python
from janssen.invert import initialize_probe, initialize_object

probe = initialize_probe(
    method="gaussian",
    size=(128, 128),
    width=20e-6,
    wavelength=632.8e-9,
)

obj = initialize_object(
    size=(512, 512),
    method="unity",
)
```

## Practical Considerations

### Scan Position Refinement

Real experiments have position errors. Janssen can refine positions:

```python
result = epie_reconstruction(
    ...,
    refine_positions=True,
    position_step_size=0.1,
)
```

### Partial Coherence

For partially coherent sources, use multiple probe modes:

```python
from janssen.invert import multimode_ptychography

result = multimode_ptychography(
    diffraction_patterns=patterns,
    scan_positions=positions,
    num_probe_modes=5,  # Number of coherent modes
)
```

### Memory Management

Large reconstructions benefit from batched processing:

```python
result = gradient_ptychography(
    ...,
    batch_size=32,  # Process 32 positions per gradient step
)
```

## Diagnostics

### Convergence Monitoring

```python
# Access iteration history
losses = result.loss_history
probe_updates = result.probe_update_norms

# Check convergence
import matplotlib.pyplot as plt
plt.semilogy(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
```

### Error Metrics

- **R-factor**: $R = \sum|\sqrt{I_{\text{meas}}} - \sqrt{I_{\text{model}}}| / \sum\sqrt{I_{\text{meas}}}$
- **SSIM**: Structural similarity for object comparison

## References

1. Rodenburg, J. M. & Faulkner, H. M. L. "A phase retrieval algorithm
   for shifting illumination" Appl. Phys. Lett. (2004)
2. Maiden, A. M. & Rodenburg, J. M. "An improved ptychographical phase
   retrieval algorithm for diffractive imaging" Ultramicroscopy (2009)
3. Thibault, P. et al. "High-resolution scanning X-ray diffraction
   microscopy" Science (2008)
