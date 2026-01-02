# Optical Propagation Methods

Janssen implements several propagation methods for simulating light
propagation through optical systems. This guide covers the mathematical
foundations and when to use each method.

## Scalar Diffraction Theory

### The Helmholtz Equation

In free space, monochromatic light satisfies the Helmholtz equation:

$$
(\nabla^2 + k^2) E(\mathbf{r}) = 0
$$

where $k = 2\pi/\lambda$ is the wavenumber. Solutions describe how the
electric field propagates from one plane to another.

### The Angular Spectrum

Any field $E(x, y, z=0)$ can be decomposed into plane waves:

$$
E(x, y, 0) = \iint A(k_x, k_y) e^{i(k_x x + k_y y)} \, dk_x \, dk_y
$$

where $A(k_x, k_y) = \mathcal{F}\{E(x, y, 0)\}$ is the angular spectrum.

## Propagation Regimes

### The Fresnel Number

The Fresnel number determines which propagation regime applies:

$$
N_F = \frac{a^2}{\lambda z}
$$

where $a$ is the aperture size and $z$ is the propagation distance.

```{figure} figures/propagation_regimes.svg
:alt: Propagation regimes diagram
:width: 85%

Propagation regimes as a function of Fresnel number. Near-field (Fresnel)
diffraction applies when $N_F > 1$, while far-field (Fraunhofer)
diffraction applies when $N_F \ll 1$.
```

| Regime | Fresnel Number | Method |
|--------|----------------|--------|
| Near-field | $N_F > 1$ | Angular spectrum, Fresnel |
| Intermediate | $N_F \approx 1$ | Angular spectrum |
| Far-field | $N_F \ll 1$ | Fraunhofer, angular spectrum |

## Angular Spectrum Method

### Mathematical Foundation

After propagation by distance $z$, the angular spectrum becomes:

$$
A(k_x, k_y; z) = A(k_x, k_y; 0) \cdot e^{i k_z z}
$$

where $k_z = \sqrt{k^2 - k_x^2 - k_y^2}$ for propagating waves.

The propagated field is:

$$
E(x, y, z) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{E(x,y,0)\} \cdot H(k_x, k_y) \right\}
$$

where $H(k_x, k_y) = e^{i k_z z}$ is the transfer function.

```{figure} figures/angular_spectrum_method.svg
:alt: Angular spectrum propagation
:width: 80%

Angular spectrum propagation. The field is transformed to k-space,
multiplied by the transfer function $H(k_x, k_y)$, then transformed back.
```

### Implementation

```python
from janssen.prop import angular_spectrum

# Propagate field by 1 mm
output = angular_spectrum(
    input_wavefront,
    propagation_distance=1e-3,
)
```

### Evanescent Wave Handling

For $k_x^2 + k_y^2 > k^2$, the wave is evanescent with:

$$
k_z = i\sqrt{k_x^2 + k_y^2 - k^2}
$$

Janssen automatically handles evanescent waves by setting them to zero
(they decay exponentially and don't contribute at macroscopic distances).

## Fresnel Propagation

### The Fresnel Approximation

For paraxial beams where $k_x, k_y \ll k$:

$$
k_z \approx k - \frac{k_x^2 + k_y^2}{2k}
$$

This gives the Fresnel transfer function:

$$
H_F(k_x, k_y) = e^{ikz} \exp\left(-i\frac{z}{2k}(k_x^2 + k_y^2)\right)
$$

### Convolution Form

The Fresnel integral can also be written as a convolution:

$$
E(x, y, z) = E(x, y, 0) * h(x, y, z)
$$

where the impulse response is:

$$
h(x, y, z) = \frac{e^{ikz}}{i\lambda z} \exp\left(\frac{ik(x^2+y^2)}{2z}\right)
$$

```{figure} figures/fresnel_number_diagram.svg
:alt: Fresnel number and validity
:width: 80%

Validity of Fresnel approximation. Errors become significant when
$N_F < 0.1$ (transition to Fraunhofer) or when aperture angles exceed
the paraxial limit.
```

### Implementation

```python
from janssen.prop import fresnel_propagation

output = fresnel_propagation(
    input_wavefront,
    propagation_distance=10e-3,
)
```

## Fraunhofer Propagation

### The Far-Field Limit

When $N_F \ll 1$, the field becomes a scaled Fourier transform:

$$
E(x, y, z) = \frac{e^{ikz}}{i\lambda z} e^{ik(x^2+y^2)/2z}
\mathcal{F}\{E\}\left(\frac{x}{\lambda z}, \frac{y}{\lambda z}\right)
$$

### Relationship to Lens Focusing

A lens performs a Fourier transform at its focal plane. For focal
length $f$:

$$
E(x_f, y_f) \propto \mathcal{F}\{E_{\text{pupil}}\}
\left(\frac{x_f}{\lambda f}, \frac{y_f}{\lambda f}\right)
$$

## Lens Propagation

### Thin Lens Model

A thin lens introduces a quadratic phase:

$$
t_{\text{lens}}(x, y) = \exp\left(-\frac{ik(x^2+y^2)}{2f}\right)
$$

```{figure} figures/lens_propagation.svg
:alt: Lens propagation model
:width: 85%

Thin lens propagation model. Light propagates to the lens (distance $d_1$),
passes through (quadratic phase), then propagates to the image plane
(distance $d_2$).
```

### Implementation

```python
from janssen.prop import lens_propagation

output = lens_propagation(
    input_wavefront,
    focal_length=10e-3,
    object_distance=20e-3,
)
```

## Choosing the Right Method

### Decision Guide

1. **Angular Spectrum**: Most general, works at all distances
   - Best for: near-field, high-NA systems, tight focusing
   - Limitation: requires sufficient sampling (Nyquist)

2. **Fresnel**: Efficient for moderate distances
   - Best for: paraxial systems, imaging
   - Limitation: breaks down at high NA

3. **Fraunhofer**: Fastest for far-field
   - Best for: focal plane calculations, far-field patterns
   - Limitation: only valid when $N_F \ll 1$

### Sampling Requirements

The angular spectrum method requires:

$$
\Delta x < \frac{\lambda z}{L}
$$

where $L$ is the field of view. Janssen automatically checks sampling
and warns if insufficient.

## Numerical Considerations

### Periodic Boundary Conditions

FFT-based methods assume periodic boundaries. To avoid wraparound
artifacts:

- Pad arrays to at least $2\times$ the feature size
- Use absorbing boundaries for long propagations

### Precision

Janssen uses 64-bit complex arithmetic by default for phase accuracy.
For short propagations, 32-bit may suffice.

## References

1. Goodman, J. W. "Introduction to Fourier Optics" (2017)
2. Born, M. & Wolf, E. "Principles of Optics" (1999)
3. Voelz, D. G. "Computational Fourier Optics: A MATLAB Tutorial" (2011)
