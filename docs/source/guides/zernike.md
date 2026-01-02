# Zernike Polynomials and Optical Aberrations

Zernike polynomials are the standard basis for describing wavefront
aberrations in optical systems. Janssen provides JAX-compatible
implementations for generating and analyzing aberrations.

## Mathematical Definition

### Zernike Polynomials

Zernike polynomials are defined on the unit disk ($\rho \leq 1$):

$$
Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \cdot
\begin{cases}
\cos(m\theta) & m \geq 0 \\
\sin(|m|\theta) & m < 0
\end{cases}
$$

where $R_n^{|m|}(\rho)$ is the radial polynomial:

$$
R_n^{|m|}(\rho) = \sum_{s=0}^{(n-|m|)/2} \frac{(-1)^s (n-s)!}
{s! \left(\frac{n+|m|}{2}-s\right)! \left(\frac{n-|m|}{2}-s\right)!}
\rho^{n-2s}
$$

```{figure} figures/zernike_pyramid.svg
:alt: Zernike polynomial pyramid
:width: 100%

First 15 Zernike modes arranged by radial order $n$ (rows) and
azimuthal frequency $m$ (columns). Each mode represents a distinct
aberration pattern.
```

### Orthogonality

Zernike polynomials are orthogonal over the unit disk:

$$
\int_0^{2\pi} \int_0^1 Z_n^m(\rho, \theta) Z_{n'}^{m'}(\rho, \theta)
\rho \, d\rho \, d\theta = \pi \epsilon_m \delta_{nn'} \delta_{mm'}
$$

where $\epsilon_m = 2$ for $m = 0$ and $\epsilon_m = 1$ otherwise.

## Indexing Conventions

### Noll Indexing

The Noll index $j$ orders modes by radial degree, then azimuthal:

| $j$ | $n$ | $m$ | Name |
|-----|-----|-----|------|
| 1 | 0 | 0 | Piston |
| 2 | 1 | 1 | Tip |
| 3 | 1 | -1 | Tilt |
| 4 | 2 | 0 | Defocus |
| 5 | 2 | -2 | Oblique astigmatism |
| 6 | 2 | 2 | Vertical astigmatism |
| 7 | 3 | -1 | Vertical coma |
| 8 | 3 | 1 | Horizontal coma |
| 9 | 3 | -3 | Vertical trefoil |
| 10 | 3 | 3 | Oblique trefoil |
| 11 | 4 | 0 | Primary spherical |

```{figure} figures/noll_indexing.svg
:alt: Noll indexing diagram
:width: 75%

Noll index ordering on the $(n, m)$ grid. The index increases left-to-
right and bottom-to-top, with alternating sign for $m \neq 0$.
```

### Converting Between Indices

```python
from janssen.optics import noll_to_nm, nm_to_noll

# Noll index 11 -> (n=4, m=0) - spherical aberration
n, m = noll_to_nm(11)
print(f"Noll 11 = (n={n}, m={m})")

# (n=3, m=1) -> Noll index 8 - horizontal coma
j = nm_to_noll(3, 1)
print(f"(n=3, m=1) = Noll {j}")
```

## Common Aberrations

### Defocus (Z4)

Defocus represents axial displacement from the focal plane:

$$
W_{\text{defocus}}(\rho) = c_4 (2\rho^2 - 1)
$$

Positive coefficient = focus behind detector.

```{figure} figures/common_aberrations.svg
:alt: Common optical aberrations
:width: 90%

Common aberrations: (a) defocus, (b) astigmatism, (c) coma,
(d) spherical aberration. Color indicates phase deviation from
flat wavefront.
```

### Astigmatism (Z5, Z6)

Astigmatism causes different focal lengths along perpendicular axes:

$$
W_{\text{astig}}(\rho, \theta) = c_5 \rho^2 \sin(2\theta) +
c_6 \rho^2 \cos(2\theta)
$$

### Coma (Z7, Z8)

Coma creates comet-shaped blur for off-axis points:

$$
W_{\text{coma}}(\rho, \theta) = c_7 (3\rho^3 - 2\rho) \sin\theta +
c_8 (3\rho^3 - 2\rho) \cos\theta
$$

### Spherical Aberration (Z11)

Spherical aberration causes focus shift that depends on pupil radius:

$$
W_{\text{sph}}(\rho) = c_{11} (6\rho^4 - 6\rho^2 + 1)
$$

## Generating Aberrations

### From Noll Coefficients

```python
from janssen.optics import generate_aberration_noll

# Generate phase map with 0.1 waves of defocus and 0.05 waves of coma
coefficients = {
    4: 0.1,   # Defocus
    8: 0.05,  # Horizontal coma
}

phase_map = generate_aberration_noll(
    coefficients=coefficients,
    grid_size=(256, 256),
    pupil_radius=1.0,
    wavelength=632.8e-9,
)
```

### From (n, m) Coefficients

```python
from janssen.optics import generate_aberration_nm

# Same aberrations using (n, m) indexing
coefficients = {
    (2, 0): 0.1,   # Defocus
    (3, 1): 0.05,  # Horizontal coma
}

phase_map = generate_aberration_nm(
    coefficients=coefficients,
    grid_size=(256, 256),
    pupil_radius=1.0,
    wavelength=632.8e-9,
)
```

## Applying Aberrations

### To a Wavefront

```python
from janssen.optics import apply_aberration

# Apply aberration to existing wavefront
aberrated = apply_aberration(
    wavefront=input_wavefront,
    coefficients={4: 0.1, 11: 0.02},
    pupil_radius=aperture_radius,
)
```

### Effect on PSF

```{figure} figures/aberration_effects.svg
:alt: Effect of aberrations on PSF
:width: 90%

Effect of aberrations on point spread function. (a) Perfect, (b) defocus
broadens symmetrically, (c) astigmatism creates elliptical blur,
(d) coma creates asymmetric comet shape, (e) spherical creates halo.
```

## Phase Analysis

### RMS Wavefront Error

The RMS wavefront error quantifies aberration severity:

$$
\sigma_W = \sqrt{\frac{1}{\pi} \iint_{pupil} W^2(\rho, \theta)
\rho \, d\rho \, d\theta}
$$

```python
from janssen.optics import phase_rms

rms_error = phase_rms(
    phase_map=phase,
    pupil_mask=pupil,
)
print(f"RMS error: {rms_error:.4f} waves")
```

### Strehl Ratio

For small aberrations, the Strehl ratio is:

$$
S \approx \exp\left(-\left(\frac{2\pi \sigma_W}{\lambda}\right)^2\right)
$$

The Maréchal criterion $S > 0.8$ requires $\sigma_W < \lambda/14$.

## Zernike Decomposition

### Fitting Coefficients

Given a measured wavefront, decompose into Zernike modes:

```python
from janssen.optics import zernike_decomposition

# Fit first 15 Zernike modes
coefficients = zernike_decomposition(
    phase_map=measured_phase,
    pupil_mask=pupil,
    num_modes=15,
)

# coefficients[j] is the coefficient for Noll index j
print(f"Defocus: {coefficients[4]:.4f} waves")
print(f"Spherical: {coefficients[11]:.4f} waves")
```

### Removing Low-Order Aberrations

Often piston, tip, and tilt are not of interest:

```python
from janssen.optics import compute_phase_from_coeffs

# Reconstruct phase without first 3 modes
high_order_phase = compute_phase_from_coeffs(
    coefficients=coefficients,
    grid_size=(256, 256),
    start_index=4,  # Skip piston, tip, tilt
)
```

## References

1. Noll, R. J. "Zernike polynomials and atmospheric turbulence"
   JOSA (1976)
2. Born, M. & Wolf, E. "Principles of Optics" Chapter 9 (1999)
3. Mahajan, V. N. "Zernike circle polynomials and optical aberrations
   of systems with circular pupils" Appl. Opt. (1994)
