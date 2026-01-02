# High-NA Vector Focusing

At high numerical apertures (NA > 0.7), scalar diffraction theory breaks
down. Janssen implements the Richards-Wolf vector diffraction integrals
for accurate focal field calculations.

## Beyond the Paraxial Approximation

### Why Vector Optics?

At high NA:

1. **Depolarization**: Light can have significant $E_z$ components
2. **Apodization**: Fresnel transmission varies across the aperture
3. **Vectorial focusing**: Electric field direction matters

```{figure} figures/debye_wolf_geometry.svg
:alt: Richards-Wolf integral geometry
:width: 85%

Geometry for Richards-Wolf integrals. Light converges from the pupil
plane to the focal region. The integration is over all angles up to
$\theta_{\max} = \arcsin(\text{NA})$.
```

### Numerical Aperture Effects

| NA | $\theta_{\max}$ | $E_z/E_x$ (radial pol.) | Notes |
|----|-----------------|-------------------------|-------|
| 0.5 | 30° | ~15% | Paraxial still reasonable |
| 0.7 | 44° | ~30% | Significant vector effects |
| 0.9 | 64° | ~60% | Strong $E_z$ component |
| 1.4 (oil) | 68° | ~75% | Immersion objective |

## Richards-Wolf Diffraction Integrals

### Mathematical Formulation

The electric field near focus is:

$$
\vec{E}(\rho_f, \phi_f, z_f) = -\frac{ikf}{2\pi}
\int_0^{\theta_{\max}} \int_0^{2\pi}
\sqrt{\cos\theta} \, \mathbf{P}(\theta, \phi)
\cdot \vec{E}_{\text{pupil}}(\theta, \phi)
$$

$$
\times e^{ikz_f\cos\theta} e^{ik\rho_f\sin\theta\cos(\phi-\phi_f)}
\sin\theta \, d\phi \, d\theta
$$

where:

- $f$ is the focal length
- $\theta$ is the convergence angle
- $\mathbf{P}(\theta, \phi)$ is the polarization rotation matrix
- $\vec{E}_{\text{pupil}}$ is the pupil field

### The Polarization Matrix

The matrix $\mathbf{P}$ accounts for how the electric field rotates
as light refracts through the lens:

$$
\mathbf{P} = \begin{pmatrix}
\cos\theta\cos^2\phi + \sin^2\phi &
(\cos\theta - 1)\cos\phi\sin\phi &
-\sin\theta\cos\phi \\
(\cos\theta - 1)\cos\phi\sin\phi &
\cos\theta\sin^2\phi + \cos^2\phi &
-\sin\theta\sin\phi \\
\sin\theta\cos\phi &
\sin\theta\sin\phi &
\cos\theta
\end{pmatrix}
$$

## Focal Field Components

### Three-Component Electric Field

At high NA, all three components $(E_x, E_y, E_z)$ can be significant:

```{figure} figures/focal_field_components.svg
:alt: Focal field components
:width: 90%

Electric field components at the focal plane for x-polarized input
at NA=0.9. (a) $|E_x|^2$ dominates, (b) $|E_y|^2$ appears at corners,
(c) $|E_z|^2$ has two lobes along polarization direction.
```

### Implementation

```python
from janssen.prop import vector_focusing

# Calculate 3D focal field
focal_field = vector_focusing(
    pupil_field=input_field,
    numerical_aperture=0.9,
    focal_length=3e-3,
    wavelength=632.8e-9,
    focal_grid_size=(64, 64, 32),  # (x, y, z)
    focal_extent=(2e-6, 2e-6, 4e-6),  # Physical size
)

# Access field components
Ex = focal_field.Ex  # Shape: (64, 64, 32)
Ey = focal_field.Ey
Ez = focal_field.Ez

# Total intensity
I_total = jnp.abs(Ex)**2 + jnp.abs(Ey)**2 + jnp.abs(Ez)**2
```

## Polarization Effects

### Input Polarization States

Different input polarizations create different focal distributions:

```{figure} figures/polarization_modes.svg
:alt: Polarization effects at focus
:width: 90%

Focal intensity distributions for different input polarizations at NA=0.9.
(a) Linear x: elongated along x, (b) circular: rotationally symmetric,
(c) radial: strong $E_z$ and donut shape, (d) azimuthal: donut with
only transverse field.
```

| Polarization | Focal Shape | $E_z$ Content | Application |
|--------------|-------------|---------------|-------------|
| Linear | Elongated | Moderate | Standard imaging |
| Circular | Symmetric | Moderate | Isotropic resolution |
| Radial | Donut + $E_z$ | Strong | STED, $z$-sensitive |
| Azimuthal | Donut | Zero | Transverse field only |

### Implementation

```python
from janssen.optics import polarizer_jones

# Create radially polarized pupil field
radial_pupil = create_radial_polarization(
    grid_size=(256, 256),
    pupil_radius=1.0,
)

# Or convert from scalar field
from janssen.models import vortex_beam

scalar_field = vortex_beam(
    wavelength=632.8e-9,
    grid_size=(256, 256),
    dx=10e-6,
    topological_charge=1,
)

# Apply polarization
polarized_field = polarizer_jones(
    field=scalar_field,
    jones_vector=[1.0, 0.0],  # x-polarized
)
```

## Apodization

### Aplanatic Condition

For an aplanatic (aberration-free) lens, the apodization function is:

$$
A(\theta) = \sqrt{\cos\theta}
$$

This appears naturally in the Richards-Wolf integrals and accounts for
the Fresnel transmission coefficients.

```{figure} figures/apodization_effects.svg
:alt: Apodization effects
:width: 80%

Effect of apodization on focal spot. (a) Uniform apodization creates
side lobes, (b) aplanatic $\sqrt{\cos\theta}$ is standard, (c) Gaussian
apodization reduces side lobes but broadens spot.
```

### Custom Apodization

```python
from janssen.optics import gaussian_apodizer

# Apply Gaussian apodization for side lobe reduction
apodized_pupil = gaussian_apodizer(
    pupil_field=input_field,
    fill_factor=0.8,  # Gaussian width relative to pupil
)
```

## Practical Considerations

### Sampling Requirements

The focal field requires sufficient sampling in both pupil and focal
planes:

- **Pupil**: $\Delta\theta \lesssim \lambda / (10 \cdot \text{spot size})$
- **Focal**: $\Delta x \lesssim \lambda / (4 \cdot \text{NA})$

### Computational Cost

Richards-Wolf integration is computationally intensive:

| Grid Size | Evaluation Points | Time (GPU) |
|-----------|-------------------|------------|
| 64×64×32 | 131,072 | ~0.5 s |
| 128×128×64 | 1,048,576 | ~2 s |
| 256×256×128 | 8,388,608 | ~15 s |

JAX's JIT compilation and GPU acceleration are essential for practical
use.

### When to Use Vector vs Scalar

| NA Range | Method | Notes |
|----------|--------|-------|
| < 0.3 | Scalar | Paraxial approximation valid |
| 0.3-0.6 | Either | Scalar may suffice for intensity |
| > 0.6 | Vector | Required for accurate results |
| > 1.0 | Vector + immersion | Account for medium |

## References

1. Richards, B. & Wolf, E. "Electromagnetic diffraction in optical
   systems, II" Proc. R. Soc. A (1959)
2. Novotny, L. & Hecht, B. "Principles of Nano-Optics" (2012)
3. Born, M. & Wolf, E. "Principles of Optics" Chapter 8 (1999)
