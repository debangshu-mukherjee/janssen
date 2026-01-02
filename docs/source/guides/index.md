# Janssen Theory and Architecture Guides

This documentation provides comprehensive coverage of the physics and
software architecture underlying janssen, a JAX-based framework for
optical microscopy simulations and ptychographic reconstruction.

## Target Audience

These guides are written for **optics researchers** who want to understand:

- The mathematical foundations of coherent and partially coherent optics
- How optical wavefronts propagate through the simulation pipeline
- The physical meaning of simulation parameters and outputs

## Guide Overview

### Physics Foundations

| Guide | Description |
|-------|-------------|
| [Partial Coherence](coherence.md) | Spatial and temporal coherence, mode decomposition |
| [Optical Propagation](propagation.md) | Fresnel, Fraunhofer, and Angular Spectrum methods |
| [Ptychography](ptychography.md) | Phase retrieval algorithms (ePIE, gradient-based) |
| [Zernike Polynomials](zernike.md) | Optical aberrations and wavefront decomposition |
| [Vector Optics](vector-optics.md) | High-NA focusing with Richards-Wolf integrals |

### Architecture

| Guide | Description |
|-------|-------------|
| [PyTree Architecture](pytree-architecture.md) | JAX data structures for GPU acceleration |

## Mathematical Notation

Throughout these guides, we use:

- $\lambda$ for wavelength (in meters)
- $k = 2\pi/\lambda$ for wavenumber
- $\mathbf{E}$ for electric field
- $I = |\mathbf{E}|^2$ for intensity
- $\phi$ for phase

```{toctree}
:maxdepth: 2
:hidden:

coherence
propagation
ptychography
zernike
vector-optics
pytree-architecture
```
