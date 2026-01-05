# Package Organization

## Overview

Janssen is a typed and tested JAX based library for optical microscopy and ptychography, to utilize two of JAX's capabilities - multi-device computation and autodifferentiation. The package is organized into nine main modules: `types` for PyTree data structures, `utils` for common math utilities, `optics` for optical elements, `prop` for propagation algorithms, `scopes` for microscope forward models, `lenses` for lens implementations, `models` for test pattern generation, `cohere` for coherence modeling, and `invert` for reconstruction algorithms.

## Module Structure

### **janssen.types**

PyTree data structures and factory functions for type-safe JAX programming. Includes `OpticalWavefront`, `CoherentModeSet`, `MixedStatePtychoData`, and other core types.

### **janssen.utils**

Common mathematical utilities including Bessel functions, Fourier shifting, Wirtinger gradients, and distributed computing helpers.

### **janssen.optics**

Optical elements and transformations for simulating light propagation through various optical components.

### **janssen.cohere**

Coherence modeling for partially coherent optical fields. Includes Gaussian Schell-model mode generation, spatial coherence kernels, and coherent mode propagation.

### **janssen.scopes**

Microscope forward models for simulating image formation and diffraction patterns in optical microscopy.

### **janssen.prop**

Propagation algorithms for simulating optical field propagation through different media. Includes free-space propagation methods (angular spectrum, Fresnel, Fraunhofer) and material propagation for volumetric optical media.

### **janssen.lenses**

Lens element definitions and optical calculations for modeling various lens types.

### **janssen.models**

Models for generating test patterns and optical beams for testing and validation.

### **janssen.invert**

The reconstruction module containing phase retrieval algorithms, ptychography, and mixed-state reconstruction for partially coherent sources.

## Design Principles

### 1. **JAX-First Architecture**
All functions are designed to be:
- **Differentiable**: Full support for `jax.grad`
- **JIT-compilable**: Optimized with `jax.jit`
- **Vectorizable**: Compatible with `jax.vmap`

### 2. **Type Safety**
- Type hints using `jaxtyping`
- Runtime type checking with `beartype`
- Using `PyTrees` for data containers
- `PyTrees` are loaded with type-checked factory functions.

## File Organization

The package structure is organized for clarity and maintainability:

```text
src/janssen/
├── __init__.py           # Top-level exports
├── types/
│   ├── __init__.py       # Types module exports
│   ├── scalars.py        # Scalar type definitions
│   ├── wavefronts.py     # OpticalWavefront, PropagatingWavefront
│   ├── coherence_types.py # CoherentModeSet, MixedStatePtychoData
│   └── factory.py        # PyTree factory functions
├── utils/
│   ├── __init__.py       # Utils module exports
│   ├── bessel.py         # Bessel functions (J, K)
│   ├── math.py           # Fourier shift, Wirtinger gradients
│   └── distributed.py    # Multi-device utilities
├── optics/
│   ├── __init__.py       # Optics module exports
│   ├── apertures.py      # Aperture functions
│   ├── elements.py       # Optical elements
│   ├── helper.py         # Utility functions
│   └── zernike.py        # Zernike polynomials
├── cohere/
│   ├── __init__.py       # Coherence module exports
│   ├── modes.py          # Gaussian Schell-model modes
│   ├── spatial.py        # Spatial coherence kernels
│   └── propagation.py    # Coherent mode propagation
├── prop/
│   ├── __init__.py       # Prop module exports
│   ├── free_space_prop.py # Free-space propagation algorithms
│   └── material_prop.py  # Material propagation
├── scopes/
│   ├── __init__.py       # Scopes module exports
│   └── simple_microscopes.py  # Simple microscope forward models
├── models/
│   ├── __init__.py       # Models module exports
│   ├── beams.py          # Optical beam generation
│   └── usaf_pattern.py   # USAF test pattern generation
├── lenses/
│   ├── __init__.py       # Lenses module exports
│   └── lens_elements.py  # Lens element definitions
└── invert/
    ├── __init__.py       # Invert module exports
    ├── engine.py         # PIE reconstruction engine
    ├── ptychography.py   # Ptychographic algorithms
    ├── mixed_state.py    # Mixed-state ptychography
    ├── initialization.py # Reconstruction initialization
    └── loss_functions.py # Loss function definitions
```

## Extension Points

The package is designed to be extensible:

1. **Custom Loss Functions**: Implement new loss functions following the pattern in `invert.loss_functions`
2. **New Optimizers**: Add optimizers with Wirtinger derivative support
3. **Additional Reconstructions**: Build on base reconstruction algorithms in `invert.ptychography`
4. **Custom Optical Elements**: Add new elements in `optics.elements`
5. **Custom Workflows**: Combine existing functions for specific use cases

## Future Directions

The package architecture supports future extensions:

- Real-time microscopy inversion
- Adaptive optics simulations
- Non-linear optics
