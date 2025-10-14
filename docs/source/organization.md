# Package Organization

## Overview

Janssen is a typed and tested JAX based library for optical microscopy and ptychography, to utilize two of JAX's capabilities - multi-device computation and autodifferentiation. The package is organized into six main modules: `utils` for common utilities, `optics` for optical elements, `scopes` for microscope forward models, `lenses` for lens implementations, `models` for test pattern generation, and `invert` for reconstruction algorithms.

## Module Structure

### **janssen.utils**

Common utilities and shared data structures used throughout the package.

### **janssen.optics**

Optical elements and transformations for simulating light propagation through various optical components.

### **janssen.scopes**

Microscope forward models for simulating image formation and diffraction patterns in optical microscopy.

### **janssen.lenses**

Dedicated module for lens implementations and optical calculations.

### **janssen.models**

Models for generating test patterns and datasets for testing and validation.

### **janssen.invert**

The reconstruction module containing phase retrieval algorithms and optimization routines.


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

```
src/janssen/
├── __init__.py           # Top-level exports
├── utils/
│   ├── __init__.py       # Utils module exports
│   ├── types.py          # Shared type definitions
│   └── decorators.py     # JAX decorators and utilities
├── optics/
│   ├── __init__.py       # Optics module exports
│   ├── apertures.py      # Aperture functions
│   ├── elements.py       # Optical elements
│   ├── helper.py         # Utility functions
│   └── zernike.py        # Zernike polynomials
├── scopes/
│   ├── __init__.py       # Scopes module exports
│   └── simple_microscopes.py  # Simple microscope forward models
├── models/
│   ├── __init__.py       # Models module exports
│   └── usaf_pattern.py   # USAF test pattern generation
├── lenses/
│   ├── __init__.py       # Lenses module exports
│   ├── lens_optics.py    # Lens calculations
│   └── lenses.py         # Lens implementations
└── invert/
    ├── __init__.py       # Invert module exports
    ├── engine.py         # Reconstruction engine
    ├── ptychography.py   # Ptychographic algorithms
    ├── optimizers.py     # Optimization routines
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
- Non-linear optics.