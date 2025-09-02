# Package Organization

## Overview

Janssen is a focused library for optical microscopy and ptychography, split from the original Ptyrodactyl project. The package is organized into three main modules: common utilities, simulator for forward models, and invertor for reconstruction algorithms. It follows a clean, hierarchical structure optimized for optical microscopy applications.

## Module Structure

### **janssen.common**
Common utilities and shared data structures used throughout the package.

#### Key Components:
- **Data Types & Structures**
  - Type definitions and common data structures
  - Decorators for JAX transformations
  - Shared utility functions


### **janssen.simulator**
The forward simulation module for optical microscopy, providing differentiable implementations of optical elements and propagation.

#### Key Components:
- **Optical Elements**
  - `apertures.py`: Circular, rectangular, and custom aperture functions
  - `elements.py`: Optical element transformations (beam splitters, waveplates)
  - `lenses.py`: Lens implementations and phase transformations
  - `lens_optics.py`: Physical lens calculations (thickness, phase profiles)

- **Propagation & Simulation**
  - `microscope.py`: Microscopy simulation pipelines
  - `helper.py`: Helper functions for optical propagation
  - Fresnel and angular spectrum propagation methods
  - Wavefront manipulation utilities

### **janssen.invertor**
The reconstruction module containing phase retrieval algorithms and optimization routines.

#### Key Components:
- **Phase Retrieval Algorithms**
  - `ptychography.py`: Ptychographic reconstruction algorithms
  - `engine.py`: Core reconstruction engine
  - Single-slice and multi-slice ptychography
  - Position-corrected algorithms
  - Multi-modal probe reconstruction

- **Optimization**
  - `optimizers.py`: Complex-valued optimizers with Wirtinger derivatives
  - `loss_functions.py`: Loss functions for phase retrieval
  - ADAM, AdaGrad, RMSProp, and SGD implementations
  - Learning rate scheduling

## Design Principles

### 1. **JAX-First Architecture**
All functions are designed to be:
- **Differentiable**: Full support for `jax.grad`
- **JIT-compilable**: Optimized with `jax.jit`
- **Vectorizable**: Compatible with `jax.vmap`
- **Device-agnostic**: Run on CPU, GPU, or TPU

### 2. **Type Safety**
- Comprehensive type hints using `jaxtyping`
- Runtime type checking with `beartype`
- Clear array shape specifications

### 3. **Functional Programming**
- Pure functions without side effects
- Immutable data structures
- Composable operations

### 4. **Optical Focus**
Optimized specifically for optical microscopy:
- Wavelength-dependent calculations
- Complex wavefront representations
- Physical optics simulations

## File Organization

The package structure is organized for clarity and maintainability:

```
src/janssen/
├── __init__.py           # Top-level exports
├── common/
│   ├── __init__.py       # Common module exports
│   ├── types.py          # Shared type definitions
│   └── decorators.py     # JAX decorators and utilities
├── simulator/
│   ├── __init__.py       # Simulator module exports
│   ├── apertures.py      # Aperture functions
│   ├── elements.py       # Optical elements
│   ├── helper.py         # Utility functions
│   ├── lens_optics.py    # Lens calculations
│   ├── lenses.py         # Lens implementations
│   └── microscope.py     # Microscopy simulations
└── invertor/
    ├── __init__.py       # Invertor module exports
    ├── engine.py         # Reconstruction engine
    ├── ptychography.py   # Ptychographic algorithms
    ├── optimizers.py     # Optimization routines
    └── loss_functions.py # Loss function definitions
```

## Import Patterns

### Public API Usage
Users should import from the three main modules:

```python
# Import from main modules
from janssen.simulator import microscope, apertures, lenses
from janssen.invertor import ptychography, optimizers
from janssen.common import types, decorators

# Import entire modules
import janssen.simulator as sim
import janssen.invertor as inv
import janssen.common as common
```

### Internal Implementation
The `__init__.py` files handle internal imports and expose a clean API:

```python
# simulator/__init__.py example
from .apertures import (
    circular_aperture, rectangular_aperture
)
from .elements import (
    apply_lens, apply_aperture, apply_beam_splitter
)
from .microscope import (
    simple_microscope, scanning_microscope
)
# ... etc
```

## Best Practices

### 1. **Use JAX Transformations**
Leverage JAX's powerful transformations:
```python
# JIT compilation for performance
@jax.jit
def simulate(wavefront, sample):
    return microscope.forward_model(wavefront, sample)
```

### 2. **Automatic Differentiation**
```python
# Automatic differentiation for optimization
grad_fn = jax.grad(loss_function)

# Vectorization
batched_simulate = jax.vmap(simulate, in_axes=(0, None))
```ad_fn = jax.grad(loss_function)

# Vectorization
batched_stem = jax.vmap(stem_4d, in_axes=(None, None, 0))
```

### 3. **Type Annotations**
Use type hints for clarity:
```python
from jaxtyping import Float, Complex

def propagate_wavefront(
    field: Complex[Array, "H W"],
    distance: float,
    wavelength: float
) -> Complex[Array, "H W"]:
    return fresnel_propagate(field, distance, wavelength)
```

### 4. **Composable Operations**
Build complex operations from simple functions:
```python
# Compose multiple operations
def full_reconstruction(raw_data, initial_guess):
    # Apply forward model
    simulated = simulator.microscope(initial_guess, probe)
    
    # Reconstruct using ptychography
    result = invertor.ptychography(raw_data, initial_guess)
    
    return result
```

## Performance Considerations

### Memory Management
- Use `jax.checkpoint` for memory-intensive operations
- Leverage `jax.lax.scan` for sequential operations
- Prefer in-place updates with `.at[].set()` for large arrays

### Parallelization
- Use `jax.pmap` for data parallelism
- Implement sharding strategies for large datasets
- Utilize device mesh for distributed computing

### Optimization
- JIT-compile hot paths
- Batch operations with `vmap`
- Use appropriate precision (float32 vs float64)

## Extension Points

The package is designed to be extensible:

1. **Custom Loss Functions**: Implement new loss functions following the pattern in `invertor.loss_functions`
2. **New Optimizers**: Add optimizers with Wirtinger derivative support
3. **Additional Reconstructions**: Build on base reconstruction algorithms in `invertor.ptychography`
4. **Custom Optical Elements**: Add new elements in `simulator.elements`
5. **Custom Workflows**: Combine existing functions for specific use cases

## Dependencies

### Core Dependencies
- **JAX**: Automatic differentiation and JIT compilation
- **NumPy**: Array operations (via JAX)
- **jaxtyping**: Type annotations for JAX arrays
- **beartype**: Runtime type checking

### Optional Dependencies
- **matplotlib**: Visualization (for examples)
- **scipy**: Additional scientific computing tools
- **h5py**: HDF5 file I/O

## Future Directions

The package architecture supports future extensions:
- Advanced ptychographic reconstruction algorithms
- GPU-optimized optical propagation kernels
- Real-time microscopy processing pipelines
- Integration with experimental microscopy data formats
- Machine learning-enhanced phase retrieval
- Adaptive optics simulations
- Coherent diffractive imaging techniques