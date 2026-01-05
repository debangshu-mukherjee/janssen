# PyTree Architecture for JAX Compatibility

Janssen uses JAX PyTrees as the foundation for all data structures.
This enables automatic differentiation, JIT compilation, and parallel
execution across the entire library.

## What is a PyTree?

### Definition

A **PyTree** is a nested structure of containers (lists, tuples, dicts)
with arbitrary leaves. JAX can automatically traverse, transform, and
differentiate through PyTrees.

```python
# Examples of PyTrees
simple = [1, 2, 3]                    # List of scalars
nested = {"a": [1, 2], "b": (3, 4)}   # Dict with list and tuple
arrays = (jnp.array([1, 2]), jnp.array([3, 4]))  # Tuple of arrays
```

### Why PyTrees?

1. **Composability**: Combine multiple arrays into one logical unit
2. **Transformation**: Apply `jit`, `grad`, `vmap` to entire structures
3. **Parallelism**: Automatic vectorization over batches
4. **Type safety**: Preserve structure through computations

```{figure} figures/pytree_hierarchy.svg
:alt: Janssen PyTree hierarchy
:width: 90%

Janssen's PyTree hierarchy. Core types like `OpticalWavefront` and
`CoherentModeSet` are registered as PyTrees, enabling seamless use
with JAX transformations.
```

## Core Data Types

### OpticalWavefront

The fundamental type for representing optical fields:

```python
from janssen.utils import OpticalWavefront

wavefront = OpticalWavefront(
    field=complex_field,      # Complex[Array, "H W"] or [H W 2]
    wavelength=632.8e-9,      # Scalar wavelength
    dx=1e-6,                  # Pixel spacing
    z_position=0.0,           # Axial position
    polarization=False,       # Scalar (False) or Jones (True)
)
```

```{figure} figures/wavefront_types.svg
:alt: Wavefront type variants
:width: 85%

Wavefront type variants. (a) Scalar wavefront with complex field,
(b) polarized wavefront with Jones vector at each pixel.
```

### PropagatingWavefront

Extends `OpticalWavefront` with propagation metadata:

```python
from janssen.utils import PropagatingWavefront

prop_wf = PropagatingWavefront(
    field=complex_field,
    wavelength=632.8e-9,
    dx=1e-6,
    z_position=0.0,
    polarization=False,
    source_distance=float("inf"),  # From plane wave
    propagation_method="angular_spectrum",
)
```

### CoherentModeSet

For partially coherent fields (see [Coherence Guide](coherence.md)):

```python
from janssen.utils import CoherentModeSet

modes = CoherentModeSet(
    modes=mode_array,         # Complex[Array, "M H W"]
    weights=weight_array,     # Float[Array, "M"]
    wavelength=632.8e-9,
    dx=1e-6,
    z_position=0.0,
    polarization=False,
)

# Access properties
print(f"Number of modes: {modes.num_modes}")
print(f"Effective modes: {modes.effective_mode_count}")
```

```{figure} figures/coherence_types.svg
:alt: Coherence type hierarchy
:width: 85%

Coherence types. `CoherentModeSet` stores multiple modes with weights,
while `PolychromaticWavefront` stores fields at multiple wavelengths.
```

## Factory Functions

### Why Factories?

Factory functions provide **validation and normalization** before
creating PyTree instances:

```python
from janssen.utils import make_optical_wavefront

# Factory validates inputs
wavefront = make_optical_wavefront(
    field=complex_field,
    wavelength=632.8e-9,
    dx=1e-6,
)
# - Converts to correct dtype
# - Validates shapes
# - Sets defaults
# - Normalizes parameters
```

```{figure} figures/factory_pattern.svg
:alt: Factory function pattern
:width: 80%

Factory function workflow. Raw inputs are validated, converted to
JAX arrays, normalized, and assembled into a registered PyTree.
```

### Available Factories

| Factory | Creates | Validations |
|---------|---------|-------------|
| `make_optical_wavefront` | `OpticalWavefront` | Shape, dtype, wavelength > 0 |
| `make_coherent_mode_set` | `CoherentModeSet` | Weights ≥ 0, matching shapes |
| `make_polychromatic_wavefront` | `PolychromaticWavefront` | Wavelengths > 0, weights sum |
| `make_lens_parameters` | `LensParameters` | Focal length, NA consistency |
| `make_ptycho_dataset` | `PtychoDataset` | Position count matches patterns |

## PyTree Registration

### The `@register_pytree_node_class` Decorator

Custom classes must be registered with JAX:

```python
from jax.tree_util import register_pytree_node_class
from typing import NamedTuple

@register_pytree_node_class
class MyDataType(NamedTuple):
    """Custom PyTree type."""

    field: Array
    parameter: float

    def tree_flatten(self):
        """Return (children, aux_data)."""
        return ((self.field, self.parameter), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from flattened form."""
        return cls(*children)
```

### NamedTuple Pattern

Janssen uses `NamedTuple` for PyTrees because:

1. Immutable (required for JAX)
2. Named fields (self-documenting)
3. Memory efficient (no `__dict__`)
4. Works with type checkers

## JAX Transformations

### JIT Compilation

PyTrees enable JIT compilation of entire workflows:

```python
from jax import jit

@jit
def propagate_and_measure(wavefront, distance):
    output = angular_spectrum(wavefront, distance)
    return output.intensity

# Compiled function works with PyTree input
intensity = propagate_and_measure(input_wavefront, 1e-3)
```

### Automatic Differentiation

Gradients flow through PyTree structures:

```python
from jax import grad

def loss_function(wavefront):
    output = propagate(wavefront)
    return jnp.sum((output.intensity - target)**2)

# Gradient with respect to wavefront field
grad_fn = grad(loss_function)
gradient = grad_fn(input_wavefront)
```

### Vectorization with vmap

Process batches automatically:

```python
from jax import vmap

# Vectorize over multiple wavefronts
batched_propagate = vmap(
    angular_spectrum,
    in_axes=(0, None),  # Batch over wavefronts, same distance
)

# Process batch at once
outputs = batched_propagate(wavefront_batch, distance)
```

## Best Practices

### Immutability

PyTree leaves should be immutable. Use functional updates:

```python
# Wrong: in-place modification
wavefront.field = new_field  # Error!

# Right: create new instance
new_wavefront = wavefront._replace(field=new_field)

# Or use factory
new_wavefront = make_optical_wavefront(
    field=new_field,
    wavelength=wavefront.wavelength,
    dx=wavefront.dx,
)
```

### Type Annotations

Use `jaxtyping` for array shapes:

```python
from jaxtyping import Array, Complex, Float

def my_function(
    field: Complex[Array, "H W"],
    wavelength: Float[Array, ""],
) -> Complex[Array, "H W"]:
    ...
```

### Device Placement

Arrays in PyTrees can be placed on specific devices:

```python
from jax import device_put

# Move entire PyTree to GPU
gpu_wavefront = device_put(wavefront, jax.devices("gpu")[0])
```

## Memory Considerations

### Efficient PyTree Operations

```python
# Prefer: single PyTree operation
output = propagate(wavefront)

# Avoid: extracting and re-creating
field = wavefront.field
wavelength = wavefront.wavelength
result = manual_propagate(field, wavelength)  # Loses structure
```

### Large Mode Sets

For many modes, consider chunked processing:

```python
from janssen.cohere import propagate_coherent_mode_set

# Process in chunks to manage memory
output = propagate_coherent_mode_set(
    mode_set,
    distance=1e-3,
    chunk_size=10,  # Process 10 modes at a time
)
```

## References

1. [JAX PyTrees Documentation](https://jax.readthedocs.io/en/latest/pytrees.html)
2. [JAX Sharp Bits: PyTrees](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pytrees)
3. Bradbury, J. et al. "JAX: composable transformations of Python+NumPy programs" (2018)
