# Contributing to Janssen

Thank you for your interest in contributing to Janssen! This document provides guidelines and best practices for contributing to the project.

## Table of Contents
- [Development Setup](#development-setup)
- [Code Style and Conventions](#code-style-and-conventions)
- [Type Hinting Rules](#type-hinting-rules)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Development Setup

### Prerequisites
- Python 3.12 or higher
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/janssen.git
cd janssen

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Running Quality Checks
```bash
# Run tests
pytest

# Run type checking
mypy src/

# Run linting
ruff check .

# Format code
ruff format .
```

## Code Style and Conventions

### General Principles
1. **Follow PEP 8** with a line length limit of 100 characters
2. **Use meaningful variable names** - prefer descriptive names over short abbreviations
3. **Keep functions focused** - each function should do one thing well
4. **Avoid magic numbers** - use named constants for repeated values
5. **No commented-out code** - remove dead code rather than commenting it out

### Import Organization
Organize imports in the following order:
1. Standard library imports
2. Type annotation imports from `typing`
3. Third-party imports (numpy, jax, etc.)
4. Local application imports

Example:
```python
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex

from janssen.lenses import create_lens_phase, lens_focal_length
from janssen.utils import LensParams, make_lens_params
```

### JAX-Specific Conventions
1. **Enable 64-bit precision** when needed:
   ```python
   jax.config.update("jax_enable_x64", True)
   ```

2. **Use PyTree structures** for complex data types:
   ```python
   @register_pytree_node_class
   class LensParams(NamedTuple):
       focal_length: Float[Array, " "]
       diameter: Float[Array, " "]
   ```

3. **Prefer functional programming** - avoid mutable state where possible

## Type Hinting Rules

### Mandatory Type Hints
All functions must include type hints for:
- **All parameters** (including `self` in methods is implicit)
- **Return types** (use `-> None` for functions that don't return a value)

### Type Annotation Guidelines

1. **Import necessary types**:
   ```python
   from typing import Any, Optional, Tuple, Union
   from jaxtyping import Array, Bool, Complex, Float, Int, Num
   ```

2. **Use specific JAX array types** when possible:
   ```python
   def process_field(field: Complex[Array, "height width"]) -> Float[Array, "height width"]:
       return jnp.abs(field)
   ```

3. **Use type aliases** for complex types:
   ```python
   from typing import TypeAlias

   ScalarFloat: TypeAlias = Union[float, Float[Array, " "]]
   ScalarComplex: TypeAlias = Union[complex, Complex[Array, " "]]
   ```

4. **Annotate test methods**:
   ```python
   def test_lens_thickness_profile(self) -> None:
       """Test lens thickness profile calculation."""
       ...
   ```

5. **Handle unused parameters** in parameterized tests:
   ```python
   # Prefix with underscore if parameter is needed but unused
   def test_something(self, used_param: float, _unused_param: float) -> None:
       ...
   ```

### Factory Functions
Always use factory functions with runtime type checking:
```python
# Good - use factory function
params = make_lens_params(
    focal_length=0.01,
    diameter=0.005,
    n=1.5,
    center_thickness=0.001,
    r1=0.01,
    r2=0.01,
)

# Bad - direct instantiation
params = LensParams(0.01, 0.005, 1.5, 0.001, 0.01, 0.01)
```

## Testing Guidelines

### Test Structure
Tests should be organized in a parallel structure to the source code:
```
src/janssen/lenses/lens_elements.py
tests/test_janssen/test_lenses/test_lens_elements.py
```

### Writing Tests

#### 1. Use Descriptive Test Names
```python
class TestLensElements(chex.TestCase, parameterized.TestCase):
    def test_lens_thickness_profile_double_convex(self) -> None:
        """Test thickness profile for double convex lens."""
        ...
    
    def test_lens_focal_length_plano_concave(self) -> None:
        """Test focal length calculation for plano-concave lens."""
        ...
```

#### 2. Use Named Parameters for Clarity
```python
@parameterized.named_parameters(
    ("double_convex_equal_radii", 1.5, 0.01, 0.01, 0.01),
    ("plano_convex", 1.5, 0.01, jnp.inf, 0.01),
    ("specific_focal_length", 1.5, 0.1, 0.3, 0.15),
)
def test_lens_focal_length(
    self, n: float, r1: float, r2: float, expected_f: float
) -> None:
    ...
```

#### 3. Test with JAX Transformations
Use `chex.variants` to test both JIT-compiled and non-JIT versions:
```python
@chex.variants(with_jit=True, without_jit=True)
def test_lens_thickness_profile(self) -> None:
    var_lens_thickness = self.variant(lens_thickness_profile)
    thickness = var_lens_thickness(self.r, r1, r2, ct, d)
    ...
```

#### 4. Common Test Setup
Use `setUp` method for common initialization:
```python
def setUp(self) -> None:
    super().setUp()
    self.nx = 128
    self.ny = 128
    self.dx = 1e-6
    self.wavelength = 500e-9
    x = jnp.arange(-self.nx // 2, self.nx // 2) * self.dx
    y = jnp.arange(-self.ny // 2, self.ny // 2) * self.dx
    self.xx, self.yy = jnp.meshgrid(x, y)
```

#### 5. Use Chex Assertions
Prefer chex assertions for numerical testing:
```python
# Shape assertions
chex.assert_shape(output, (128, 128))
chex.assert_trees_all_equal_shapes(output, input_field)

# Value assertions
chex.assert_trees_all_close(f, expected, rtol=1e-5)
chex.assert_scalar_positive(float(params.r1))
chex.assert_scalar_negative(float(params.r2))
chex.assert_tree_all_finite(values)

# Type assertions
chex.assert_type(output, jnp.complex128)
```

#### 6. Test Edge Cases
Always include tests for:
- Boundary conditions (zero, infinity, negative values)
- Special cases (plano lenses, meniscus configurations)
- Different data types and precisions
- JAX transformations (jit, vmap, grad)

Example edge case testing:
```python
@parameterized.named_parameters(
    ("zero_field", jnp.zeros((128, 128), dtype=complex)),
    ("complex_uniform", jnp.ones((128, 128)) * (1 + 1j)),
    ("phase_uniform", jnp.exp(1j * jnp.ones((128, 128)))),
)
def test_propagate_edge_cases(
    self, input_field: Complex[Array, "128 128"]
) -> None:
    ...
```

#### 7. Test JAX Transformations
Ensure your code works with JAX transformations:
```python
def test_jax_transformations_on_thickness_profile(self) -> None:
    # Test JIT compilation
    @jax.jit
    def jitted_thickness(
        r: Array, r1: float, r2: float, ct: float, d: float
    ) -> Array:
        return lens_thickness_profile(r, r1, r2, ct, d)
    
    # Test gradient computation
    def loss_fn(r1: float) -> Array:
        thickness = lens_thickness_profile(self.r, r1, 0.01, 0.001, 0.005)
        return jnp.sum(thickness**2)
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(0.01)
    chex.assert_scalar_non_negative(abs(float(grad)))
    
    # Test vmap
    vmapped_create = jax.vmap(
        lambda f: double_convex_lens(f, 0.005, 1.5, 0.001, 1.0)
    )
    focal_lengths = jnp.array([0.01, 0.02, 0.03])
    params_batch = vmapped_create(focal_lengths)
    chex.assert_shape(params_batch.focal_length, (3,))
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_janssen/test_lenses/test_lens_elements.py

# Run with coverage
pytest --cov=janssen --cov-report=html

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf
```

## Documentation Standards

### Docstring Format
Use NumPy-style docstrings:
```python
def lens_focal_length(n: float, r1: float, r2: float) -> float:
    """Calculate the focal length of a lens using the lensmaker's equation.
    
    Parameters
    ----------
    n : float
        Refractive index of the lens material
    r1 : float
        Radius of curvature of the first surface (positive for convex)
    r2 : float
        Radius of curvature of the second surface (positive for convex)
    
    Returns
    -------
    float
        Focal length of the lens in meters
    
    Notes
    -----
    Uses the thin lens approximation. For thick lenses, corrections
    may be necessary.
    
    Examples
    --------
    >>> focal_length = lens_focal_length(1.5, 0.01, 0.01)
    >>> print(f"Focal length: {focal_length:.3f} m")
    Focal length: 0.010 m
    """
```

### Module Documentation
Each module should have a docstring explaining its purpose:
```python
"""
Module: janssen.lenses.lens_elements
------------------------------------
Differentiable optical lens elements and propagation functions.

This module provides functions for creating and simulating various
types of optical lenses including convex, concave, plano, and meniscus
configurations.
"""
```

## Commit Messages

Follow conventional commit format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Example:
```
feat(lenses): add meniscus lens configuration support

- Implemented meniscus_lens factory function
- Added comprehensive tests for convex/concave first configurations
- Updated documentation with usage examples
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following all guidelines above

3. **Run all quality checks**:
   ```bash
   # Format code
   ruff format .
   
   # Check linting
   ruff check .
   
   # Run tests
   pytest
   
   # Check types (if configured)
   mypy src/
   ```

4. **Ensure all tests pass** including any new tests you've added

5. **Update documentation** if you've added new features or changed APIs

6. **Create a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

7. **Address review feedback** promptly

### PR Checklist
Before submitting a PR, ensure:
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Type hints are complete and correct
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No debugging code or print statements remain
- [ ] New features have corresponding tests

## Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Open a new issue for clarification
3. Reach out to maintainers

Thank you for contributing to Janssen!