# JAX Sharp Edges

This guide documents common pitfalls when working with JAX in janssen.
Understanding these "sharp edges" helps avoid subtle bugs and performance
issues.

## Arrays First, Always

**The most important rule: prefer JAX arrays over Python lists/tuples.**

JAX traces through your code to compile it. Python lists and tuples are
opaque to the tracer, leading to:

- Silent performance degradation (loops don't vectorize)
- `ConcretizationTypeError` when JIT encounters traced values in Python
  control flow

### Bad: Python List with Loop

```python
def generate_modes_bad(max_order: int) -> list:
    """Returns a Python list - can't be vmapped."""
    mode_indices = []
    for n in range(max_order):
        for m in range(max_order):
            mode_indices.append((n, m))
    return mode_indices

# This loop runs in Python, not on GPU
for n, m in generate_modes_bad(10):
    process_mode(n, m)
```

### Good: JAX Array with vmap

```python
def generate_modes_good(max_order: int) -> Float[Array, " num_modes 2"]:
    """Returns a JAX array - ready for vmap."""
    indices = []
    for n in range(max_order):
        for m in range(max_order):
            indices.append((n, m))
    return jnp.array(indices, dtype=jnp.float64)

# This runs vectorized on GPU
mode_indices = generate_modes_good(10)
jax.vmap(lambda idx: process_mode(idx[0], idx[1]))(mode_indices)
```

Note: The Python loop in `generate_modes_good` is fine because it runs
once at trace time to build the array. The key is that the *output* is
a JAX array that can be vmapped over.

## Control Flow with Traced Values

Python control flow (`if`, `for`, `while`) doesn't work with traced values.

### Bad: Python if with Traced Value

```python
@jax.jit
def bad_conditional(x):
    if x > 0:  # ConcretizationTypeError!
        return x
    else:
        return -x
```

### Good: jax.lax.cond

```python
@jax.jit
def good_conditional(x):
    return jax.lax.cond(
        x > 0,
        lambda: x,
        lambda: -x
    )
```

### Bad: Python for with Traced Bound

```python
@jax.jit
def bad_loop(n, x):
    result = 0.0
    for i in range(n):  # n is traced - fails!
        result += x[i]
    return result
```

### Good: jax.lax.fori_loop

```python
@jax.jit
def good_loop(n, x):
    def body(i, acc):
        return acc + x[i]
    return jax.lax.fori_loop(0, n, body, 0.0)
```

## Static vs Dynamic Arguments

Some arguments determine output *shape* and must be static (known at
compile time).

### The `_impl` Pattern

```python
from functools import partial

@partial(jax.jit, static_argnums=(0, 1))
def _compute_impl(height: int, width: int, data: Array) -> Array:
    """Height and width are static - they determine output shape."""
    result = jnp.zeros((height, width))
    # ... computation ...
    return result

@jaxtyped(typechecker=beartype)
def compute(grid_size: Tuple[int, int], data: Array) -> Array:
    """Public API extracts static args."""
    return _compute_impl(grid_size[0], grid_size[1], data)
```

## Common Patterns

### Hermite Polynomial Recurrence

The Hermite polynomial recurrence $H_n(x) = 2xH_{n-1}(x) - 2(n-1)H_{n-2}(x)$
requires iteration. Use `jax.lax.fori_loop`:

```python
def hermite_polynomial(order: Array, x: Array) -> Array:
    def body_fn(k, carry):
        h_prev2, h_prev1 = carry
        h_curr = 2.0 * x * h_prev1 - 2.0 * (k - 1) * h_prev2
        return h_prev1, h_curr

    h0 = jnp.ones_like(x)
    h1 = 2.0 * x

    _, h_n = jax.lax.fori_loop(2, order + 1, body_fn, (h0, h1))

    return jnp.where(order == 0, h0, jnp.where(order == 1, h1, h_n))
```

### Vectorizing Over Indices

Instead of looping over mode indices, use vmap:

```python
# Bad
modes = []
for n, m in mode_indices:
    modes.append(compute_mode(n, m))
modes = jnp.stack(modes)

# Good
def compute_single_mode(indices):
    n, m = indices[0], indices[1]
    return compute_mode(n, m)

modes = jax.vmap(compute_single_mode)(mode_indices_array)
```

## Why jaxtyping + beartype Helps

With runtime type checking, incorrect array types fail fast:

```python
@jaxtyped(typechecker=beartype)
def process(data: Float[Array, " N 2"]) -> Float[Array, " N"]:
    return jnp.sum(data, axis=1)

# This fails immediately with a clear error:
process([(1, 2), (3, 4)])  # TypeError: expected Array, got list

# This works:
process(jnp.array([[1, 2], [3, 4]]))
```

The type annotations serve as documentation *and* runtime validation,
catching list/tuple misuse at function boundaries rather than deep in
traced code.

## Summary

| Pattern | Bad | Good |
| ------- | --- | ---- |
| Data structure | Python list/tuple | `jnp.array` |
| Iteration | `for i in range(n)` | `jax.vmap` or `jax.lax.fori_loop` |
| Conditional | `if x > 0` | `jax.lax.cond` |
| Shape params | Traced values | `static_argnums` |
| Type safety | Hope for the best | `jaxtyping` + `beartype` |
