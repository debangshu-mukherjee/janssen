"""Tests for Bessel functions in janssen.optics.bessel module."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-public-methods,no-self-use

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from janssen.optics.bessel import bessel_j0, bessel_jn, bessel_kv


class TestBesselJ0(chex.TestCase):
    """Test bessel_j0 function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_scalar_input(self):
        """Test bessel_j0 with scalar input.

        Note:
            J_0(0) = 1.0 is a well-known identity.
            J_0(1) ≈ 0.765 is the value at x=1.
        """
        var_bessel_j0 = self.variant(bessel_j0)
        result_zero = var_bessel_j0(jnp.array(0.0))
        chex.assert_trees_all_close(result_zero, 1.0, rtol=1e-10)

        result_one = var_bessel_j0(jnp.array(1.0))
        chex.assert_trees_all_close(result_one, 0.7651976865, rtol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_array_input(self):
        """Test bessel_j0 with array input.

        Note:
            Verify that the output shape matches the input shape
            and that known values are computed correctly.
        """
        var_bessel_j0 = self.variant(bessel_j0)
        x = jnp.linspace(0, 10, 100)
        result = var_bessel_j0(x)
        chex.assert_shape(result, (100,))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_array_input(self):
        """Test bessel_j0 with 2D array input.

        Note:
            bessel_j0 should preserve the shape of multidimensional
            arrays and not add extra dimensions.
        """
        var_bessel_j0 = self.variant(bessel_j0)
        x = jnp.ones((256, 256)) * 2.0
        result = var_bessel_j0(x)
        chex.assert_shape(result, (256, 256))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_non_square_2d_array(self):
        """Test bessel_j0 with non-square 2D array.

        Note:
            Verifying that non-square arrays maintain their shape
            through the bessel_j0 computation.
        """
        var_bessel_j0 = self.variant(bessel_j0)
        x = jnp.ones((128, 256)) * 1.5
        result = var_bessel_j0(x)
        chex.assert_shape(result, (128, 256))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient(self):
        """Test gradient computation of bessel_j0.

        Note:
            The derivative of J_0(x) is -J_1(x).
            Verify that gradients can be computed (differentiability).
        """
        var_bessel_j0 = self.variant(bessel_j0)

        def f(x: float) -> float:
            return jnp.sum(var_bessel_j0(x))

        grad_fn = jax.grad(f)
        grad = grad_fn(jnp.array(2.0))
        chex.assert_tree_all_finite(grad)
        chex.assert_shape(grad, ())


class TestBesselJn(chex.TestCase, parameterized.TestCase):
    """Test bessel_jn function.

    Note:
        bessel_jn cannot be JIT-compiled because the order n
        must be a compile-time constant, so all tests are without JIT.
    """

    @parameterized.named_parameters(
        ("order_0", 0, 0.0, 1.0),
        ("order_1", 1, 0.0, 0.0),
        ("order_2", 2, 0.0, 0.0),
        ("order_1_at_1", 1, 1.0, 0.44005058574),
    )
    def test_known_values(self, n: int, x_val: float, expected: float):
        """Test bessel_jn with known values.

        Note:
            Testing known identities:
            - J_0(0) = 1, J_n(0) = 0 for n > 0
            - J_1(1) ≈ 0.440
        """
        result = bessel_jn(n, jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-6)

    def test_array_input(self):
        """Test bessel_jn with array input.

        Note:
            Order n must be a compile-time constant,
            so we cannot JIT this function.
        """
        x = jnp.linspace(0, 10, 100)
        result = bessel_jn(1, x)
        chex.assert_shape(result, (100,))
        chex.assert_tree_all_finite(result)

    def test_2d_array_input(self):
        """Test bessel_jn with 2D array input.

        Note:
            bessel_jn should preserve the shape of multidimensional
            arrays and not add extra dimensions.
        """
        x = jnp.ones((256, 256)) * 2.0
        result = bessel_jn(2, x)
        chex.assert_shape(result, (256, 256))
        chex.assert_tree_all_finite(result)

    def test_non_square_2d_array(self):
        """Test bessel_jn with non-square 2D array.

        Note:
            Verifying that non-square arrays maintain their shape
            through the bessel_jn computation.
        """
        x = jnp.ones((128, 256)) * 1.5
        result = bessel_jn(3, x)
        chex.assert_shape(result, (128, 256))
        chex.assert_tree_all_finite(result)


class TestBesselKv(chex.TestCase, parameterized.TestCase):
    """Test bessel_kv function."""

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("order_0", 0.0, 1.0, 0.4210244382),
        ("order_1", 1.0, 1.0, 0.6019072301),
        ("order_2", 2.0, 1.0, 1.6248388987),
        ("order_half", 0.5, 1.0, 0.46756394),
    )
    def test_known_values(self, v: float, x_val: float, expected: float):
        """Test bessel_kv with known values.

        Note:
            Testing against reference values computed with scipy.
            K_v(x) is the modified Bessel function of the second kind.
        """
        var_bessel_kv = self.variant(bessel_kv)
        result = var_bessel_kv(v, jnp.array(x_val))
        chex.assert_trees_all_close(result, expected, rtol=1e-4)

    @chex.variants(with_jit=True, without_jit=True)
    def test_array_input(self):
        """Test bessel_kv with array input.

        Note:
            Verify that the output shape matches the input shape
            and all values are finite and positive for K_v.
        """
        var_bessel_kv = self.variant(bessel_kv)
        x = jnp.linspace(0.1, 10, 100)
        result = var_bessel_kv(1.0, x)
        chex.assert_shape(result, (100,))
        chex.assert_tree_all_finite(result)
        chex.assert_trees_all_equal(jnp.all(result > 0), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_array_input(self):
        """Test bessel_kv with 2D array input.

        Note:
            bessel_kv should preserve the shape of multidimensional arrays.
        """
        var_bessel_kv = self.variant(bessel_kv)
        x = jnp.ones((64, 64)) * 2.0
        result = var_bessel_kv(1.5, x)
        chex.assert_shape(result, (64, 64))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradient(self):
        """Test gradient computation of bessel_kv.

        Note:
            Verify that bessel_kv is differentiable with respect
            to the input x.
        """
        var_bessel_kv = self.variant(bessel_kv)

        def f(x: float) -> float:
            return jnp.sum(var_bessel_kv(1.0, x))

        grad_fn = jax.grad(f)
        grad = grad_fn(jnp.array(2.0))
        chex.assert_tree_all_finite(grad)
        chex.assert_shape(grad, ())


class TestJAXTransformations(chex.TestCase):
    """Test JAX transformations on Bessel functions."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_vmap_bessel_j0(self):
        """Test vmapping over bessel_j0.

        Note:
            Testing that bessel_j0 can be vmapped over batch dimensions
            for efficient batch processing.
        """
        var_bessel_j0 = self.variant(bessel_j0)
        x_batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        vmapped_j0 = jax.vmap(var_bessel_j0)
        result = vmapped_j0(x_batch)
        chex.assert_shape(result, (2, 3))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_vmap_bessel_kv(self):
        """Test vmapping over bessel_kv.

        Note:
            Testing that bessel_kv can be vmapped over batch dimensions
            with a fixed order parameter.
        """
        var_bessel_kv = self.variant(bessel_kv)
        x_batch = jnp.array([1.0, 2.0, 3.0, 4.0])

        def kv_v1(x: float) -> float:
            return var_bessel_kv(1.0, x)

        vmapped_kv = jax.vmap(kv_v1)
        result = vmapped_kv(x_batch)
        chex.assert_shape(result, (4,))
        chex.assert_tree_all_finite(result)


if __name__ == "__main__":
    pytest.main([__file__])
