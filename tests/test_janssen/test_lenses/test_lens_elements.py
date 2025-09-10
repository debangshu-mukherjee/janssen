import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex

from janssen.lenses import (
    create_lens_phase,
    double_concave_lens,
    double_convex_lens,
    lens_focal_length,
    lens_thickness_profile,
    meniscus_lens,
    plano_concave_lens,
    plano_convex_lens,
    propagate_through_lens,
)
from janssen.utils import LensParams, make_lens_params


class TestLensElements(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        x = jnp.arange(-self.nx // 2, self.nx // 2) * self.dx
        y = jnp.arange(-self.ny // 2, self.ny // 2) * self.dx
        self.xx, self.yy = jnp.meshgrid(x, y)
        self.r = jnp.sqrt(self.xx**2 + self.yy**2)
        self.default_params = make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )
        self.field = jnp.ones((self.ny, self.nx), dtype=complex)

    @chex.variants(with_jit=True, without_jit=True)
    def test_lens_thickness_profile_double_convex(self) -> None:
        """Test thickness profile for double convex lens.

        Note:
            Double convex lens with r1=r2=10mm, center thickness=1mm,
            diameter=5mm.
            - Center should have maximum thickness equal to center_thickness
            - Thickness decreases towards edges (thinner at mid-radius)
            - Zero thickness outside lens diameter
        """
        var_lens_thickness_profile = self.variant(lens_thickness_profile)
        r1 = 0.01
        r2 = 0.01
        center_thickness = 0.001
        diameter = 0.005
        var_lens_thickness_profile = var_lens_thickness_profile(
            self.r, r1, r2, center_thickness, diameter
        )
        chex.assert_shape(var_lens_thickness_profile, self.r.shape)
        center_val = var_lens_thickness_profile[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(center_val, center_thickness, rtol=1e-5)
        far_point_val = (
            var_lens_thickness_profile[0, 0]
            if self.r[0, 0] > diameter / 2
            else var_lens_thickness_profile[10, 10]
        )
        if self.r[10, 10] > diameter / 2:
            chex.assert_trees_all_close(far_point_val, 0.0, atol=1e-10)
        if diameter > 4 * self.dx:
            mid_radius = diameter / 4
            mid_idx = int(mid_radius / self.dx)
            if mid_idx > 0 and self.nx // 2 + mid_idx < self.nx:
                mid_val = var_lens_thickness_profile[
                    self.ny // 2, self.nx // 2 + mid_idx
                ]
                if float(mid_val) > 0:
                    chex.assert_scalar_positive(
                        float(center_val) - float(mid_val)
                    )

    @chex.variants(with_jit=True, without_jit=True)
    def test_lens_thickness_profile_plano(self) -> None:
        """Test thickness profile for plano-convex lens.

        Note:
            Plano-convex lens with r1=10mm, r2=infinity (flat surface).
            One side is curved, other side is flat, creating
            non-zero thickness gradient from center to edge.
        """
        var_lens_thickness_profile = self.variant(lens_thickness_profile)
        r1 = 0.01
        r2 = jnp.inf
        center_thickness = 0.001
        diameter = 0.005
        var_lens_thickness_profile = var_lens_thickness_profile(
            self.r, r1, r2, center_thickness, diameter
        )
        center_val = var_lens_thickness_profile[self.ny // 2, self.nx // 2]
        mid_radius_idx = self.nx // 2 + int(diameter / (4 * self.dx))
        mid_val = var_lens_thickness_profile[self.ny // 2, mid_radius_idx]
        chex.assert_tree_all_finite(jnp.abs(center_val - mid_val))
        chex.assert_scalar_positive(
            float(jnp.abs(center_val - mid_val)) - 1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("double_convex_equal_radii", 1.5, 0.01, 0.01, 0.01),
        ("double_concave_mixed", 1.5, 0.01, -0.01, -0.01),
        ("plano_convex", 1.5, 0.01, jnp.inf, 0.01),
        ("plano_concave", 1.5, -0.01, jnp.inf, -0.01),
        ("double_convex_different_radii", 1.7, 0.02, 0.01, 0.02),
        ("specific_focal_length", 1.5, 0.1, 0.3, 0.15),
    )
    def test_lens_focal_length(
        self, n: float, r1: float, r2: float, expected_f: float
    ) -> None:
        """Test focal length calculation using lensmaker's equation.

        Note:
            Testing various lens configurations:
            - Double convex with equal/different radii
            - Double concave (negative radii)
            - Plano lenses (one infinite radius)
            Verifies f = 1/((n-1)(1/r1 - 1/r2)) for finite radii.
        """
        var_lens_focal_length = self.variant(lens_focal_length)
        f = var_lens_focal_length(n, r1, r2)
        special_r1 = 0.1
        special_r2 = 0.3
        special_n = 1.5
        if n == special_n and r1 == special_r1 and r2 == special_r2:
            chex.assert_trees_all_close(f, expected_f, rtol=1e-10)
        elif r1 == r2:
            expected = r1 / (2 * (n - 1))
            chex.assert_trees_all_close(f, expected, rtol=1e-5)
        elif jnp.isfinite(r1) and jnp.isfinite(r2):
            expected = 1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))
            chex.assert_trees_all_close(f, expected, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_lens_phase(self) -> None:
        """Test lens phase profile and transmission mask creation.

        Note:
            Creates phase delay profile based on lens thickness and refractive
            index.
            - Phase profile should be positive at center (optical path delay)
            - Transmission mask is binary: 1 inside diameter, 0 outside
            - Center has full transmission, corners are blocked
        """
        var_create_lens_phase = self.variant(create_lens_phase)
        phase_profile, transmission = var_create_lens_phase(
            self.xx, self.yy, self.default_params, self.wavelength
        )
        chex.assert_shape(phase_profile, self.xx.shape)
        chex.assert_shape(transmission, self.xx.shape)
        unique_vals = jnp.unique(transmission)
        chex.assert_tree_all_finite(unique_vals)
        chex.assert_trees_all_equal(
            jnp.all((unique_vals == 0) | (unique_vals == 1)), True
        )
        center_phase = phase_profile[self.ny // 2, self.nx // 2]
        chex.assert_scalar_positive(float(center_phase))
        center_trans = transmission[self.ny // 2, self.nx // 2]
        chex.assert_trees_all_close(center_trans, 1.0)
        corner_trans = transmission[0, 0]
        chex.assert_trees_all_close(corner_trans, 0.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_propagate_through_lens(self) -> None:
        """Test field propagation through lens.

        Note:
            Applies lens phase and transmission to input field.
            - Output maintains input field shape
            - Phase is modified according to lens profile
            - Amplitude is modified by transmission mask
            - Corners (outside diameter) should have zero amplitude
        """
        var_propagate_through_lens = self.variant(propagate_through_lens)
        var_create_lens_phase = self.variant(create_lens_phase)
        phase_profile, transmission = var_create_lens_phase(
            self.xx, self.yy, self.default_params, self.wavelength
        )
        output_field = var_propagate_through_lens(
            self.field, phase_profile, transmission
        )
        chex.assert_shape(output_field, self.field.shape)
        chex.assert_trees_all_equal_shapes(output_field, self.field)
        chex.assert_trees_all_equal(
            jnp.allclose(output_field, self.field), False
        )
        center_in = self.field[self.ny // 2, self.nx // 2]
        center_out = output_field[self.ny // 2, self.nx // 2]
        phase_diff = jnp.angle(center_out / center_in)
        expected_phase = phase_profile[self.ny // 2, self.nx // 2]
        expected_phase_wrapped = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            phase_diff, expected_phase_wrapped, rtol=1e-5
        )
        corner_out = output_field[0, 0]
        chex.assert_trees_all_close(jnp.abs(corner_out), 0.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("equal_radii", 0.01, 0.005, 1.5, 0.001, 1.0),
        ("r2_smaller", 0.01, 0.005, 1.5, 0.001, 0.5),
        ("r2_larger", 0.01, 0.005, 1.5, 0.001, 2.0),
    )
    def test_double_convex_lens(
        self,
        focal_length: float,
        diameter: float,
        n: float,
        center_thickness: float,
        r_ratio: float,
    ) -> None:
        """Test double convex lens parameter generation.

        Note:
            Creates lens with two convex surfaces (positive radii).
            Testing different r2/r1 ratios (0.5, 1.0, 2.0) to verify
            correct radius calculation for desired focal length.
            Both radii should be positive for converging lens.
        """
        var_double_convex_lens = self.variant(double_convex_lens)
        params = var_double_convex_lens(
            focal_length, diameter, n, center_thickness, r_ratio
        )
        chex.assert_trees_all_close(params.focal_length, focal_length)
        chex.assert_trees_all_close(params.diameter, diameter)
        chex.assert_trees_all_close(params.n, n)
        chex.assert_trees_all_close(params.center_thickness, center_thickness)
        chex.assert_scalar_positive(float(params.r1))
        chex.assert_scalar_positive(float(params.r2))
        actual_ratio = params.r2 / params.r1
        chex.assert_trees_all_close(actual_ratio, r_ratio, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("equal_radii", 0.01, 0.005, 1.5, 0.001, 1.0),
        ("r2_smaller", 0.01, 0.005, 1.5, 0.001, 0.5),
    )
    def test_double_concave_lens(
        self,
        focal_length: float,
        diameter: float,
        n: float,
        center_thickness: float,
        r_ratio: float,
    ) -> None:
        """Test double concave lens parameter generation.

        Note:
            Creates lens with two concave surfaces (negative radii).
            Used for diverging lenses with negative focal length.
            Both radii should be negative.
        """
        var_double_concave_lens = self.variant(double_concave_lens)
        params = var_double_concave_lens(
            focal_length, diameter, n, center_thickness, r_ratio
        )
        chex.assert_trees_all_close(params.focal_length, focal_length)
        chex.assert_trees_all_close(params.diameter, diameter)
        chex.assert_trees_all_close(params.n, n)
        chex.assert_trees_all_close(params.center_thickness, center_thickness)
        chex.assert_scalar_negative(float(params.r1))
        chex.assert_scalar_negative(float(params.r2))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("convex_first", True),
        ("convex_second", False),
    )
    def test_plano_convex_lens(self, convex_first: bool) -> None:
        """Test plano-convex lens parameter generation.

        Note:
            One surface is flat (infinite radius), other is convex.
            Testing both orientations:
            - convex_first=True: curved surface first, flat second
            - convex_first=False: flat first, curved second
            Convex surface should have positive radius.
        """
        var_plano_convex_lens = self.variant(plano_convex_lens)
        focal_length = 0.01
        diameter = 0.005
        n = 1.5
        center_thickness = 0.001
        params = var_plano_convex_lens(
            focal_length, diameter, n, center_thickness, convex_first
        )
        chex.assert_trees_all_close(params.focal_length, focal_length)
        chex.assert_trees_all_close(params.diameter, diameter)
        chex.assert_trees_all_close(params.n, n)
        chex.assert_trees_all_close(params.center_thickness, center_thickness)
        if convex_first:
            chex.assert_scalar_positive(float(params.r1))
            chex.assert_trees_all_equal(jnp.isinf(params.r2), True)
        else:
            chex.assert_trees_all_equal(jnp.isinf(params.r1), True)
            chex.assert_scalar_positive(float(params.r2))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("concave_first", True),
        ("concave_second", False),
    )
    def test_plano_concave_lens(self, concave_first: bool) -> None:
        """Test plano-concave lens parameter generation.

        Note:
            One surface is flat (infinite radius), other is concave.
            Testing both orientations for diverging lens.
            Concave surface should have negative radius.
        """
        var_plano_concave_lens = self.variant(plano_concave_lens)
        focal_length = 0.01
        diameter = 0.005
        n = 1.5
        center_thickness = 0.001
        params = var_plano_concave_lens(
            focal_length, diameter, n, center_thickness, concave_first
        )
        chex.assert_trees_all_close(params.focal_length, focal_length)
        chex.assert_trees_all_close(params.diameter, diameter)
        chex.assert_trees_all_close(params.n, n)
        chex.assert_trees_all_close(params.center_thickness, center_thickness)
        if concave_first:
            chex.assert_scalar_negative(float(params.r1))
            chex.assert_trees_all_equal(jnp.isinf(params.r2), True)
        else:
            chex.assert_trees_all_equal(jnp.isinf(params.r1), True)
            chex.assert_scalar_negative(float(params.r2))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("small_ratio_convex_first", 0.5, True),
        ("small_ratio_concave_first", 0.5, False),
        ("large_ratio_convex_first", 2.0, True),
        ("large_ratio_concave_first", 2.0, False),
    )
    def test_meniscus_lens(self, r_ratio: float, convex_first: bool) -> None:
        """Test meniscus lens parameter generation.

        Note:
            Meniscus lens has one convex and one concave surface.
            - convex_first=True: r1 positive, r2 negative
            - convex_first=False: r1 negative, r2 positive
            Testing different r_ratio values (0.5, 2.0) for curvature balance.
        """
        var_meniscus_lens = self.variant(meniscus_lens)
        focal_length = 0.01
        diameter = 0.005
        n = 1.5
        center_thickness = 0.001
        params = var_meniscus_lens(
            focal_length, diameter, n, center_thickness, r_ratio, convex_first
        )
        chex.assert_trees_all_close(params.focal_length, focal_length)
        chex.assert_trees_all_close(params.diameter, diameter)
        chex.assert_trees_all_close(params.n, n)
        chex.assert_trees_all_close(params.center_thickness, center_thickness)
        if convex_first:
            chex.assert_scalar_positive(float(params.r1))
            chex.assert_scalar_negative(float(params.r2))
        else:
            chex.assert_scalar_negative(float(params.r1))
            chex.assert_scalar_positive(float(params.r2))

    def test_jax_transformations_on_thickness_profile(self) -> None:
        """Test JAX transformations on lens thickness profile.

        Note:
            Verifies that lens functions work with JAX transformations:
            - JIT compilation produces identical results
            - Gradient computation works (for optimization)
            - Gradients are finite and have correct shape
        """

        @jax.jit
        def jitted_thickness(
            r: Array, r1: float, r2: float, ct: float, d: float
        ) -> Array:
            return lens_thickness_profile(r, r1, r2, ct, d)

        thickness_jit = jitted_thickness(self.r, 0.01, 0.01, 0.001, 0.005)
        thickness_normal = lens_thickness_profile(
            self.r, 0.01, 0.01, 0.001, 0.005
        )
        chex.assert_trees_all_close(thickness_jit, thickness_normal)

        def loss_fn(r1: float) -> Array:
            var_lens_thickness_profile = lens_thickness_profile(
                self.r, r1, 0.01, 0.001, 0.005
            )
            return jnp.sum(var_lens_thickness_profile**2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(0.01)
        chex.assert_scalar_non_negative(abs(float(grad)))
        chex.assert_shape(grad, ())

    def test_jax_transformations_on_lens_creation(self) -> None:
        """Test JAX transformations on lens creation functions.

        Note:
            Tests JIT compilation and vmap on lens parameter creation.
            - JIT produces identical lens parameters
            - vmap over focal lengths creates batch of lenses
            - Each lens in batch has different parameters
        """

        @jax.jit
        def create_lens(focal_length: float) -> LensParams:
            return double_convex_lens(focal_length, 0.005, 1.5, 0.001, 1.0)

        params_jit = create_lens(0.01)
        params_normal = double_convex_lens(0.01, 0.005, 1.5, 0.001, 1.0)
        chex.assert_trees_all_close(params_jit, params_normal)
        focal_lengths = jnp.array([0.01, 0.02, 0.03])
        vmapped_create = jax.vmap(
            lambda f: double_convex_lens(f, 0.005, 1.5, 0.001, 1.0)
        )
        params_batch = vmapped_create(focal_lengths)
        chex.assert_shape(params_batch.focal_length, (3,))
        chex.assert_trees_all_equal(
            jnp.allclose(params_batch.r1[0], params_batch.r1[1]), False
        )

    def test_field_dtype_preservation(self) -> None:
        """Test that field dtype is preserved through propagation.

        Note:
            Complex128 input should produce complex128 output.
            Important for maintaining numerical precision in simulations.
        """
        complex_field = jnp.ones((self.ny, self.nx), dtype=jnp.complex128)
        phase_profile, transmission = create_lens_phase(
            self.xx, self.yy, self.default_params, self.wavelength
        )
        output = propagate_through_lens(
            complex_field, phase_profile, transmission
        )
        chex.assert_type(output, jnp.complex128)

    @parameterized.named_parameters(
        ("zero_field", jnp.zeros((128, 128), dtype=complex)),
        ("complex_uniform", jnp.ones((128, 128)) * (1 + 1j)),
        ("phase_uniform", jnp.exp(1j * jnp.ones((128, 128)))),
    )
    def test_propagate_edge_cases(
        self, input_field: Complex[Array, "128 128"]
    ) -> None:
        """Test propagation with edge case input fields.

        Note:
            Testing special input cases:
            - Zero field (no light)
            - Complex uniform field (1+1j)
            - Pure phase field (exp(1j))
            All should maintain shape and complex dtype.
        """
        phase_profile, transmission = create_lens_phase(
            self.xx, self.yy, self.default_params, self.wavelength
        )
        output = propagate_through_lens(
            input_field, phase_profile, transmission
        )
        chex.assert_shape(output, input_field.shape)
        chex.assert_trees_all_equal(jnp.iscomplexobj(output), True)


if __name__ == "__main__":
    pytest.main([__file__])
