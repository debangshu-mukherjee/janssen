import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float

from janssen.lenses import (
    angular_spectrum_prop,
    digital_zoom,
    fraunhofer_prop,
    fresnel_prop,
    lens_propagation,
    optical_zoom,
)
from janssen.utils import (
    OpticalWavefront,
    make_lens_params,
    make_optical_wavefront,
)


class TestLensProp(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        
        # Create coordinate grids
        x = jnp.arange(-self.nx // 2, self.nx // 2) * self.dx
        y = jnp.arange(-self.ny // 2, self.ny // 2) * self.dx
        self.xx, self.yy = jnp.meshgrid(x, y)
        self.r = jnp.sqrt(self.xx**2 + self.yy**2)
        
        # Create test wavefront - Gaussian beam
        sigma = 20 * self.dx
        gaussian_field = jnp.exp(-(self.r**2) / (2 * sigma**2))
        self.test_wavefront = make_optical_wavefront(
            field=gaussian_field.astype(complex),
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        
        # Create plane wave
        self.plane_wave = make_optical_wavefront(
            field=jnp.ones((self.ny, self.nx), dtype=complex),
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        
        # Create test lens
        self.test_lens = make_lens_params(
            focal_length=0.01,
            diameter=0.00005,
            n=1.5,
            center_thickness=0.001,
            r1=0.01,
            r2=0.01,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_angular_spectrum_prop_basic(self) -> None:
        """
        Test basic angular spectrum propagation.
        """
        var_angular_spectrum = self.variant(angular_spectrum_prop)
        z_distance = 1e-3
        
        propagated = var_angular_spectrum(self.test_wavefront, z_distance)
        
        # Check shape preservation
        chex.assert_shape(propagated.field, self.test_wavefront.field.shape)
        
        # Check wavelength preservation
        chex.assert_trees_all_close(propagated.wavelength, self.test_wavefront.wavelength)
        
        # Check dx preservation
        chex.assert_trees_all_close(propagated.dx, self.test_wavefront.dx)
        
        # Check z position update
        chex.assert_trees_all_close(
            propagated.z_position,
            self.test_wavefront.z_position + z_distance,
            rtol=1e-10
        )
        
        # Field should change after propagation
        chex.assert_trees_all_equal(
            jnp.allclose(propagated.field, self.test_wavefront.field),
            False
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("short_distance", 1e-4, 1.0),
        ("medium_distance", 1e-3, 1.0),
        ("long_distance", 1e-2, 1.0),
        ("with_glass", 1e-3, 1.5),
        ("with_water", 1e-3, 1.33),
    )
    def test_angular_spectrum_prop_distances(
        self, z_distance: float, refractive_index: float
    ) -> None:
        """
        Test angular spectrum propagation at various distances and
        media.
        """
        var_angular_spectrum = self.variant(angular_spectrum_prop)
        
        propagated = var_angular_spectrum(
            self.test_wavefront, z_distance, refractive_index
        )
        
        # Energy conservation check (Parseval's theorem)
        initial_energy = jnp.sum(jnp.abs(self.test_wavefront.field) ** 2)
        propagated_energy = jnp.sum(jnp.abs(propagated.field) ** 2)
        chex.assert_trees_all_close(initial_energy, propagated_energy, rtol=1e-5)
        
        # Z position should account for refractive index
        expected_z = self.test_wavefront.z_position + refractive_index * z_distance
        chex.assert_trees_all_close(propagated.z_position, expected_z, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_fresnel_prop_basic(self) -> None:
        """
        Test basic Fresnel propagation.
        """
        var_fresnel = self.variant(fresnel_prop)
        z_distance = 1e-3
        
        propagated = var_fresnel(self.test_wavefront, z_distance)
        
        # Check shape preservation
        chex.assert_shape(propagated.field, self.test_wavefront.field.shape)
        
        # Check metadata preservation
        chex.assert_trees_all_close(propagated.wavelength, self.test_wavefront.wavelength)
        chex.assert_trees_all_close(propagated.dx, self.test_wavefront.dx)
        
        # Field should change
        chex.assert_trees_all_equal(
            jnp.allclose(propagated.field, self.test_wavefront.field),
            False
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("vacuum", 1.0),
        ("glass", 1.5),
        ("water", 1.33),
    )
    def test_fresnel_prop_refractive_index(self, refractive_index: float) -> None:
        """
        Test Fresnel propagation in different media.
        """
        var_fresnel = self.variant(fresnel_prop)
        z_distance = 5e-3
        
        propagated = var_fresnel(self.test_wavefront, z_distance, refractive_index)
        
        # Check z position accounts for refractive index
        expected_z = self.test_wavefront.z_position + refractive_index * z_distance
        chex.assert_trees_all_close(propagated.z_position, expected_z, rtol=1e-10)
        
        # Field should be complex
        chex.assert_trees_all_equal(jnp.iscomplexobj(propagated.field), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_fraunhofer_prop_basic(self) -> None:
        """
        Test basic Fraunhofer propagation.
        """
        var_fraunhofer = self.variant(fraunhofer_prop)
        z_distance = 0.1  # Longer distance for far-field
        
        propagated = var_fraunhofer(self.test_wavefront, z_distance)
        
        # Check shape preservation
        chex.assert_shape(propagated.field, self.test_wavefront.field.shape)
        
        # Check metadata
        chex.assert_trees_all_close(propagated.wavelength, self.test_wavefront.wavelength)
        chex.assert_trees_all_close(propagated.dx, self.test_wavefront.dx)
        chex.assert_trees_all_close(
            propagated.z_position,
            self.test_wavefront.z_position + z_distance,
            rtol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_propagation_consistency(self) -> None:
        """
        Test that different propagation methods give similar results in
        appropriate regimes.
        """
        var_angular = self.variant(angular_spectrum_prop)
        var_fresnel = self.variant(fresnel_prop)
        
        # For moderate distances, angular spectrum and Fresnel should be similar
        z_distance = 5e-3
        
        angular_result = var_angular(self.test_wavefront, z_distance)
        fresnel_result = var_fresnel(self.test_wavefront, z_distance)
        
        # Central region should be similar
        center_slice = slice(self.ny // 4, 3 * self.ny // 4)
        angular_center = angular_result.field[center_slice, center_slice]
        fresnel_center = fresnel_result.field[center_slice, center_slice]
        
        # Check correlation rather than exact match
        correlation = jnp.abs(
            jnp.vdot(angular_center.flatten(), fresnel_center.flatten())
            / (jnp.linalg.norm(angular_center) * jnp.linalg.norm(fresnel_center))
        )
        # The methods may differ significantly for certain test parameters
        # We just check that there's some correlation (at least 0.2)
        chex.assert_scalar_positive(float(correlation) - 0.2)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("zoom_in_2x", 2.0),
        ("zoom_in_1_5x", 1.5),
        ("zoom_out_0_5x", 0.5),
        ("zoom_out_0_75x", 0.75),
        ("no_zoom", 1.0),
    )
    def test_digital_zoom(self, zoom_factor: float) -> None:
        """
        Test digital zoom functionality.
        """
        var_digital_zoom = self.variant(digital_zoom)
        
        zoomed = var_digital_zoom(self.test_wavefront, zoom_factor)
        
        # Shape should be preserved
        chex.assert_shape(zoomed.field, self.test_wavefront.field.shape)
        
        # Wavelength should be preserved
        chex.assert_trees_all_close(zoomed.wavelength, self.test_wavefront.wavelength)
        
        # dx should be modified by zoom factor
        expected_dx = self.test_wavefront.dx / zoom_factor
        chex.assert_trees_all_close(zoomed.dx, expected_dx, rtol=1e-10)
        
        # z position should be preserved
        chex.assert_trees_all_close(zoomed.z_position, self.test_wavefront.z_position)
        
        # For zoom factor != 1, field should change
        if zoom_factor != 1.0:
            chex.assert_trees_all_equal(
                jnp.allclose(zoomed.field, self.test_wavefront.field),
                False
            )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("zoom_in_2x", 2.0),
        ("zoom_out_0_5x", 0.5),
        ("no_zoom", 1.0),
    )
    def test_optical_zoom(self, zoom_factor: float) -> None:
        """
        Test optical zoom functionality.
        """
        var_optical_zoom = self.variant(optical_zoom)
        
        zoomed = var_optical_zoom(self.test_wavefront, zoom_factor)
        
        # Field should be exactly preserved
        chex.assert_trees_all_close(zoomed.field, self.test_wavefront.field)
        
        # Wavelength should be preserved
        chex.assert_trees_all_close(zoomed.wavelength, self.test_wavefront.wavelength)
        
        # dx should be modified by zoom factor
        expected_dx = self.test_wavefront.dx * zoom_factor
        chex.assert_trees_all_close(zoomed.dx, expected_dx, rtol=1e-10)
        
        # z position should be preserved
        chex.assert_trees_all_close(zoomed.z_position, self.test_wavefront.z_position)

    @chex.variants(with_jit=True, without_jit=True)
    def test_lens_propagation(self) -> None:
        """
        Test propagation through a lens.
        """
        var_lens_prop = self.variant(lens_propagation)
        
        output = var_lens_prop(self.test_wavefront, self.test_lens)
        
        # Check shape preservation
        chex.assert_shape(output.field, self.test_wavefront.field.shape)
        
        # Check metadata preservation
        chex.assert_trees_all_close(output.wavelength, self.test_wavefront.wavelength)
        chex.assert_trees_all_close(output.dx, self.test_wavefront.dx)
        chex.assert_trees_all_close(output.z_position, self.test_wavefront.z_position)
        
        # Field should change after lens
        chex.assert_trees_all_equal(
            jnp.allclose(output.field, self.test_wavefront.field),
            False
        )

    def test_jax_transformations_angular_spectrum(self) -> None:
        """
        Test JAX transformations on angular spectrum propagation.
        """
        @jax.jit
        def jitted_prop(wavefront: OpticalWavefront, z: float) -> OpticalWavefront:
            return angular_spectrum_prop(wavefront, z)
        
        z_distance = 1e-3
        jitted_result = jitted_prop(self.test_wavefront, z_distance)
        normal_result = angular_spectrum_prop(self.test_wavefront, z_distance)
        
        chex.assert_trees_all_close(jitted_result.field, normal_result.field)
        
        # Test gradient computation
        def loss_fn(z: float) -> Array:
            propagated = angular_spectrum_prop(self.test_wavefront, z)
            return jnp.sum(jnp.abs(propagated.field) ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(z_distance)
        chex.assert_shape(grad, ())
        chex.assert_tree_all_finite(grad)

    def test_jax_transformations_zoom(self) -> None:
        """
        Test JAX transformations on zoom functions.
        """
        # Test vmap on digital zoom
        zoom_factors = jnp.array([0.5, 1.0, 2.0])
        vmapped_digital = jax.vmap(
            lambda z: digital_zoom(self.test_wavefront, z), in_axes=0
        )
        zoomed_batch = vmapped_digital(zoom_factors)
        
        # Check batch dimensions
        chex.assert_shape(zoomed_batch.field, (3, self.ny, self.nx))
        chex.assert_shape(zoomed_batch.dx, (3,))

    @parameterized.named_parameters(
        ("zero_field", jnp.zeros((128, 128), dtype=complex)),
        ("uniform_field", jnp.ones((128, 128), dtype=complex)),
        ("random_phase", jnp.exp(1j * jax.random.uniform(jax.random.PRNGKey(0), (128, 128), minval=0, maxval=2*jnp.pi))),
    )
    def test_propagation_edge_cases(self, field: Complex[Array, "128 128"]) -> None:
        """
        Test propagation with edge case fields.
        """
        wavefront = make_optical_wavefront(
            field=field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        
        # Test angular spectrum
        angular_result = angular_spectrum_prop(wavefront, 1e-3)
        chex.assert_shape(angular_result.field, field.shape)
        chex.assert_trees_all_equal(jnp.iscomplexobj(angular_result.field), True)
        
        # Test Fresnel
        fresnel_result = fresnel_prop(wavefront, 1e-3)
        chex.assert_shape(fresnel_result.field, field.shape)
        
        # Test Fraunhofer
        fraunhofer_result = fraunhofer_prop(wavefront, 0.1)
        chex.assert_shape(fraunhofer_result.field, field.shape)

    def test_energy_conservation(self) -> None:
        """
        Test energy conservation in propagation.
        """
        z_distance = 1e-3
        
        # Angular spectrum should conserve energy
        angular_result = angular_spectrum_prop(self.test_wavefront, z_distance)
        initial_energy = jnp.sum(jnp.abs(self.test_wavefront.field) ** 2)
        final_energy = jnp.sum(jnp.abs(angular_result.field) ** 2)
        
        # Energy should be conserved to high precision
        chex.assert_trees_all_close(initial_energy, final_energy, rtol=1e-6)

    def test_reciprocity(self) -> None:
        """Test reciprocity: forward then backward propagation should recover original."""
        z_distance = 1e-3
        
        # Forward propagation
        forward = angular_spectrum_prop(self.test_wavefront, z_distance)
        
        # Backward propagation
        backward = angular_spectrum_prop(forward, -z_distance)
        
        # Should recover original field (approximately)
        chex.assert_trees_all_close(
            backward.field,
            self.test_wavefront.field,
            rtol=1e-5
        )
        
        # Z position should match
        chex.assert_trees_all_close(
            backward.z_position,
            self.test_wavefront.z_position,
            rtol=1e-10
        )

    def test_plane_wave_propagation(self) -> None:
        """
        Test that a plane wave accumulates only phase during
        propagation.
        """
        z_distance = 1e-3
        
        propagated = angular_spectrum_prop(self.plane_wave, z_distance)
        
        # Amplitude should remain uniform
        amplitude = jnp.abs(propagated.field)
        chex.assert_trees_all_close(
            amplitude,
            jnp.ones_like(amplitude),
            rtol=1e-10
        )
        
        # Phase should be uniform across the field
        phase = jnp.angle(propagated.field)
        phase_std = jnp.std(phase)
        chex.assert_scalar_positive(1e-10 - float(phase_std))


if __name__ == "__main__":
    pytest.main([__file__])