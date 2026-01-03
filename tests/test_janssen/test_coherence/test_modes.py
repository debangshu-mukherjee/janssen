"""Tests for coherent mode functions in janssen.coherence.modes."""

import chex
import jax.numpy as jnp
from absl.testing import parameterized

from janssen.coherence.modes import (
    effective_mode_count,
    eigenmode_decomposition,
    gaussian_schell_model_modes,
    hermite_gaussian_modes,
    modes_from_wavefront,
    mutual_intensity_from_modes,
)
from janssen.utils import make_mutual_intensity


class TestHermiteGaussianModes(chex.TestCase):
    """Test hermite_gaussian_modes function.

    Note: These tests use without_jit only because the function has static
    arguments (grid_size, max_order) that must remain concrete. The internal
    implementation is JIT-compiled with static_argnums.
    """

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 2
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")
        assert hasattr(mode_set, "wavelength")

    @chex.variants(without_jit=True)
    def test_mode_count(self) -> None:
        """Test correct number of modes for given max_order."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 3
        expected_modes = (max_order + 1) * (max_order + 2) // 2
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        chex.assert_shape(mode_set.modes, (expected_modes, 32, 32))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights sum to 1."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 2
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_modes_normalized(self) -> None:
        """Test that each mode has unit energy."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 2
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        for i in range(mode_set.modes.shape[0]):
            mode_energy = jnp.sum(jnp.abs(mode_set.modes[i]) ** 2)
            chex.assert_trees_all_close(mode_energy, 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_custom_weights(self) -> None:
        """Test that custom weights are applied."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 1
        num_modes = 3
        custom_weights = jnp.array([0.7, 0.2, 0.1])
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_waist, max_order, custom_weights
        )
        chex.assert_trees_all_close(
            mode_set.weights, custom_weights / jnp.sum(custom_weights), atol=1e-10
        )

    @chex.variants(without_jit=True)
    def test_wavelength_stored(self) -> None:
        """Test that wavelength is correctly stored."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_waist = 10e-6
        max_order = 1
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        chex.assert_trees_all_close(mode_set.wavelength, wavelength, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_fundamental_mode_gaussian(self) -> None:
        """Test that fundamental mode (0,0) is Gaussian-like."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (64, 64)
        beam_waist = 15e-6
        max_order = 0
        var_fn = self.variant(hermite_gaussian_modes)
        mode_set = var_fn(wavelength, dx, grid_size, beam_waist, max_order)
        mode = mode_set.modes[0]
        center = grid_size[0] // 2
        center_value = jnp.abs(mode[center, center])
        edge_value = jnp.abs(mode[center, 0])
        assert center_value > edge_value


class TestGaussianSchellModelModes(chex.TestCase):
    """Test gaussian_schell_model_modes function.

    Note: These tests use without_jit only because the function has static
    arguments (grid_size, num_modes) that must remain concrete. The internal
    implementation is JIT-compiled with static_argnums.
    """

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_width = 50e-6
        coherence_width = 20e-6
        num_modes = 5
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 48)
        beam_width = 50e-6
        coherence_width = 20e-6
        num_modes = 7
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        chex.assert_shape(mode_set.modes, (num_modes, 32, 48))
        chex.assert_shape(mode_set.weights, (num_modes,))

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights sum to 1."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_width = 50e-6
        coherence_width = 20e-6
        num_modes = 10
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        chex.assert_trees_all_close(jnp.sum(mode_set.weights), 1.0, atol=1e-10)

    @chex.variants(without_jit=True)
    def test_weights_decrease(self) -> None:
        """Test that eigenvalues decrease with mode index."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_width = 50e-6
        coherence_width = 20e-6
        num_modes = 10
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        weights = mode_set.weights
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    @chex.variants(without_jit=True)
    def test_high_coherence_single_mode(self) -> None:
        """Test that high coherence gives dominant first mode."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_width = 50e-6
        coherence_width = 500e-6
        num_modes = 5
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        first_mode_fraction = mode_set.weights[0]
        assert first_mode_fraction > 0.9

    @chex.variants(without_jit=True)
    def test_low_coherence_many_modes(self) -> None:
        """Test that low coherence spreads power across modes."""
        wavelength = 633e-9
        dx = 1e-6
        grid_size = (32, 32)
        beam_width = 50e-6
        coherence_width = 5e-6
        num_modes = 10
        var_fn = self.variant(gaussian_schell_model_modes)
        mode_set = var_fn(
            wavelength, dx, grid_size, beam_width, coherence_width, num_modes
        )
        first_mode_fraction = mode_set.weights[0]
        assert first_mode_fraction < 0.5


class TestEigenmodeDecomposition(chex.TestCase):
    """Test eigenmode_decomposition function.

    Note: These tests use without_jit only because the function has static
    arguments (num_modes, grid dimensions) that must remain concrete.
    """

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        hh, ww = 8, 8
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        mutual_intensity = make_mutual_intensity(
            j_matrix=j_matrix, wavelength=633e-9, dx=1e-6, z_position=0.0
        )
        num_modes = 3
        var_fn = self.variant(eigenmode_decomposition)
        mode_set = var_fn(mutual_intensity, num_modes)
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(without_jit=True)
    def test_mode_shapes(self) -> None:
        """Test that modes have correct shape."""
        hh, ww = 8, 10
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        mutual_intensity = make_mutual_intensity(
            j_matrix=j_matrix, wavelength=633e-9, dx=1e-6, z_position=0.0
        )
        num_modes = 5
        var_fn = self.variant(eigenmode_decomposition)
        mode_set = var_fn(mutual_intensity, num_modes)
        chex.assert_shape(mode_set.modes, (num_modes, hh, ww))
        chex.assert_shape(mode_set.weights, (num_modes,))

    @chex.variants(without_jit=True)
    def test_fully_coherent_single_mode(self) -> None:
        """Test that fully coherent field gives single dominant mode."""
        hh, ww = 8, 8
        field = jnp.exp(1j * 0.5) * jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        mutual_intensity = make_mutual_intensity(
            j_matrix=j_matrix, wavelength=633e-9, dx=1e-6, z_position=0.0
        )
        num_modes = 5
        var_fn = self.variant(eigenmode_decomposition)
        mode_set = var_fn(mutual_intensity, num_modes)
        first_mode_fraction = mode_set.weights[0] / jnp.sum(mode_set.weights)
        chex.assert_trees_all_close(first_mode_fraction, 1.0, atol=1e-6)

    @chex.variants(without_jit=True)
    def test_weights_non_negative(self) -> None:
        """Test that all weights are non-negative."""
        hh, ww = 8, 8
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        j_matrix = jnp.einsum("ij,kl->ijkl", jnp.conj(field), field)
        mutual_intensity = make_mutual_intensity(
            j_matrix=j_matrix, wavelength=633e-9, dx=1e-6, z_position=0.0
        )
        num_modes = 3
        var_fn = self.variant(eigenmode_decomposition)
        mode_set = var_fn(mutual_intensity, num_modes)
        assert jnp.all(mode_set.weights >= 0)


class TestEffectiveModeCount(chex.TestCase):
    """Test effective_mode_count function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        weights = jnp.array([0.5, 0.3, 0.2])
        var_fn = self.variant(effective_mode_count)
        m_eff = var_fn(weights)
        chex.assert_shape(m_eff, ())

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_mode(self) -> None:
        """Test that single mode gives M_eff = 1."""
        weights = jnp.array([1.0])
        var_fn = self.variant(effective_mode_count)
        m_eff = var_fn(weights)
        chex.assert_trees_all_close(m_eff, 1.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_equal_weights(self) -> None:
        """Test that N equal weights gives M_eff = N."""
        n_modes = 5
        weights = jnp.ones(n_modes) / n_modes
        var_fn = self.variant(effective_mode_count)
        m_eff = var_fn(weights)
        chex.assert_trees_all_close(m_eff, float(n_modes), atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_dominated_by_first(self) -> None:
        """Test that dominant first mode gives M_eff close to 1."""
        weights = jnp.array([0.99, 0.005, 0.005])
        var_fn = self.variant(effective_mode_count)
        m_eff = var_fn(weights)
        assert m_eff < 1.1

    @chex.variants(with_jit=True, without_jit=True)
    def test_between_bounds(self) -> None:
        """Test that M_eff is between 1 and N."""
        weights = jnp.array([0.5, 0.3, 0.15, 0.05])
        n_modes = len(weights)
        var_fn = self.variant(effective_mode_count)
        m_eff = var_fn(weights)
        assert 1.0 <= m_eff <= n_modes


class TestModesFromWavefront(chex.TestCase):
    """Test modes_from_wavefront function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a CoherentModeSet."""
        field = jnp.ones((32, 32), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        var_fn = self.variant(modes_from_wavefront)
        mode_set = var_fn(field, wavelength, dx)
        assert hasattr(mode_set, "modes")
        assert hasattr(mode_set, "weights")

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_mode(self) -> None:
        """Test that output has exactly one mode."""
        field = jnp.ones((32, 32), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        var_fn = self.variant(modes_from_wavefront)
        mode_set = var_fn(field, wavelength, dx)
        chex.assert_shape(mode_set.modes, (1, 32, 32))
        chex.assert_shape(mode_set.weights, (1,))

    @chex.variants(with_jit=True, without_jit=True)
    def test_weight_is_one(self) -> None:
        """Test that single mode weight is 1."""
        field = jnp.ones((32, 32), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        var_fn = self.variant(modes_from_wavefront)
        mode_set = var_fn(field, wavelength, dx)
        chex.assert_trees_all_close(mode_set.weights[0], 1.0, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_field_preserved(self) -> None:
        """Test that input field is preserved in mode."""
        field = jnp.exp(1j * jnp.linspace(0, jnp.pi, 32 * 32)).reshape(32, 32)
        wavelength = 633e-9
        dx = 1e-6
        var_fn = self.variant(modes_from_wavefront)
        mode_set = var_fn(field, wavelength, dx)
        chex.assert_trees_all_close(mode_set.modes[0], field, atol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_z_position_stored(self) -> None:
        """Test that z_position is correctly stored."""
        field = jnp.ones((32, 32), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        z_position = 0.05
        var_fn = self.variant(modes_from_wavefront)
        mode_set = var_fn(field, wavelength, dx, z_position)
        chex.assert_trees_all_close(mode_set.z_position, z_position, rtol=1e-10)


class TestMutualIntensityFromModes(chex.TestCase):
    """Test mutual_intensity_from_modes function.

    Note: These tests use without_jit only because the function has static
    arguments (num_modes) that must remain concrete.
    """

    @chex.variants(without_jit=True)
    def test_output_type(self) -> None:
        """Test that output is a MutualIntensity."""
        field = jnp.ones((8, 8), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        mode_set = modes_from_wavefront(field, wavelength, dx)
        var_fn = self.variant(mutual_intensity_from_modes)
        mi = var_fn(mode_set)
        assert hasattr(mi, "j_matrix")
        assert hasattr(mi, "wavelength")

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that j_matrix has correct shape."""
        hh, ww = 8, 10
        field = jnp.ones((hh, ww), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        mode_set = modes_from_wavefront(field, wavelength, dx)
        var_fn = self.variant(mutual_intensity_from_modes)
        mi = var_fn(mode_set)
        chex.assert_shape(mi.j_matrix, (hh, ww, hh, ww))

    @chex.variants(without_jit=True)
    def test_hermitian_symmetry(self) -> None:
        """Test that J(r1, r2) = J*(r2, r1)."""
        field = jnp.exp(1j * jnp.linspace(0, jnp.pi, 64)).reshape(8, 8)
        wavelength = 633e-9
        dx = 1e-6
        mode_set = modes_from_wavefront(field, wavelength, dx)
        var_fn = self.variant(mutual_intensity_from_modes)
        mi = var_fn(mode_set)
        j = mi.j_matrix
        j_transpose = jnp.transpose(j, (2, 3, 0, 1))
        chex.assert_trees_all_close(j, jnp.conj(j_transpose), atol=1e-10)

    @chex.variants(without_jit=True)
    def test_diagonal_gives_intensity(self) -> None:
        """Test that diagonal J(r, r) gives intensity."""
        field = 2.0 * jnp.ones((8, 8), dtype=jnp.complex128)
        wavelength = 633e-9
        dx = 1e-6
        mode_set = modes_from_wavefront(field, wavelength, dx)
        var_fn = self.variant(mutual_intensity_from_modes)
        mi = var_fn(mode_set)
        expected_intensity = jnp.abs(field) ** 2
        for i in range(8):
            for j in range(8):
                chex.assert_trees_all_close(
                    jnp.real(mi.j_matrix[i, j, i, j]),
                    expected_intensity[i, j],
                    atol=1e-10,
                )
