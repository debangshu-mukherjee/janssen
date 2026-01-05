"""Tests for mixed-state ptychography in janssen.invert.mixed_state."""

import chex
import jax
import jax.numpy as jnp

from janssen.cohere import gaussian_schell_model_modes
from janssen.invert import (
    MixedStatePtychoData,
    coherence_parameterized_loss,
    make_mixed_state_ptycho_data,
    mixed_state_forward,
    mixed_state_forward_single_position,
    mixed_state_gradient_step,
    mixed_state_loss,
    mixed_state_reconstruct,
)


def _make_test_data(
    grid_size: tuple[int, int] = (16, 16),
    num_positions: int = 4,
    num_modes: int = 3,
) -> MixedStatePtychoData:
    """Create test MixedStatePtychoData for unit tests."""
    hh, ww = grid_size
    wavelength = 633e-9
    dx = 1e-6
    beam_width = 5e-6
    coherence_width = 10e-6

    probe_modes = gaussian_schell_model_modes(
        wavelength=wavelength,
        dx=dx,
        grid_size=grid_size,
        beam_width=beam_width,
        coherence_width=coherence_width,
        num_modes=num_modes,
    )

    sample = jnp.ones((hh, ww), dtype=jnp.complex128)

    positions = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]][:num_positions]
    )

    diffraction_patterns = jnp.ones((num_positions, hh, ww))

    return make_mixed_state_ptycho_data(
        diffraction_patterns=diffraction_patterns,
        probe_modes=probe_modes,
        sample=sample,
        positions=positions,
        wavelength=wavelength,
        dx=dx,
    )


class TestMixedStatePtychoData(chex.TestCase):
    """Test MixedStatePtychoData PyTree structure."""

    @chex.variants(without_jit=True)
    def test_creation(self) -> None:
        """Test that MixedStatePtychoData can be created."""
        var_fn = self.variant(_make_test_data)
        data = var_fn()
        assert isinstance(data, MixedStatePtychoData)
        assert hasattr(data, "diffraction_patterns")
        assert hasattr(data, "probe_modes")
        assert hasattr(data, "sample")
        assert hasattr(data, "positions")

    @chex.variants(without_jit=True)
    def test_pytree_flatten_unflatten(self) -> None:
        """Test that data survives PyTree round-trip."""
        var_fn = self.variant(_make_test_data)
        data = var_fn()
        leaves, treedef = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        chex.assert_trees_all_close(
            data.diffraction_patterns, reconstructed.diffraction_patterns
        )
        chex.assert_trees_all_close(data.sample, reconstructed.sample)


class TestMixedStateForwardSinglePosition(chex.TestCase):
    """Test mixed_state_forward_single_position function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        hh, ww = 16, 16
        num_modes = 3
        probe_modes = jnp.ones((num_modes, hh, ww), dtype=jnp.complex128)
        mode_weights = jnp.array([0.6, 0.3, 0.1])
        obj = jnp.ones((hh, ww), dtype=jnp.complex128)
        shift_x = jnp.array(0.0)
        shift_y = jnp.array(0.0)
        var_fn = self.variant(mixed_state_forward_single_position)
        result = var_fn(probe_modes, mode_weights, obj, shift_x, shift_y)
        chex.assert_shape(result, (hh, ww))

    @chex.variants(without_jit=True)
    def test_output_non_negative(self) -> None:
        """Test that output intensity is non-negative."""
        hh, ww = 16, 16
        num_modes = 3
        probe_modes = jnp.ones((num_modes, hh, ww), dtype=jnp.complex128)
        mode_weights = jnp.array([0.6, 0.3, 0.1])
        obj = jnp.ones((hh, ww), dtype=jnp.complex128)
        shift_x = jnp.array(0.0)
        shift_y = jnp.array(0.0)
        var_fn = self.variant(mixed_state_forward_single_position)
        result = var_fn(probe_modes, mode_weights, obj, shift_x, shift_y)
        assert jnp.all(result >= 0)


class TestMixedStateForward(chex.TestCase):
    """Test mixed_state_forward function."""

    @chex.variants(without_jit=True)
    def test_output_shape(self) -> None:
        """Test that output has correct shape for all positions."""
        data = _make_test_data(grid_size=(16, 16), num_positions=4)
        var_fn = self.variant(mixed_state_forward)
        result = var_fn(data)
        chex.assert_shape(result, (4, 16, 16))

    @chex.variants(without_jit=True)
    def test_output_non_negative(self) -> None:
        """Test that all predicted intensities are non-negative."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_forward)
        result = var_fn(data)
        assert jnp.all(result >= 0)


class TestMixedStateLoss(chex.TestCase):
    """Test mixed_state_loss function."""

    @chex.variants(without_jit=True)
    def test_output_scalar(self) -> None:
        """Test that loss is a scalar."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_loss)
        loss = var_fn(data)
        chex.assert_shape(loss, ())

    @chex.variants(without_jit=True)
    def test_loss_non_negative(self) -> None:
        """Test that loss is non-negative."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_loss)
        loss = var_fn(data)
        assert loss >= 0

    @chex.variants(without_jit=True)
    def test_amplitude_loss_type(self) -> None:
        """Test amplitude loss type."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_loss)
        loss = var_fn(data, loss_type="amplitude")
        assert loss >= 0

    @chex.variants(without_jit=True)
    def test_intensity_loss_type(self) -> None:
        """Test intensity loss type."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_loss)
        loss = var_fn(data, loss_type="intensity")
        assert loss >= 0

    @chex.variants(without_jit=True)
    def test_poisson_loss_type(self) -> None:
        """Test Poisson loss type."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_loss)
        loss = var_fn(data, loss_type="poisson")
        chex.assert_shape(loss, ())


class TestCoherenceParameterizedLoss(chex.TestCase):
    """Test coherence_parameterized_loss function."""

    @chex.variants(without_jit=True)
    def test_output_scalar(self) -> None:
        """Test that loss is a scalar."""
        hh, ww = 16, 16
        coherence_width = jnp.array(10e-6)
        sample = jnp.ones((hh, ww), dtype=jnp.complex128)
        diffraction_patterns = jnp.ones((4, hh, ww))
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        beam_width = jnp.array(5e-6)
        wavelength = jnp.array(633e-9)
        dx = jnp.array(1e-6)
        var_fn = self.variant(coherence_parameterized_loss)
        loss = var_fn(
            coherence_width,
            sample,
            diffraction_patterns,
            positions,
            beam_width,
            wavelength,
            dx,
            num_modes=3,
        )
        chex.assert_shape(loss, ())

    @chex.variants(without_jit=True)
    def test_differentiable_wrt_coherence_width(self) -> None:
        """Test that loss is differentiable w.r.t. coherence_width."""
        hh, ww = 16, 16
        coherence_width = jnp.array(10e-6)
        sample = jnp.ones((hh, ww), dtype=jnp.complex128)
        diffraction_patterns = jnp.ones((4, hh, ww))
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        beam_width = jnp.array(5e-6)
        wavelength = jnp.array(633e-9)
        dx = jnp.array(1e-6)

        def loss_fn(cw):
            return coherence_parameterized_loss(
                cw,
                sample,
                diffraction_patterns,
                positions,
                beam_width,
                wavelength,
                dx,
                num_modes=3,
            )

        var_fn = self.variant(jax.grad(loss_fn))
        grad = var_fn(coherence_width)
        assert jnp.isfinite(grad)


class TestMixedStateGradientStep(chex.TestCase):
    """Test mixed_state_gradient_step function."""

    @chex.variants(without_jit=True)
    def test_returns_same_type(self) -> None:
        """Test that output is MixedStatePtychoData."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_gradient_step)
        updated = var_fn(data)
        assert isinstance(updated, MixedStatePtychoData)

    @chex.variants(without_jit=True)
    def test_sample_updated(self) -> None:
        """Test that sample is updated when update_object=True."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_gradient_step)
        updated = var_fn(data, update_object=True, update_modes=False)
        diff = jnp.sum(jnp.abs(updated.sample - data.sample))
        assert diff > 0

    @chex.variants(without_jit=True)
    def test_weights_normalized(self) -> None:
        """Test that weights remain normalized after update."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_gradient_step)
        updated = var_fn(data, update_weights=True)
        weight_sum = jnp.sum(updated.probe_modes.weights)
        chex.assert_trees_all_close(weight_sum, 1.0, atol=1e-10)


class TestMixedStateReconstruct(chex.TestCase):
    """Test mixed_state_reconstruct function."""

    @chex.variants(without_jit=True)
    def test_returns_tuple(self) -> None:
        """Test that output is (data, loss_history) tuple."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_reconstruct)
        result = var_fn(data, num_iterations=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @chex.variants(without_jit=True)
    def test_loss_history_shape(self) -> None:
        """Test that loss history has correct length."""
        data = _make_test_data()
        num_iterations = 5
        var_fn = self.variant(mixed_state_reconstruct)
        _, loss_history = var_fn(data, num_iterations=num_iterations)
        chex.assert_shape(loss_history, (num_iterations,))

    @chex.variants(without_jit=True)
    def test_loss_decreases(self) -> None:
        """Test that loss generally decreases during reconstruction."""
        data = _make_test_data()
        var_fn = self.variant(mixed_state_reconstruct)
        _, loss_history = var_fn(data, num_iterations=10, learning_rate=1e-4)
        assert loss_history[-1] <= loss_history[0]
