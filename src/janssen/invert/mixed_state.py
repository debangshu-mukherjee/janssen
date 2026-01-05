"""Mixed-state ptychography for partially coherent sources.

Extended Summary
----------------
Mixed-state ptychography reconstructs both object and probe modes when
the illumination is partially coherent. The forward model sums diffraction
intensities from each coherent mode incoherently:

    I_i = Sigma_n w_n |FFT(probe_n * shift(object, r_i))|^2

All functions are fully differentiable, enabling gradient-based optimization
of object, probe modes, mode weights, and even coherence parameters.

Routine Listings
----------------
mixed_state_forward : function
    Compute predicted diffraction patterns for all positions.
mixed_state_forward_single_position : function
    Forward model for one scan position with mixed-state illumination.
mixed_state_loss : function
    Compute reconstruction loss for mixed-state ptychography.
coherence_parameterized_loss : function
    Loss function with coherence width as optimizable parameter.
mixed_state_gradient_step : function
    Single gradient descent step for mixed-state reconstruction.
mixed_state_reconstruct : function
    Run mixed-state ptychography reconstruction.

Notes
-----
The coherent mode representation enables efficient simulation of partial
coherence. For M modes on an N×N grid, memory scales as O(M * N^2) versus
O(N^4) for full mutual intensity.

The MixedStatePtychoData PyTree and make_mixed_state_ptycho_data factory
function are defined in janssen.utils.coherence_types and re-exported here
for convenience.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from janssen.coherence import gaussian_schell_model_modes
from janssen.utils import (
    CoherentModeSet,
    MixedStatePtychoData,
    ScalarInteger,
    make_coherent_mode_set,
    make_mixed_state_ptycho_data,
)


@jaxtyped(typechecker=beartype)
def _fourier_shift(
    field: Complex[Array, " H W"],
    shift_x: Float[Array, " "],
    shift_y: Float[Array, " "],
) -> Complex[Array, " H W"]:
    """Shift field using Fourier phase ramp."""
    h, w = field.shape
    ky: Float[Array, " H 1"] = jnp.fft.fftfreq(h)[:, None]
    kx: Float[Array, " 1 W"] = jnp.fft.fftfreq(w)[None, :]
    phase_ramp: Complex[Array, " H W"] = jnp.exp(
        -2j * jnp.pi * (kx * shift_x + ky * shift_y)
    )
    shifted: Complex[Array, " H W"] = jnp.fft.ifft2(
        jnp.fft.fft2(field) * phase_ramp
    )
    return shifted


@jaxtyped(typechecker=beartype)
def mixed_state_forward_single_position(
    probe_modes: Complex[Array, " M H W"],
    mode_weights: Float[Array, " M"],
    obj: Complex[Array, " H W"],
    shift_x: Float[Array, " "],
    shift_y: Float[Array, " "],
) -> Float[Array, " H W"]:
    """Forward model for one scan position with mixed-state illumination.

    Computes:
        I = Σₙ wₙ |FFT(probe_n · shift(object, r_i))|²

    Parameters
    ----------
    probe_modes : Complex[Array, " M H W"]
        Coherent probe modes.
    mode_weights : Float[Array, " M"]
        Mode weights (eigenvalues).
    obj : Complex[Array, " H W"]
        Object transmission.
    shift_x, shift_y : Float[Array, " "]
        Scan position in pixels.

    Returns
    -------
    intensity : Float[Array, " H W"]
        Diffraction intensity (incoherent sum over modes).
    """

    def _single_mode_intensity_int(
        mode: Complex[Array, " H W"],
    ) -> Float[Array, " H W"]:
        """Compute diffraction intensity for single probe mode."""
        probe_shifted: Complex[Array, " H W"] = _fourier_shift(
            mode, shift_x, shift_y
        )
        exit_wave: Complex[Array, " H W"] = obj * probe_shifted
        exit_wave_ft: Complex[Array, " H W"] = jnp.fft.fftshift(
            jnp.fft.fft2(exit_wave)
        )
        intensity: Float[Array, " H W"] = jnp.abs(exit_wave_ft) ** 2
        return intensity

    mode_intensities: Float[Array, " M H W"] = jax.vmap(
        _single_mode_intensity_int
    )(probe_modes)
    total: Float[Array, " H W"] = jnp.einsum(
        "m,mhw->hw", mode_weights, mode_intensities
    )
    return total


@jaxtyped(typechecker=beartype)
def mixed_state_forward(
    data: MixedStatePtychoData,
) -> Float[Array, " N H W"]:
    """Compute predicted diffraction patterns for all positions.

    Parameters
    ----------
    data : MixedStatePtychoData
        Ptychography data with probe modes and object.

    Returns
    -------
    predicted : Float[Array, " N H W"]
        Predicted diffraction intensities.
    """
    modes: Complex[Array, " M H W"] = data.probe_modes.modes
    weights: Float[Array, " M"] = data.probe_modes.weights
    obj: Complex[Array, " H W"] = data.sample
    positions: Float[Array, " N 2"] = data.positions

    def _forward_one_int(pos: Float[Array, " 2"]) -> Float[Array, " H W"]:
        """Compute forward model for single scan position."""
        intensity: Float[Array, " H W"] = mixed_state_forward_single_position(
            modes, weights, obj, pos[0], pos[1]
        )
        return intensity

    predicted: Float[Array, " N H W"] = jax.vmap(_forward_one_int)(positions)
    return predicted


@jaxtyped(typechecker=beartype)
def mixed_state_loss(
    data: MixedStatePtychoData,
    loss_type: str = "amplitude",
) -> Float[Array, " "]:
    """Compute reconstruction loss for mixed-state ptychography.

    Parameters
    ----------
    data : MixedStatePtychoData
        Current state with probe modes and object.
    loss_type : str
        "amplitude" for ||√I_exp - √I_pred||² (default, robust)
        "intensity" for ||I_exp - I_pred||²
        "poisson" for Poisson negative log-likelihood

    Returns
    -------
    loss : Float[Array, " "]
        Scalar loss value.
    """
    predicted: Float[Array, " N H W"] = mixed_state_forward(data)
    measured: Float[Array, " N H W"] = data.diffraction_patterns
    if loss_type == "amplitude":
        amp_meas: Float[Array, " N H W"] = jnp.sqrt(jnp.maximum(measured, 0.0))
        amp_pred: Float[Array, " N H W"] = jnp.sqrt(
            jnp.maximum(predicted, 1e-10)
        )
        loss: Float[Array, " "] = jnp.sum((amp_meas - amp_pred) ** 2)
        return loss
    if loss_type == "intensity":
        loss: Float[Array, " "] = jnp.sum((measured - predicted) ** 2)
        return loss
    if loss_type == "poisson":
        eps: float = 1e-10
        loss: Float[Array, " "] = jnp.sum(
            predicted - measured * jnp.log(predicted + eps)
        )
        return loss
    raise ValueError(f"Unknown loss_type: {loss_type}")


@jaxtyped(typechecker=beartype)
def coherence_parameterized_loss(
    coherence_width: Float[Array, " "],
    sample: Complex[Array, " H W"],
    diffraction_patterns: Float[Array, " N H W"],
    positions: Float[Array, " N 2"],
    beam_width: Float[Array, " "],
    wavelength: Float[Array, " "],
    dx: Float[Array, " "],
    num_modes: ScalarInteger = 10,
) -> Float[Array, " "]:
    """Loss function with coherence width as the optimizable parameter.

    This enables gradient-based recovery of source coherence:
        σ_c* = argmin_{σ_c} L(σ_c; I_measured)

    Parameters
    ----------
    coherence_width : Float[Array, " "]
        Coherence width to optimize.
    sample : Complex[Array, " H W"]
        Object (fixed or jointly optimized).
    diffraction_patterns : Float[Array, " N H W"]
        Measured data.
    positions : Float[Array, " N 2"]
        Scan positions.
    beam_width : Float[Array, " "]
        Known beam intensity width.
    wavelength : Float[Array, " "]
        Wavelength.
    dx : Float[Array, " "]
        Pixel size.
    num_modes : int
        Number of GSM modes.

    Returns
    -------
    loss : Float[Array, " "]
        Scalar loss.

    Notes
    -----
    The gradient ∂L/∂σ_c flows through:
    1. σ_c → eigenvalues λₙ (analytical for GSM)
    2. σ_c → mode width w_mode → mode shapes φₙ
    3. modes → forward model → loss

    No hand-derived update rules needed!
    """
    hh: int = sample.shape[0]
    ww: int = sample.shape[1]
    probe_modes: CoherentModeSet = gaussian_schell_model_modes(
        wavelength=wavelength,
        dx=dx,
        grid_size=(hh, ww),
        beam_width=beam_width,
        coherence_width=coherence_width,
        num_modes=num_modes,
    )
    data: MixedStatePtychoData = make_mixed_state_ptycho_data(
        diffraction_patterns=diffraction_patterns,
        probe_modes=probe_modes,
        sample=sample,
        positions=positions,
        wavelength=wavelength,
        dx=dx,
    )
    loss: Float[Array, " "] = mixed_state_loss(data, loss_type="amplitude")
    return loss


@jaxtyped(typechecker=beartype)
def mixed_state_gradient_step(
    data: MixedStatePtychoData,
    learning_rate: float = 1e-3,
    update_object: bool = True,
    update_modes: bool = True,
    update_weights: bool = False,
) -> MixedStatePtychoData:
    """Single gradient descent step for mixed-state reconstruction.

    Parameters
    ----------
    data : MixedStatePtychoData
        Current state.
    learning_rate : float
        Step size.
    update_object : bool
        Whether to update object.
    update_modes : bool
        Whether to update probe mode fields.
    update_weights : bool
        Whether to update mode weights.

    Returns
    -------
    updated_data : MixedStatePtychoData
        State after one gradient step.
    """

    def _loss_fn_int(
        sample: Complex[Array, " Hs Ws"],
        modes: Complex[Array, " M H W"],
        weights: Float[Array, " M"],
    ) -> Float[Array, " "]:
        """Compute mixed-state loss for gradient computation."""
        updated_probe: CoherentModeSet = make_coherent_mode_set(
            modes=modes,
            weights=weights,
            wavelength=data.probe_modes.wavelength,
            dx=data.probe_modes.dx,
            z_position=data.probe_modes.z_position,
            polarization=data.probe_modes.polarization,
        )
        updated_data: MixedStatePtychoData = make_mixed_state_ptycho_data(
            diffraction_patterns=data.diffraction_patterns,
            probe_modes=updated_probe,
            sample=sample,
            positions=data.positions,
            wavelength=data.wavelength,
            dx=data.dx,
        )
        loss: Float[Array, " "] = mixed_state_loss(updated_data)
        return loss

    grad_fn = jax.grad(_loss_fn_int, argnums=(0, 1, 2))
    grad_sample: Complex[Array, " Hs Ws"]
    grad_modes: Complex[Array, " M H W"]
    grad_weights: Float[Array, " M"]
    grad_sample, grad_modes, grad_weights = grad_fn(
        data.sample, data.probe_modes.modes, data.probe_modes.weights
    )
    new_sample: Complex[Array, " Hs Ws"] = jax.lax.cond(
        update_object,
        lambda: data.sample - learning_rate * grad_sample,
        lambda: data.sample,
    )
    new_modes: Complex[Array, " M H W"] = jax.lax.cond(
        update_modes,
        lambda: data.probe_modes.modes - learning_rate * grad_modes,
        lambda: data.probe_modes.modes,
    )
    new_weights: Float[Array, " M"] = jax.lax.cond(
        update_weights,
        lambda: data.probe_modes.weights - learning_rate * grad_weights,
        lambda: data.probe_modes.weights,
    )
    new_weights = jnp.maximum(new_weights, 1e-10)
    new_weights = new_weights / jnp.sum(new_weights)
    updated_probe: CoherentModeSet = make_coherent_mode_set(
        modes=new_modes,
        weights=new_weights,
        wavelength=data.probe_modes.wavelength,
        dx=data.probe_modes.dx,
        z_position=data.probe_modes.z_position,
        polarization=data.probe_modes.polarization,
        normalize_weights=False,
    )
    updated_data: MixedStatePtychoData = make_mixed_state_ptycho_data(
        diffraction_patterns=data.diffraction_patterns,
        probe_modes=updated_probe,
        sample=new_sample,
        positions=data.positions,
        wavelength=data.wavelength,
        dx=data.dx,
    )
    return updated_data


@jaxtyped(typechecker=beartype)
def mixed_state_reconstruct(
    data: MixedStatePtychoData,
    num_iterations: ScalarInteger = 100,
    learning_rate: float = 1e-3,
    update_object: bool = True,
    update_modes: bool = True,
    update_weights: bool = False,
) -> tuple[MixedStatePtychoData, Float[Array, " I"]]:
    """Run mixed-state ptychography reconstruction.

    Parameters
    ----------
    data : MixedStatePtychoData
        Initial state with probe modes and object estimate.
    num_iterations : int
        Number of gradient descent iterations.
    learning_rate : float
        Step size.
    update_object : bool
        Whether to reconstruct object.
    update_modes : bool
        Whether to reconstruct probe modes.
    update_weights : bool
        Whether to learn mode weights.

    Returns
    -------
    final_data : MixedStatePtychoData
        Reconstructed state.
    loss_history : Float[Array, " I"]
        Loss at each iteration.
    """

    def _step_int(
        carry: MixedStatePtychoData, _: None
    ) -> tuple[MixedStatePtychoData, Float[Array, " "]]:
        """Single gradient descent iteration."""
        current_data: MixedStatePtychoData = carry
        loss: Float[Array, " "] = mixed_state_loss(current_data)
        updated: MixedStatePtychoData = mixed_state_gradient_step(
            current_data,
            learning_rate,
            update_object,
            update_modes,
            update_weights,
        )
        return updated, loss

    final_data: MixedStatePtychoData
    loss_history: Float[Array, " I"]
    final_data, loss_history = jax.lax.scan(
        _step_int, data, None, length=int(num_iterations)
    )
    return final_data, loss_history
