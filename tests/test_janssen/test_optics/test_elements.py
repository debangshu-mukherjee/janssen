import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Complex, Float

from janssen.optics.elements import (
    amplitude_grating_binary,
    apply_phase_mask,
    apply_phase_mask_fn,
    beam_splitter,
    half_waveplate,
    mirror_reflection,
    nd_filter,
    phase_grating_blazed_elliptical,
    phase_grating_sawtooth,
    phase_grating_sine,
    polarizer_jones,
    prism_phase_ramp,
    quarter_waveplate,
    waveplate_jones,
)
from janssen.types import make_optical_wavefront


class TestElements(chex.TestCase, parameterized.TestCase):
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
        self.field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.wavefront = make_optical_wavefront(
            field=self.field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        self.polarized_field = jnp.ones((self.ny, self.nx, 2), dtype=complex)
        self.polarized_wavefront = make_optical_wavefront(
            field=self.polarized_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_prism_phase_ramp_small_angle(self) -> None:
        """Test prism phase ramp with small angle approximation."""
        var_prism_phase_ramp = self.variant(prism_phase_ramp)
        deflect_x = 0.001
        deflect_y = 0.002
        output = var_prism_phase_ramp(
            self.wavefront, deflect_x, deflect_y, use_small_angle=True
        )
        chex.assert_shape(output.field, self.field.shape)
        chex.assert_trees_all_equal(output.wavelength, self.wavelength)
        chex.assert_trees_all_equal(output.dx, self.dx)
        k = 2.0 * jnp.pi / self.wavelength
        expected_phase = k * (deflect_x * self.xx + deflect_y * self.yy)
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_prism_phase_ramp_spatial_freq(self) -> None:
        """Test prism phase ramp with direct spatial frequency."""
        var_prism_phase_ramp = self.variant(prism_phase_ramp)
        kx = 1000.0
        ky = 2000.0
        output = var_prism_phase_ramp(
            self.wavefront, kx, ky, use_small_angle=False
        )
        expected_phase = kx * self.xx + ky * self.yy
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("50_50_split", 0.5, 0.5, True),
        ("70_30_split", 0.7, 0.3, True),
        ("90_10_split", 0.9, 0.1, True),
        ("unnormalized", 0.6, 0.6, False),
    )
    def test_beam_splitter(
        self, t2: float, r2: float, normalize: bool
    ) -> None:
        """Test beam splitter with various splitting ratios."""
        var_beam_splitter = self.variant(beam_splitter)
        wf_t, wf_r = var_beam_splitter(self.wavefront, t2, r2, normalize)
        chex.assert_shape(wf_t.field, self.field.shape)
        chex.assert_shape(wf_r.field, self.field.shape)
        power_in = jnp.sum(jnp.abs(self.field) ** 2)
        power_t = jnp.sum(jnp.abs(wf_t.field) ** 2)
        power_r = jnp.sum(jnp.abs(wf_r.field) ** 2)
        if normalize:
            power_out = power_t + power_r
            chex.assert_trees_all_close(power_out, power_in, rtol=1e-5)
            t_actual = jnp.sqrt(power_t / power_in)
            r_actual = jnp.sqrt(power_r / power_in)
            expected_sum = 1.0
            actual_sum = t_actual**2 + r_actual**2
            chex.assert_trees_all_close(actual_sum, expected_sum, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("flip_x_only", True, False, True, True),
        ("flip_y_only", False, True, True, True),
        ("flip_both", True, True, True, True),
        ("no_flip", False, False, True, True),
        ("no_pi_phase", True, False, False, True),
        ("no_conjugate", True, False, True, False),
    )
    def test_mirror_reflection(
        self, flip_x: bool, flip_y: bool, add_pi_phase: bool, conjugate: bool
    ) -> None:
        """Test mirror reflection with various configurations."""
        var_mirror_reflection = self.variant(mirror_reflection)
        test_field = jnp.arange(self.nx * self.ny, dtype=complex).reshape(
            self.ny, self.nx
        )
        test_wavefront = make_optical_wavefront(
            field=test_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        output = var_mirror_reflection(
            test_wavefront, flip_x, flip_y, add_pi_phase, conjugate
        )
        expected = test_field.copy()
        if flip_x:
            expected = jnp.flip(expected, axis=-1)
        if flip_y:
            expected = jnp.flip(expected, axis=-2)
        if conjugate:
            expected = jnp.conjugate(expected)
        if add_pi_phase:
            expected = -expected
        chex.assert_trees_all_close(output.field, expected, rtol=1e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_phase_grating_sine(self) -> None:
        """Test sinusoidal phase grating."""
        var_phase_grating_sine = self.variant(phase_grating_sine)
        period = 10e-6
        depth = jnp.pi
        theta = 0.0
        output = var_phase_grating_sine(self.wavefront, period, depth, theta)
        expected_phase = depth * jnp.sin(2.0 * jnp.pi * self.xx / period)
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("theta_45", jnp.pi / 4),
        ("theta_90", jnp.pi / 2),
        ("theta_135", 3 * jnp.pi / 4),
    )
    def test_phase_grating_sine_rotated(self, theta: float) -> None:
        """Test sinusoidal phase grating with rotation."""
        var_phase_grating_sine = self.variant(phase_grating_sine)
        period = 10e-6
        depth = jnp.pi
        output = var_phase_grating_sine(self.wavefront, period, depth, theta)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        uu = ct * self.xx + st * self.yy
        expected_phase = depth * jnp.sin(2.0 * jnp.pi * uu / period)
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("duty_25", 0.25, 1.0, 0.0),
        ("duty_50", 0.5, 1.0, 0.0),
        ("duty_75", 0.75, 1.0, 0.0),
        ("partial_trans", 0.5, 0.8, 0.2),
    )
    def test_amplitude_grating_binary(
        self, duty_cycle: float, trans_high: float, trans_low: float
    ) -> None:
        """Test binary amplitude grating with various duty cycles."""
        var_amplitude_grating_binary = self.variant(amplitude_grating_binary)
        period = 10e-6
        theta = 0.0
        output = var_amplitude_grating_binary(
            self.wavefront, period, duty_cycle, theta, trans_high, trans_low
        )
        frac = (self.xx / period) - jnp.floor(self.xx / period)
        mask_high = frac < duty_cycle
        expected_trans = jnp.where(mask_high, trans_high, trans_low)
        expected_field = self.field * expected_trans
        chex.assert_trees_all_close(output.field, expected_field, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_phase_grating_sawtooth(self) -> None:
        """Test sawtooth (blazed) phase grating."""
        var_phase_grating_sawtooth = self.variant(phase_grating_sawtooth)
        period = 10e-6
        depth = 2 * jnp.pi
        theta = 0.0
        output = var_phase_grating_sawtooth(
            self.wavefront, period, depth, theta
        )
        frac = (self.xx / period) - jnp.floor(self.xx / period)
        expected_phase = depth * frac
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_phase_mask(self) -> None:
        """Test applying arbitrary phase mask."""
        var_apply_phase_mask = self.variant(apply_phase_mask)
        phase_map = jnp.pi * jnp.ones((self.ny, self.nx))
        output = var_apply_phase_mask(self.wavefront, phase_map)
        expected_field = self.field * jnp.exp(1j * phase_map)
        chex.assert_trees_all_close(output.field, expected_field, rtol=1e-10)

    @chex.variants(without_jit=True)
    def test_apply_phase_mask_fn(self) -> None:
        """Test applying phase mask from function.

        Note: Only tested without JIT since phase_fn is a Python callable
        that cannot be traced by JAX without static_argnames.
        """
        var_apply_phase_mask_fn = self.variant(apply_phase_mask_fn)

        def phase_fn(
            xx: Float[Array, " H W"], yy: Float[Array, " H W"]
        ) -> Float[Array, " H W"]:
            _ = yy  # Required by signature but not used
            return jnp.pi * (xx / xx.max())

        output = var_apply_phase_mask_fn(self.wavefront, phase_fn)
        expected_phase = phase_fn(self.xx, self.yy)
        expected_field = self.field * jnp.exp(1j * expected_phase)
        chex.assert_trees_all_close(output.field, expected_field, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("theta_0", 0.0),
        ("theta_45", jnp.pi / 4),
        ("theta_90", jnp.pi / 2),
    )
    def test_polarizer_jones(self, theta: float) -> None:
        """Test linear polarizer at various angles."""
        var_polarizer_jones = self.variant(polarizer_jones)
        output = var_polarizer_jones(self.polarized_wavefront, theta)
        chex.assert_shape(output.field, self.polarized_field.shape)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        ex_in, ey_in = (
            self.polarized_field[..., 0],
            self.polarized_field[..., 1],
        )
        e_par = ex_in * ct + ey_in * st
        ex_expected = e_par * ct
        ey_expected = e_par * st
        chex.assert_trees_all_close(
            output.field[..., 0], ex_expected, rtol=1e-5
        )
        chex.assert_trees_all_close(
            output.field[..., 1], ey_expected, rtol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("quarter_wave", jnp.pi / 2, 0.0),
        ("half_wave", jnp.pi, 0.0),
        ("arbitrary", jnp.pi / 3, jnp.pi / 6),
    )
    def test_waveplate_jones(self, delta: float, theta: float) -> None:
        """Test waveplate with various retardances and orientations."""
        var_waveplate_jones = self.variant(waveplate_jones)
        output = var_waveplate_jones(self.polarized_wavefront, delta, theta)
        chex.assert_shape(output.field, self.polarized_field.shape)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        e = jnp.exp(1j * delta)
        ex_in, ey_in = (
            self.polarized_field[..., 0],
            self.polarized_field[..., 1],
        )
        a = ct * ct + e * st * st
        b = (1.0 - e) * ct * st
        c = b
        d = st * st + e * ct * ct
        ex_expected = a * ex_in + b * ey_in
        ey_expected = c * ex_in + d * ey_in
        chex.assert_trees_all_close(
            output.field[..., 0], ex_expected, rtol=1e-5
        )
        chex.assert_trees_all_close(
            output.field[..., 1], ey_expected, rtol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("od_1", 1.0, -1.0),
        ("od_2", 2.0, -1.0),
        ("od_3", 3.0, -1.0),
        ("trans_50", 0.0, 0.5),
        ("trans_10", 0.0, 0.1),
    )
    def test_nd_filter(
        self, optical_density: float, transmittance: float
    ) -> None:
        """Test neutral density filter with OD and transmittance."""
        var_nd_filter = self.variant(nd_filter)
        output = var_nd_filter(self.wavefront, optical_density, transmittance)
        if optical_density != 0:
            expected_trans = 10 ** (-optical_density)
        else:
            expected_trans = jnp.clip(transmittance, 0.0, 1.0)
        expected_amp = jnp.sqrt(expected_trans)
        expected_field = self.field * expected_amp
        chex.assert_trees_all_close(output.field, expected_field, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_quarter_waveplate(self) -> None:
        """Test quarter-wave plate."""
        var_quarter_waveplate = self.variant(quarter_waveplate)
        theta = jnp.pi / 4
        output = var_quarter_waveplate(self.polarized_wavefront, theta)
        linear_x = jnp.array([1.0, 0.0])
        test_field = jnp.zeros((self.ny, self.nx, 2), dtype=complex)
        test_field = test_field + linear_x
        test_wf = make_optical_wavefront(
            field=test_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        output = var_quarter_waveplate(test_wf, theta)
        center_field = output.field[self.ny // 2, self.nx // 2]
        ex_out = center_field[0]
        ey_out = center_field[1]
        intensity_x = jnp.abs(ex_out) ** 2
        intensity_y = jnp.abs(ey_out) ** 2
        chex.assert_trees_all_close(intensity_x, intensity_y, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_half_waveplate(self) -> None:
        """Test half-wave plate."""
        var_half_waveplate = self.variant(half_waveplate)
        theta = jnp.pi / 4
        linear_x = jnp.array([1.0, 0.0])
        test_field = jnp.zeros((self.ny, self.nx, 2), dtype=complex)
        test_field = test_field + linear_x
        test_wf = make_optical_wavefront(
            field=test_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        output = var_half_waveplate(test_wf, theta)
        center_field = output.field[self.ny // 2, self.nx // 2]
        ex_out = jnp.abs(center_field[0])
        ey_out = jnp.abs(center_field[1])
        chex.assert_trees_all_close(ex_out, 0.0, atol=1e-5)
        chex.assert_trees_all_close(ey_out, 1.0, rtol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("1d_blaze", 10e-6, 20e-6, 0.0, 2 * jnp.pi, False),
        ("2d_blaze", 10e-6, 10e-6, 0.0, 2 * jnp.pi, True),
        ("rotated_1d", 10e-6, 20e-6, jnp.pi / 4, 2 * jnp.pi, False),
        ("rotated_2d", 10e-6, 15e-6, jnp.pi / 3, jnp.pi, True),
    )
    def test_phase_grating_blazed_elliptical(
        self,
        period_x: float,
        period_y: float,
        theta: float,
        depth: float,
        two_dim: bool,
    ) -> None:
        """Test elliptical blazed phase grating."""
        var_phase_grating_blazed_elliptical = self.variant(
            phase_grating_blazed_elliptical
        )
        output = var_phase_grating_blazed_elliptical(
            self.wavefront, period_x, period_y, theta, depth, two_dim
        )
        chex.assert_shape(output.field, self.field.shape)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        uu = ct * self.xx + st * self.yy
        vv = ct * self.yy - st * self.xx
        fu = (uu / period_x) - jnp.floor(uu / period_x)
        fv = (vv / period_y) - jnp.floor(vv / period_y)
        if two_dim:
            expected_phase = depth * ((fu + fv) - jnp.floor(fu + fv))
        else:
            expected_phase = depth * fu
        actual_phase = jnp.angle(output.field / self.field)
        wrapped_expected = jnp.angle(jnp.exp(1j * expected_phase))
        chex.assert_trees_all_close(
            actual_phase, wrapped_expected, rtol=1e-5, atol=1e-14
        )

    def test_vmap_on_elements(self) -> None:
        """Test vmap on optical elements."""
        deflections = jnp.array([0.001, 0.002, 0.003])
        vmapped_prism = jax.vmap(
            lambda d: prism_phase_ramp(self.wavefront, d, 0.0)
        )
        outputs = vmapped_prism(deflections)
        chex.assert_shape(outputs.field, (3, self.ny, self.nx))

    def test_energy_conservation_beam_splitter(self) -> None:
        """Test energy conservation in beam splitter."""
        wf_t, wf_r = beam_splitter(self.wavefront, 0.7, 0.3, normalize=True)
        power_in = jnp.sum(jnp.abs(self.field) ** 2)
        power_t = jnp.sum(jnp.abs(wf_t.field) ** 2)
        power_r = jnp.sum(jnp.abs(wf_r.field) ** 2)
        power_out = power_t + power_r
        chex.assert_trees_all_close(power_out, power_in, rtol=1e-5)

    @parameterized.named_parameters(
        ("zero_field", jnp.zeros((128, 128), dtype=complex)),
        ("complex_uniform", jnp.ones((128, 128)) * (1 + 1j)),
        ("pure_phase", jnp.exp(1j * jnp.ones((128, 128)))),
    )
    def test_elements_edge_cases(
        self, input_field: Complex[Array, "128 128"]
    ) -> None:
        """Test optical elements with edge case input fields."""
        test_wf = make_optical_wavefront(
            field=input_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        output_prism = prism_phase_ramp(test_wf, 0.001, 0.001)
        chex.assert_shape(output_prism.field, input_field.shape)
        chex.assert_trees_all_equal(jnp.iscomplexobj(output_prism.field), True)
        wf_t, wf_r = beam_splitter(test_wf, 0.5, 0.5)
        chex.assert_shape(wf_t.field, input_field.shape)
        chex.assert_shape(wf_r.field, input_field.shape)
        output_mirror = mirror_reflection(test_wf)
        chex.assert_shape(output_mirror.field, input_field.shape)

    def test_polarization_element_validation(self) -> None:
        """Test that polarization elements work only with polarized fields."""
        linear_pol = jnp.array([1.0, 0.0])
        pol_field = jnp.zeros((self.ny, self.nx, 2), dtype=complex)
        pol_field = pol_field + linear_pol
        pol_wf = make_optical_wavefront(
            field=pol_field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )
        output_pol = polarizer_jones(pol_wf, jnp.pi / 4)
        chex.assert_shape(output_pol.field, (self.ny, self.nx, 2))
        output_qwp = quarter_waveplate(pol_wf, jnp.pi / 4)
        chex.assert_shape(output_qwp.field, (self.ny, self.nx, 2))
        output_hwp = half_waveplate(pol_wf, jnp.pi / 4)
        chex.assert_shape(output_hwp.field, (self.ny, self.nx, 2))

    def test_phase_wrapping(self) -> None:
        """Test phase wrapping behavior in phase elements."""
        large_phase = 10 * jnp.pi * jnp.ones((self.ny, self.nx))
        output = apply_phase_mask(self.wavefront, large_phase)
        actual_phase = jnp.angle(output.field / self.field)
        chex.assert_trees_all_equal(
            jnp.all(jnp.abs(actual_phase) <= jnp.pi), True
        )

    def test_grating_periodicity(self) -> None:
        """Test that gratings have correct periodicity."""
        period = 10e-6
        depth = jnp.pi
        output = phase_grating_sine(self.wavefront, period, depth, 0.0)
        phase = jnp.angle(output.field / self.field)
        center_y = self.ny // 2
        x_slice = phase[center_y, :]
        dx_pixels = int(period / self.dx)
        if dx_pixels < self.nx // 2:
            val1 = x_slice[self.nx // 2]
            val2 = x_slice[self.nx // 2 + dx_pixels]
            chex.assert_trees_all_close(val1, val2, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
