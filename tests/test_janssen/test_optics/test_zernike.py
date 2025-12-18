import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from janssen.optics.zernike import (
    apply_aberration,
    astigmatism,
    coma,
    defocus,
    factorial,
    generate_aberration_nm,
    generate_aberration_noll,
    nm_to_noll,
    noll_to_nm,
    spherical_aberration,
    trefoil,
    zernike_polynomial,
    zernike_radial,
)
from janssen.utils import make_optical_wavefront


class TestZernike(chex.TestCase, parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.nx = 128
        self.ny = 128
        self.dx = 1e-6
        self.wavelength = 500e-9
        self.pupil_radius = 50e-6

        # Create coordinate grids
        x = jnp.arange(-self.nx // 2, self.nx // 2) * self.dx
        y = jnp.arange(-self.ny // 2, self.ny // 2) * self.dx
        self.xx, self.yy = jnp.meshgrid(x, y)
        self.rho = jnp.sqrt(self.xx**2 + self.yy**2) / self.pupil_radius
        self.theta = jnp.arctan2(self.yy, self.xx)

        # Create test wavefront
        self.field = jnp.ones((self.ny, self.nx), dtype=complex)
        self.wavefront = make_optical_wavefront(
            field=self.field,
            wavelength=self.wavelength,
            dx=self.dx,
            z_position=0.0,
        )

    def test_factorial(self) -> None:
        """Test factorial computation."""
        chex.assert_trees_all_equal(factorial(jnp.array(0)), 1)
        chex.assert_trees_all_equal(factorial(jnp.array(1)), 1)
        chex.assert_trees_all_equal(factorial(jnp.array(5)), 120)
        chex.assert_trees_all_equal(factorial(jnp.array(10)), 3628800)

    @parameterized.named_parameters(
        ("piston", 1, 0, 0),
        ("tilt_x", 2, 1, 1),
        ("tilt_y", 3, 1, -1),
        ("defocus", 4, 2, 0),
        ("astig_45", 5, 2, -2),
        ("astig_0", 6, 2, 2),
        ("coma_y", 7, 3, -1),
        ("coma_x", 8, 3, 1),
        ("trefoil_30", 9, 3, -3),
        ("trefoil_0", 10, 3, 3),
        ("spherical", 11, 4, 0),
    )
    def test_noll_to_nm(
        self, j: int, expected_n: int, expected_m: int
    ) -> None:
        """Test Noll index to (n, m) conversion."""
        n, m = noll_to_nm(j)
        chex.assert_trees_all_equal(n, expected_n)
        chex.assert_trees_all_equal(m, expected_m)

    @parameterized.named_parameters(
        ("piston", 0, 0, 1),
        ("tilt_x", 1, 1, 2),
        ("tilt_y", 1, -1, 3),
        ("defocus", 2, 0, 4),
        ("astig_45", 2, -2, 5),
        ("astig_0", 2, 2, 6),
        ("coma_y", 3, -1, 7),
        ("coma_x", 3, 1, 8),
        ("spherical", 4, 0, 11),
    )
    def test_nm_to_noll(self, n: int, m: int, expected_j: int) -> None:
        """Test (n, m) to Noll index conversion."""
        j = nm_to_noll(n, m)
        chex.assert_trees_all_equal(j, expected_j)

    def test_noll_roundtrip(self) -> None:
        """Test Noll index conversion roundtrip."""
        for j in range(1, 37):  # First 36 Zernike terms
            n, m = noll_to_nm(j)
            j_recovered = nm_to_noll(n, m)
            chex.assert_trees_all_equal(j_recovered, j)

    @parameterized.named_parameters(
        ("n2_m0", 2, 0),
        ("n2_m2", 2, 2),
        ("n3_m1", 3, 1),
        ("n4_m0", 4, 0),
        ("n4_m2", 4, 2),
        ("n4_m4", 4, 4),
    )
    def test_zernike_radial(self, n: int, m: int) -> None:
        """Test radial Zernike polynomial properties."""
        rho_test = jnp.linspace(0, 1, 100)
        R = zernike_radial(rho_test, n, m)

        # Check that R(0) has expected value
        R_0 = R[0]
        if m == 0:
            expected_R_0 = (-1) ** (n // 2) if n % 2 == 0 else 0
            chex.assert_trees_all_close(R_0, expected_R_0, atol=1e-10)
        else:
            chex.assert_trees_all_close(R_0, 0.0, atol=1e-10)

        # Check that R(1) = 1 for all valid (n, m)
        R_1 = R[-1]
        chex.assert_trees_all_close(R_1, 1.0, atol=1e-10)

        # Check that |R| <= 1 everywhere
        chex.assert_trees_all_equal(jnp.all(jnp.abs(R) <= 1.0001), True)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("defocus", 2, 0),
        ("astig_0", 2, 2),
        ("astig_45", 2, -2),
        ("coma_x", 3, 1),
        ("coma_y", 3, -1),
        ("spherical", 4, 0),
    )
    def test_zernike_polynomial(self, n: int, m: int) -> None:
        """Test full Zernike polynomial properties."""
        # Use partial to bind n, m, normalize before variant wrapping
        # since zernike_polynomial requires these as static Python values
        from functools import partial

        zp_bound = partial(zernike_polynomial, n=n, m=m, normalize=True)
        var_zernike_polynomial = self.variant(zp_bound)
        Z = var_zernike_polynomial(self.rho, self.theta)

        # Check that polynomial is zero outside unit circle
        outside_mask = self.rho > 1.0
        if jnp.any(outside_mask):
            chex.assert_trees_all_close(
                jnp.max(jnp.abs(Z[outside_mask])), 0.0, atol=1e-10
            )

        # Check shape
        chex.assert_shape(Z, self.rho.shape)

    def test_zernike_orthogonality(self) -> None:
        """Test orthogonality of Zernike polynomials over unit circle."""
        # Create finer grid for better integration
        # endpoint=False avoids double-counting at theta boundaries
        n_pts = 200
        r = jnp.linspace(0, 1, n_pts)
        t = jnp.linspace(0, 2 * jnp.pi, n_pts, endpoint=False)
        rr, tt = jnp.meshgrid(r, t)
        dr = r[1] - r[0]
        dt = t[1] - t[0]

        # Test orthogonality for a few low-order polynomials
        test_cases = [(2, 0), (2, 2), (3, 1), (3, -1)]

        for i, (n1, m1) in enumerate(test_cases):
            Z1 = zernike_polynomial(rr, tt, n1, m1, normalize=True)

            # Test orthogonality with itself: ∫∫ Z^2 r dr dθ = π (normalized)
            integral_self = jnp.sum(Z1 * Z1 * rr * dr * dt)
            chex.assert_trees_all_close(integral_self, jnp.pi, rtol=0.05)

            # Test orthogonality with others (should give 0)
            for n2, m2 in test_cases[i + 1 :]:
                Z2 = zernike_polynomial(rr, tt, n2, m2, normalize=True)
                integral_cross = jnp.sum(Z1 * Z2 * rr * dr * dt)
                chex.assert_trees_all_close(integral_cross, 0.0, atol=0.1)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_aberration_noll(self) -> None:
        """Test Noll-indexed aberration generation."""
        var_generate_aberration_noll = self.variant(generate_aberration_noll)

        # Test with defocus (Noll index 4) and spherical (Noll index 11)
        # Create coefficients array where index i corresponds to Noll index i+1
        coeffs = jnp.zeros(11)  # Support up to Noll index 11
        coeffs = coeffs.at[3].set(0.5)  # Noll 4 (defocus) = 0.5 waves
        coeffs = coeffs.at[10].set(0.1)  # Noll 11 (spherical) = 0.1 waves

        phase = var_generate_aberration_noll(
            self.xx, self.yy, coeffs, self.pupil_radius
        )

        # Check that phase is real
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)

        # Check shape
        chex.assert_shape(phase, self.xx.shape)

        # Check that phase is zero outside pupil
        outside_mask = self.rho > 1.0
        if jnp.any(outside_mask):
            chex.assert_trees_all_close(
                jnp.max(jnp.abs(phase[outside_mask])), 0.0, atol=1e-10
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_aberration_nm(self) -> None:
        """Test (n,m) indexed aberration generation."""
        var_generate_aberration_nm = self.variant(generate_aberration_nm)

        # Test with defocus (n=2, m=0) and astigmatism (n=2, m=2)
        n_indices = jnp.array([2, 2])
        m_indices = jnp.array([0, 2])
        coefficients = jnp.array(
            [0.5, 0.3]
        )  # 0.5 waves defocus, 0.3 waves astig

        phase = var_generate_aberration_nm(
            self.xx,
            self.yy,
            n_indices,
            m_indices,
            coefficients,
            self.pupil_radius,
        )

        # Check that phase is real
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)

        # Check shape
        chex.assert_shape(phase, self.xx.shape)

        # Check that phase is zero outside pupil
        outside_mask = self.rho > 1.0
        if jnp.any(outside_mask):
            chex.assert_trees_all_close(
                jnp.max(jnp.abs(phase[outside_mask])), 0.0, atol=1e-10
            )

    @chex.variants(with_jit=True, without_jit=True)
    def test_defocus(self) -> None:
        """Test defocus aberration generation."""
        var_defocus = self.variant(defocus)
        amplitude = 0.25  # waves
        phase = var_defocus(self.xx, self.yy, amplitude, self.pupil_radius)

        # Check basic properties
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)
        chex.assert_shape(phase, self.xx.shape)

        # Check that phase is zero outside pupil
        outside_mask = self.rho > 1.0
        if jnp.any(outside_mask):
            chex.assert_trees_all_close(
                jnp.max(jnp.abs(phase[outside_mask])), 0.0, atol=1e-10
            )

        # Check that defocus matches analytical formula: Z4 = sqrt(3)*(2*rho^2 - 1)
        # phase = 2*pi * amplitude * Z4
        inside_mask = self.rho <= 1.0
        z4_analytical = jnp.sqrt(3.0) * (2.0 * self.rho**2 - 1.0)
        expected_phase = 2.0 * jnp.pi * amplitude * z4_analytical
        chex.assert_trees_all_close(
            phase[inside_mask],
            expected_phase[inside_mask],
            rtol=1e-2,
            atol=1e-4,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_astigmatism(self) -> None:
        """Test astigmatism aberration generation."""
        var_astigmatism = self.variant(astigmatism)
        phase = var_astigmatism(self.xx, self.yy, 0.3, 0.2, self.pupil_radius)

        # Check that phase is real and has correct shape
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)
        chex.assert_shape(phase, self.xx.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_coma(self) -> None:
        """Test coma aberration generation."""
        var_coma = self.variant(coma)
        phase = var_coma(self.xx, self.yy, 0.15, 0.1, self.pupil_radius)

        # Check basic properties
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)
        chex.assert_shape(phase, self.xx.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_spherical_aberration(self) -> None:
        """Test spherical aberration generation."""
        var_spherical = self.variant(spherical_aberration)
        amplitude = 0.2
        phase = var_spherical(self.xx, self.yy, amplitude, self.pupil_radius)

        # Check basic properties
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)
        chex.assert_shape(phase, self.xx.shape)

        # Check that phase is zero outside pupil
        outside_mask = self.rho > 1.0
        if jnp.any(outside_mask):
            chex.assert_trees_all_close(
                jnp.max(jnp.abs(phase[outside_mask])), 0.0, atol=1e-10
            )

        # Check that spherical matches analytical formula:
        # Z11 = sqrt(5)*(6*rho^4 - 6*rho^2 + 1) [normalized]
        # phase = 2*pi * amplitude * Z11
        inside_mask = self.rho <= 1.0
        z11_analytical = jnp.sqrt(5.0) * (
            6.0 * self.rho**4 - 6.0 * self.rho**2 + 1.0
        )
        expected_phase = 2.0 * jnp.pi * amplitude * z11_analytical
        chex.assert_trees_all_close(
            phase[inside_mask], expected_phase[inside_mask], rtol=1e-2
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_trefoil(self) -> None:
        """Test trefoil aberration generation."""
        var_trefoil = self.variant(trefoil)
        phase = var_trefoil(self.xx, self.yy, 0.1, 0.15, self.pupil_radius)

        # Check basic properties
        chex.assert_trees_all_equal(jnp.isrealobj(phase), True)
        chex.assert_shape(phase, self.xx.shape)

    @chex.variants(with_jit=True, without_jit=True)
    def test_apply_aberration(self) -> None:
        """Test applying aberration to wavefront."""
        var_apply_aberration = self.variant(apply_aberration)

        # Defocus (Noll 4) and spherical (Noll 11)
        coeffs = jnp.zeros(11)
        coeffs = coeffs.at[3].set(0.25)  # Noll 4 (defocus) = 0.25 waves
        coeffs = coeffs.at[10].set(0.1)  # Noll 11 (spherical) = 0.1 waves

        output = var_apply_aberration(
            self.wavefront, coeffs, self.pupil_radius
        )

        # Check output properties
        chex.assert_shape(output.field, self.field.shape)
        chex.assert_trees_all_equal(output.wavelength, self.wavelength)
        chex.assert_trees_all_equal(output.dx, self.dx)

        # Check that phase was applied (field should be different)
        chex.assert_trees_all_equal(
            jnp.allclose(output.field, self.field), False
        )

    def test_jax_transformations(self) -> None:
        """Test JAX transformations on Zernike functions."""

        @jax.jit
        def jitted_defocus(xx, yy, amp, radius):
            return defocus(xx, yy, amp, radius)

        phase_jit = jitted_defocus(self.xx, self.yy, 0.5, self.pupil_radius)
        phase_normal = defocus(self.xx, self.yy, 0.5, self.pupil_radius)
        chex.assert_trees_all_close(phase_jit, phase_normal)

        # Test gradient computation
        def loss_fn(amplitude: float) -> float:
            phase = defocus(self.xx, self.yy, amplitude, self.pupil_radius)
            return jnp.sum(phase**2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(0.5)
        chex.assert_shape(grad, ())
        chex.assert_tree_all_finite(grad)

    def test_vmap_on_aberrations(self) -> None:
        """Test vmap on aberration functions."""
        amplitudes = jnp.array([0.1, 0.2, 0.3])

        vmapped_defocus = jax.vmap(
            lambda amp: defocus(self.xx, self.yy, amp, self.pupil_radius)
        )
        phases = vmapped_defocus(amplitudes)

        chex.assert_shape(phases, (3, self.ny, self.nx))

        # Check that each phase is different
        for i in range(3):
            for j in range(i + 1, 3):
                chex.assert_trees_all_equal(
                    jnp.allclose(phases[i], phases[j]), False
                )


if __name__ == "__main__":
    pytest.main([__file__])
