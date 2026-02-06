"""Tests for Gauss-Newton utilities."""

import chex
import jax
import jax.numpy as jnp

from janssen.types import make_gauss_newton_state
from janssen.utils import (
    compute_jt_residual,
    estimate_jtj_diagonal,
    estimate_max_eigenvalue,
    gauss_newton_solve,
    gauss_newton_step,
    make_hessian_matvec,
    make_jtj_matvec,
    unflatten_params,
)


class TestGaussNewtonStep(chex.TestCase):
    """Tests for gauss_newton_step."""

    def test_accepted_step_updates_state(self) -> None:
        """Accepted step should update params and reduce damping."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn, cg_maxiter=20)

        expected_params = target / (1.0 + state.damping)
        expected_sample, expected_probe = unflatten_params(
            expected_params, sample.shape, probe.shape
        )
        expected_loss = 0.5 * jnp.sum((expected_params - target) ** 2)

        chex.assert_trees_all_close(
            new_state.sample, expected_sample, rtol=1e-5
        )
        chex.assert_trees_all_close(new_state.probe, expected_probe, rtol=1e-5)
        chex.assert_trees_all_close(new_state.loss, expected_loss, rtol=1e-5)
        chex.assert_trees_all_equal(new_state.iteration, jnp.array(1))
        chex.assert_trees_all_equal(new_state.converged, jnp.array(False))
        assert new_state.damping < state.damping

    def test_rejected_step_increases_damping(self) -> None:
        """Rejecting a step should keep params and increase damping."""

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([params[0] ** 2 + 1.0])

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn)

        chex.assert_trees_all_close(new_state.sample, sample)
        chex.assert_trees_all_close(new_state.probe, probe)
        chex.assert_trees_all_close(new_state.loss, jnp.array(0.5))
        chex.assert_trees_all_equal(new_state.iteration, jnp.array(1))
        chex.assert_trees_all_equal(new_state.converged, jnp.array(False))
        chex.assert_trees_all_close(new_state.damping, state.damping * 10.0)

    def test_non_positive_predicted_reduction_increases_damping(self) -> None:
        """Non-positive predicted reduction should increase damping."""

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([1.0, -2.0])

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn)

        chex.assert_trees_all_close(new_state.sample, sample)
        chex.assert_trees_all_close(new_state.probe, probe)
        chex.assert_trees_all_close(new_state.damping, state.damping * 10.0)

    def test_converges_when_improvement_is_tiny(self) -> None:
        """Small relative improvement should mark convergence."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])
        epsilon = 1e-11
        params = target + epsilon

        def residual_fn(params_in: jnp.ndarray) -> jnp.ndarray:
            return params_in - target

        sample, probe = unflatten_params(params, (1, 1), (1, 1))
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn, cg_maxiter=20)

        chex.assert_trees_all_equal(new_state.converged, jnp.array(True))

    def test_preconditioning_converges(self) -> None:
        """Preconditioning should help convergence for ill-conditioned problems."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state_no_precond = gauss_newton_step(
            state, residual_fn, cg_maxiter=20, use_preconditioner=False
        )
        new_state_with_precond = gauss_newton_step(
            state, residual_fn, cg_maxiter=20, use_preconditioner=True
        )

        expected_params = target / (1.0 + state.damping)
        expected_sample, expected_probe = unflatten_params(
            expected_params, sample.shape, probe.shape
        )
        expected_loss = 0.5 * jnp.sum((expected_params - target) ** 2)

        chex.assert_trees_all_close(
            new_state_with_precond.sample, expected_sample, rtol=1e-5
        )
        chex.assert_trees_all_close(
            new_state_with_precond.probe, expected_probe, rtol=1e-5
        )
        chex.assert_trees_all_close(
            new_state_with_precond.loss, expected_loss, rtol=1e-5
        )
        chex.assert_trees_all_close(
            new_state_no_precond.loss, new_state_with_precond.loss, rtol=1e-4
        )

    def test_non_finite_residuals_mark_converged(self) -> None:
        """Non-finite residuals should set converged so the solver can stop."""

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([jnp.nan, jnp.nan])

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn, cg_maxiter=20)

        chex.assert_trees_all_equal(new_state.converged, jnp.array(True))
        chex.assert_trees_all_equal(new_state.iteration, jnp.array(1))

    def test_near_zero_loss_marks_converged(self) -> None:
        """Loss below LOSS_ZERO_TOL should mark converged without requiring a step."""

        target = jnp.array([1e-10, -1e-10, 1e-10, -1e-10])
        params = target.copy()

        def residual_fn(params_in: jnp.ndarray) -> jnp.ndarray:
            return params_in - target

        sample, probe = unflatten_params(params, (1, 1), (1, 1))
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        new_state = gauss_newton_step(state, residual_fn, cg_maxiter=20)

        chex.assert_trees_all_equal(new_state.converged, jnp.array(True))
        chex.assert_trees_all_equal(new_state.iteration, jnp.array(1))
        assert float(new_state.loss) < 1e-14


class TestMakeJtjMatvec(chex.TestCase):
    """Tests for make_jtj_matvec."""

    @chex.variants(without_jit=True)
    def test_matches_explicit_jtj(self) -> None:
        """Check (J^T J + lambda I) @ v against explicit computation."""

        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                [x[0] ** 2 + 2.0 * x[1], jnp.sin(x[0]) - x[1] ** 2]
            )

        params = jnp.array([0.1, -0.2])
        damping = jnp.array(0.5)
        v = jnp.array([1.0, -2.0])
        matvec = self.variant(make_jtj_matvec)(residual_fn, params, damping)

        j = jax.jacobian(residual_fn)(params)
        expected = j.T @ j @ v + damping * v
        actual = matvec(v)
        chex.assert_trees_all_close(actual, expected, rtol=1e-6)


class TestComputeJtResidual(chex.TestCase):
    """Tests for compute_jt_residual."""

    @chex.variants(without_jit=True)
    def test_gradient_matches_autodiff(self) -> None:
        """J^T r should match grad of 0.5 * ||r||^2."""

        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                [x[0] ** 2 + 2.0 * x[1], jnp.sin(x[0]) - x[1] ** 2]
            )

        params = jnp.array([0.4, -0.1])
        var_compute = self.variant(compute_jt_residual)
        residuals, jt_r = var_compute(residual_fn, params)

        loss_fn = lambda x: 0.5 * jnp.sum(residual_fn(x) ** 2)
        expected = jax.grad(loss_fn)(params)
        chex.assert_trees_all_close(jt_r, expected, rtol=1e-6)
        chex.assert_trees_all_close(residuals, residual_fn(params), rtol=1e-6)

    @chex.variants(without_jit=True)
    def test_weighted_gradient_matches_autodiff(self) -> None:
        """J^T W r should match grad of 0.5 * r^T W r."""

        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(
                [x[0] ** 2 + 2.0 * x[1], jnp.sin(x[0]) - x[1] ** 2]
            )

        params = jnp.array([0.4, -0.1])
        weights = jnp.array([2.0, 0.5])
        var_compute = self.variant(compute_jt_residual)
        residuals, jt_wr = var_compute(residual_fn, params, weights)

        weighted_loss_fn = lambda x: 0.5 * jnp.sum(
            weights * residual_fn(x) ** 2
        )
        expected = jax.grad(weighted_loss_fn)(params)
        chex.assert_trees_all_close(jt_wr, expected, rtol=1e-6)
        chex.assert_trees_all_close(residuals, residual_fn(params), rtol=1e-6)


class TestMakeHessianMatvec(chex.TestCase):
    """Tests for make_hessian_matvec."""

    @chex.variants(without_jit=True)
    def test_matches_explicit_hessian(self) -> None:
        """Exact Hessian-vector product matches explicit Hessian."""

        def loss_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(jnp.sin(x) ** 2 + 0.25 * x**4)

        params = jnp.array([0.5, -1.5])
        v = jnp.array([1.0, -2.0])
        hvp = self.variant(make_hessian_matvec)(loss_fn, params)

        h = jax.hessian(loss_fn)(params)
        expected = h @ v
        actual = hvp(v)
        chex.assert_trees_all_close(actual, expected, rtol=1e-6)


class TestEstimateMaxEigenvalue(chex.TestCase):
    """Tests for estimate_max_eigenvalue."""

    @chex.variants(without_jit=True)
    def test_known_largest_eigenvalue(self) -> None:
        """Largest eigenvalue for linear residual should be consistent."""

        a = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))

        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        params = jnp.array([0.1, -0.2, 0.3, -0.4])
        var_estimate = self.variant(estimate_max_eigenvalue)
        lambda_max = var_estimate(residual_fn, params, num_iterations=30)

        chex.assert_trees_all_close(lambda_max, jnp.array(16.0), rtol=1e-2)


class TestEstimateJtjDiagonal(chex.TestCase):
    """Tests for estimate_jtj_diagonal."""

    @chex.variants(without_jit=True)
    def test_diagonal_matches_linear_case(self) -> None:
        """Diagonal estimate should match exact diagonal for linear J."""

        a = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))

        def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        params = jnp.array([0.1, -0.2, 0.3, -0.4])
        var_estimate = self.variant(estimate_jtj_diagonal)
        diag = var_estimate(residual_fn, params, num_samples=300)

        chex.assert_trees_all_close(
            diag, jnp.array([1.0, 4.0, 9.0, 16.0]), rtol=0.1
        )


class TestGaussNewtonSolve(chex.TestCase):
    """Tests for gauss_newton_solve."""

    def test_converges_to_solution(self) -> None:
        """Solver should converge to optimal solution within max iterations."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        final_state = gauss_newton_solve(
            state, residual_fn, max_iterations=50, cg_maxiter=20
        )

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.loss < 1e-10
        assert final_state.iteration > 0

    def test_respects_max_iterations(self) -> None:
        """Solver should stop at max_iterations even if not converged."""

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([params[0] ** 2 + 1.0, params[1] ** 2 + 2.0])

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        max_iter = 5
        final_state = gauss_newton_solve(
            state, residual_fn, max_iterations=max_iter
        )

        assert final_state.iteration <= max_iter + 1

    def test_early_stop_on_near_zero_loss(self) -> None:
        """Solver should stop early when loss is near zero."""

        target = jnp.array([1e-15, -1e-15, 1e-15, -1e-15])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        final_state = gauss_newton_solve(
            state, residual_fn, max_iterations=50, cg_maxiter=20
        )

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.iteration <= 2

    def test_handles_nan_residuals(self) -> None:
        """Solver should mark converged when residuals become NaN."""

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([jnp.nan, jnp.nan])

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        final_state = gauss_newton_solve(state, residual_fn, max_iterations=10)

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.iteration <= 2

    def test_with_preconditioning(self) -> None:
        """Solver should work with preconditioning enabled."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        final_state = gauss_newton_solve(
            state,
            residual_fn,
            max_iterations=50,
            cg_maxiter=20,
            use_preconditioner=True,
        )

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.loss < 1e-10

    def test_jit_compatible(self) -> None:
        """Solver should be JIT-compilable."""

        target = jnp.array([1.0, -2.0, 0.5, 1.5])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        jitted_solve = jax.jit(
            gauss_newton_solve,
            static_argnums=(1, 2, 3, 4, 5),
        )

        final_state = jitted_solve(state, residual_fn, 50, 20, 1e-5, False)

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.loss < 1e-10

    def test_stops_on_convergence(self) -> None:
        """Solver should stop iterating once converged flag is True."""

        target = jnp.array([0.1, -0.1, 0.05, 0.05])

        def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
            return params - target

        sample = jnp.zeros((1, 1), dtype=jnp.complex128)
        probe = jnp.zeros((1, 1), dtype=jnp.complex128)
        state = make_gauss_newton_state(sample, probe, damping=1e-3)

        final_state = gauss_newton_solve(
            state, residual_fn, max_iterations=100, cg_maxiter=50
        )

        chex.assert_trees_all_equal(final_state.converged, jnp.array(True))
        assert final_state.iteration < 100
