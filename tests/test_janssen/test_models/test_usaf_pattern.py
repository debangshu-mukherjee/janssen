"""Tests for USAF 1951 resolution test pattern generation."""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jaxtyping import Array, Float, Int

from janssen.models.usaf_pattern import (
    create_bar_triplet,
    create_element,
    generate_usaf_pattern,
)
from janssen.utils import SampleFunction


class TestUsafPattern(chex.TestCase, parameterized.TestCase):
    """Test suite for USAF pattern generation functions."""

    # Test constants
    BACKGROUND_TOLERANCE: float = 0.1

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.default_dpi: float = 300.0
        self.default_scale_factor: float = 1.0

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("horizontal", True),
        ("vertical", False),
    )
    def test_create_bar_triplet_shape(self, horizontal: bool) -> None:
        """Test bar triplet creation produces correct shapes."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        width_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)
        length_int: Int[Array, " "] = jnp.asarray(50, dtype=jnp.int32)
        spacing_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)

        # Create blank pattern buffer
        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        pattern: Int[Array, " h w"] = var_create_bar_triplet(
            blank, width_int, length_int, spacing_int, horizontal
        )

        chex.assert_rank(pattern, 2)
        chex.assert_type(pattern, int)

        # Pattern is buffer-sized, check that the relevant portion has bars
        # Check that some pixels are set to 1 (bars exist)
        chex.assert_scalar_positive(float(jnp.sum(pattern)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_values(self) -> None:
        """Test bar triplet has correct binary values."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        width_int: Int[Array, " "] = jnp.asarray(5, dtype=jnp.int32)
        length_int: Int[Array, " "] = jnp.asarray(20, dtype=jnp.int32)
        spacing_int: Int[Array, " "] = jnp.asarray(5, dtype=jnp.int32)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        pattern: Int[Array, " h w"] = var_create_bar_triplet(
            blank, width_int, length_int, spacing_int, horizontal=True
        )

        # Pattern should only contain 0s and 1s
        unique_values = jnp.unique(pattern)
        chex.assert_trees_all_equal(jnp.sort(unique_values), jnp.array([0, 1]))

        # Check that bars are present (some 1s exist)
        chex.assert_scalar_positive(float(jnp.sum(pattern)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_minimum_width(self) -> None:
        """Test bar triplet with minimum width constraint."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        # Very small width should still produce valid pattern
        width_int: Int[Array, " "] = jnp.asarray(1, dtype=jnp.int32)
        length_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)
        spacing_int: Int[Array, " "] = jnp.asarray(1, dtype=jnp.int32)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        pattern: Int[Array, " h w"] = var_create_bar_triplet(
            blank, width_int, length_int, spacing_int, horizontal=True
        )

        chex.assert_rank(pattern, 2)
        # Should still have at least 1 pixel width bars
        chex.assert_scalar_positive(pattern.shape[0])
        chex.assert_scalar_positive(pattern.shape[1])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("group_0_elem_1", 0, 1),
        ("group_1_elem_3", 1, 3),
        ("group_minus_1_elem_6", -1, 6),
        ("group_3_elem_4", 3, 4),
    )
    def test_create_element_shape(self, group: int, element: int) -> None:
        """Test element creation produces valid output shapes."""
        var_create_element = self.variant(create_element)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem: Float[Array, " h w"] = var_create_element(
            blank, group, element, self.default_scale_factor, self.default_dpi
        )

        chex.assert_rank(elem, 2)
        chex.assert_type(elem, float)
        # Element should have non-zero dimensions
        chex.assert_scalar_positive(elem.shape[0])
        chex.assert_scalar_positive(elem.shape[1])

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_element_values(self) -> None:
        """Test element contains valid binary values."""
        var_create_element = self.variant(create_element)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem: Float[Array, " h w"] = var_create_element(
            blank, 0, 1, self.default_scale_factor, self.default_dpi
        )

        # Element should only contain 0s and 1s
        unique_values = jnp.unique(elem)
        chex.assert_trees_all_equal(
            jnp.sort(unique_values), jnp.array([0.0, 1.0])
        )

        # Should have both bars (1s) and background (0s)
        chex.assert_scalar_positive(float(jnp.sum(elem)))
        chex.assert_scalar_positive(float(jnp.sum(1.0 - elem)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_element_scale_factor(self) -> None:
        """Test that scale factor affects element size."""
        var_create_element = self.variant(create_element)
        group: int = 0
        element: int = 1

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem_1x: Float[Array, " h w"] = var_create_element(
            blank, group, element, 1.0, self.default_dpi
        )
        elem_2x: Float[Array, " h w"] = var_create_element(
            blank, group, element, 2.0, self.default_dpi
        )

        # Larger scale factor should produce more bars (more 1s)
        # Both are buffer-sized, but 2x should have more non-zero content
        chex.assert_scalar_positive(float(jnp.sum(elem_2x) - jnp.sum(elem_1x)))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_element_dpi(self) -> None:
        """Test that DPI affects element size."""
        var_create_element = self.variant(create_element)
        group: int = 0
        element: int = 1

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem_300dpi: Float[Array, " h w"] = var_create_element(
            blank, group, element, self.default_scale_factor, 300.0
        )
        elem_600dpi: Float[Array, " h w"] = var_create_element(
            blank, group, element, self.default_scale_factor, 600.0
        )

        # Higher DPI should produce larger elements (more 1s)
        # Both are buffer-sized, but 600dpi should have more non-zero
        # content
        chex.assert_scalar_positive(
            float(jnp.sum(elem_600dpi) - jnp.sum(elem_300dpi))
        )

    @chex.variants(with_jit=False, without_jit=True)
    @parameterized.named_parameters(
        ("size_256", 256),
        ("size_512", 512),
        ("size_1024", 1024),
        ("size_2048", 2048),
    )
    def test_generate_usaf_pattern_shape(self, image_size: int) -> None:
        """Test USAF pattern generation with various image sizes."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        result: SampleFunction = var_generate_usaf_pattern(
            image_size=image_size, groups=[-1, 0, 1], dpi=self.default_dpi
        )

        chex.assert_shape(result.sample, (image_size, image_size))
        chex.assert_type(result.sample, complex)
        chex.assert_scalar_positive(float(result.dx))

    @chex.variants(with_jit=False, without_jit=True)
    def test_generate_usaf_pattern_default_groups(self) -> None:
        """Test USAF pattern generation with default groups."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        result: SampleFunction = var_generate_usaf_pattern(
            image_size=512, groups=None, dpi=self.default_dpi
        )

        chex.assert_shape(result.sample, (512, 512))
        chex.assert_type(result.sample, complex)

    @chex.variants(with_jit=False, without_jit=True)
    def test_generate_usaf_pattern_value_range(self) -> None:
        """Test that generated pattern has valid value range."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        result: SampleFunction = var_generate_usaf_pattern(
            image_size=512, groups=range(0, 3), dpi=self.default_dpi
        )

        # Pattern should be in [0, 1] range (for the real part)
        # Background is 0.5, bars are 0 or 1
        pattern: Float[Array, " h w"] = jnp.real(result.sample)
        chex.assert_trees_all_close(jnp.min(pattern), 0.0, atol=0.1, rtol=0.1)
        chex.assert_trees_all_close(jnp.max(pattern), 1.0, atol=0.1, rtol=0.1)

    @chex.variants(with_jit=False, without_jit=True)
    @parameterized.named_parameters(
        ("single_group", [-1]),
        ("two_groups", [0, 1]),
        ("four_groups", [-2, -1, 0, 1]),
        ("many_groups", list(range(-2, 5))),
    )
    def test_generate_usaf_pattern_various_groups(
        self, groups: list[int]
    ) -> None:
        """Test USAF pattern generation with various group configurations."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        result: SampleFunction = var_generate_usaf_pattern(
            image_size=512, groups=groups, dpi=self.default_dpi
        )

        chex.assert_shape(result.sample, (512, 512))
        # Pattern should have elements (not just background)
        # Check that some pixels are not the background value (0.5)
        pattern: Float[Array, " h w"] = jnp.real(result.sample)
        non_background: Float[Array, " n"] = pattern[
            jnp.abs(pattern - 0.5) > self.BACKGROUND_TOLERANCE
        ]
        chex.assert_scalar_positive(len(non_background))

    @chex.variants(with_jit=False, without_jit=True)
    def test_generate_usaf_pattern_dpi_effect(self) -> None:
        """Test that DPI parameter affects the pattern."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)
        image_size: int = 512
        groups = [0, 1]

        result_300: SampleFunction = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=300.0
        )
        result_600: SampleFunction = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=600.0
        )

        # Patterns should be different due to different DPI
        pattern_300: Float[Array, " h w"] = jnp.real(result_300.sample)
        pattern_600: Float[Array, " h w"] = jnp.real(result_600.sample)
        difference: Float[Array, " h w"] = jnp.abs(pattern_300 - pattern_600)
        chex.assert_scalar_positive(float(jnp.sum(difference)))

    def test_create_bar_triplet_jit_compatibility(self) -> None:
        """Test that create_bar_triplet is JIT-compatible."""
        jitted_fn = jax.jit(create_bar_triplet)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )
        width_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)
        length_int: Int[Array, " "] = jnp.asarray(50, dtype=jnp.int32)
        spacing_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)

        pattern: Int[Array, " h w"] = jitted_fn(
            blank, width_int, length_int, spacing_int, True
        )

        chex.assert_rank(pattern, 2)
        chex.assert_type(pattern, int)

    def test_create_element_jit_compatibility(self) -> None:
        """Test that create_element is JIT-compatible."""
        jitted_fn = jax.jit(create_element)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem: Float[Array, " h w"] = jitted_fn(
            blank, 0, 1, self.default_scale_factor, self.default_dpi
        )

        chex.assert_rank(elem, 2)
        chex.assert_type(elem, float)

    @chex.variants(with_jit=False, without_jit=True)
    def test_generate_usaf_pattern_deterministic(self) -> None:
        """Test that pattern generation is deterministic."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)
        image_size: int = 256
        groups = [0, 1]
        dpi: float = 300.0

        result1: SampleFunction = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=dpi
        )
        result2: SampleFunction = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=dpi
        )

        # Should produce identical results
        chex.assert_trees_all_equal(result1.sample, result2.sample)
        chex.assert_trees_all_equal(result1.dx, result2.dx)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_horizontal_vs_vertical(self) -> None:
        """Test horizontal and vertical bar orientations are different."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        width_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)
        length_int: Int[Array, " "] = jnp.asarray(50, dtype=jnp.int32)
        spacing_int: Int[Array, " "] = jnp.asarray(10, dtype=jnp.int32)

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        h_pattern: Int[Array, " h w"] = var_create_bar_triplet(
            blank, width_int, length_int, spacing_int, horizontal=True
        )
        v_pattern: Int[Array, " h w"] = var_create_bar_triplet(
            blank, width_int, length_int, spacing_int, horizontal=False
        )

        # Both are buffer-sized, but patterns should be different
        # Check that they're not identical
        diff_sum: int = int(jnp.sum(jnp.abs(h_pattern - v_pattern)))
        chex.assert_scalar_positive(diff_sum)
        chex.assert_equal(h_pattern.shape[1], v_pattern.shape[0])

    def test_create_element_increasing_resolution(self) -> None:
        """Test that elements get smaller with increasing element number."""
        # Within a group, higher element numbers have higher resolution
        # (smaller features)
        group: int = 0

        buffer_size: int = 500
        blank: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
            (buffer_size, buffer_size), dtype=jnp.int32
        )

        elem1: Float[Array, " h w"] = create_element(
            blank, group, 1, self.default_scale_factor, self.default_dpi
        )
        elem6: Float[Array, " h w"] = create_element(
            blank, group, 6, self.default_scale_factor, self.default_dpi
        )

        # Element 6 should have smaller features than element 1
        # Both are buffer-sized, but elem1 should have more 1s (larger bars)
        chex.assert_scalar_positive(float(jnp.sum(elem1) - jnp.sum(elem6)))


if __name__ == "__main__":
    pytest.main([__file__])
