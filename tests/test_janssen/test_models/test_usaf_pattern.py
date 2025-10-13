"""Tests for USAF 1951 resolution test pattern generation."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jaxtyping import Array, Float

from janssen.models.usaf_pattern import (
    create_bar_triplet,
    create_element,
    generate_usaf_pattern,
)


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
        width: float = 10.0
        length: float = 50.0
        spacing: float = 10.0

        pattern: Float[Array, " h w"] = var_create_bar_triplet(
            width, length, spacing, horizontal
        )

        chex.assert_rank(pattern, 2)
        chex.assert_type(pattern, float)

        # Check dimensions based on orientation
        bar_width: int = int(width)
        bar_length: int = int(length)
        gap: int = int(spacing)

        if horizontal:
            expected_height: int = 3 * bar_width + 2 * gap
            chex.assert_shape(pattern, (expected_height, bar_length))
        else:
            expected_width: int = 3 * bar_width + 2 * gap
            chex.assert_shape(pattern, (bar_length, expected_width))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_values(self) -> None:
        """Test bar triplet has correct binary values."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        width: float = 5.0
        length: float = 20.0
        spacing: float = 5.0

        pattern: Float[Array, " h w"] = var_create_bar_triplet(
            width, length, spacing, horizontal=True
        )

        # Pattern should only contain 0s and 1s
        unique_values = jnp.unique(pattern)
        chex.assert_trees_all_equal(
            jnp.sort(unique_values), jnp.array([0.0, 1.0])
        )

        # Check that bars are present (some 1s exist)
        chex.assert_scalar_positive(jnp.sum(pattern))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_minimum_width(self) -> None:
        """Test bar triplet with minimum width constraint."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        # Very small width should still produce valid pattern
        width: float = 0.5
        length: float = 10.0
        spacing: float = 0.5

        pattern: Float[Array, " h w"] = var_create_bar_triplet(
            width, length, spacing, horizontal=True
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
    def test_create_element_shape(
        self, group: int, element: int
    ) -> None:
        """Test element creation produces valid output shapes."""
        var_create_element = self.variant(create_element)

        elem: Float[Array, " h w"] = var_create_element(
            group, element, self.default_scale_factor, self.default_dpi
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

        elem: Float[Array, " h w"] = var_create_element(
            0, 1, self.default_scale_factor, self.default_dpi
        )

        # Element should only contain 0s and 1s
        unique_values = jnp.unique(elem)
        chex.assert_trees_all_equal(
            jnp.sort(unique_values), jnp.array([0.0, 1.0])
        )

        # Should have both bars (1s) and background (0s)
        chex.assert_scalar_positive(jnp.sum(elem))
        chex.assert_scalar_positive(jnp.sum(1.0 - elem))

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_element_scale_factor(self) -> None:
        """Test that scale factor affects element size."""
        var_create_element = self.variant(create_element)
        group: int = 0
        element: int = 1

        elem_1x: Float[Array, " h w"] = var_create_element(
            group, element, 1.0, self.default_dpi
        )
        elem_2x: Float[Array, " h w"] = var_create_element(
            group, element, 2.0, self.default_dpi
        )

        # Larger scale factor should produce larger elements
        size_1x: int = elem_1x.shape[0] * elem_1x.shape[1]
        size_2x: int = elem_2x.shape[0] * elem_2x.shape[1]

        # 2x scale should be roughly 4x larger (area scales quadratically)
        chex.assert_scalar_positive(size_2x - size_1x)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_element_dpi(self) -> None:
        """Test that DPI affects element size."""
        var_create_element = self.variant(create_element)
        group: int = 0
        element: int = 1

        elem_300dpi: Float[Array, " h w"] = var_create_element(
            group, element, self.default_scale_factor, 300.0
        )
        elem_600dpi: Float[Array, " h w"] = var_create_element(
            group, element, self.default_scale_factor, 600.0
        )

        # Higher DPI should produce larger elements
        size_300: int = elem_300dpi.shape[0] * elem_300dpi.shape[1]
        size_600: int = elem_600dpi.shape[0] * elem_600dpi.shape[1]

        chex.assert_scalar_positive(size_600 - size_300)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("size_256", 256),
        ("size_512", 512),
        ("size_1024", 1024),
        ("size_2048", 2048),
    )
    def test_generate_usaf_pattern_shape(self, image_size: int) -> None:
        """Test USAF pattern generation with various image sizes."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        pattern: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=image_size, groups=[-1, 0, 1], dpi=self.default_dpi
        )

        chex.assert_shape(pattern, (image_size, image_size))
        chex.assert_type(pattern, float)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_usaf_pattern_default_groups(self) -> None:
        """Test USAF pattern generation with default groups."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        pattern: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=512, groups=None, dpi=self.default_dpi
        )

        chex.assert_shape(pattern, (512, 512))
        chex.assert_type(pattern, float)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_usaf_pattern_value_range(self) -> None:
        """Test that generated pattern has valid value range."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)

        pattern: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=512, groups=range(0, 3), dpi=self.default_dpi
        )

        # Pattern should be in [0, 1] range
        # Background is 0.5, bars are 0 or 1
        chex.assert_trees_all_close(
            jnp.min(pattern), 0.0, atol=0.1, rtol=0.1
        )
        chex.assert_trees_all_close(
            jnp.max(pattern), 1.0, atol=0.1, rtol=0.1
        )

    @chex.variants(with_jit=True, without_jit=True)
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

        pattern: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=512, groups=groups, dpi=self.default_dpi
        )

        chex.assert_shape(pattern, (512, 512))
        # Pattern should have elements (not just background)
        # Check that some pixels are not the background value (0.5)
        non_background: Float[Array, " n"] = pattern[
            jnp.abs(pattern - 0.5) > self.BACKGROUND_TOLERANCE
        ]
        chex.assert_scalar_positive(len(non_background))

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_usaf_pattern_dpi_effect(self) -> None:
        """Test that DPI parameter affects the pattern."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)
        image_size: int = 512
        groups = [0, 1]

        pattern_300: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=300.0
        )
        pattern_600: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=600.0
        )

        # Patterns should be different due to different DPI
        difference: Float[Array, " h w"] = jnp.abs(pattern_300 - pattern_600)
        chex.assert_scalar_positive(jnp.sum(difference))

    def test_create_bar_triplet_jit_compatibility(self) -> None:
        """Test that create_bar_triplet is JIT-compatible."""
        jitted_fn = jax.jit(create_bar_triplet)

        pattern: Float[Array, " h w"] = jitted_fn(10.0, 50.0, 10.0, True)

        chex.assert_rank(pattern, 2)
        chex.assert_type(pattern, float)

    def test_create_element_jit_compatibility(self) -> None:
        """Test that create_element is JIT-compatible."""
        jitted_fn = jax.jit(create_element)

        elem: Float[Array, " h w"] = jitted_fn(
            0, 1, self.default_scale_factor, self.default_dpi
        )

        chex.assert_rank(elem, 2)
        chex.assert_type(elem, float)

    def test_generate_usaf_pattern_jit_compatibility(self) -> None:
        """Test that generate_usaf_pattern is JIT-compatible."""
        # Note: groups parameter cannot be None in JIT context
        # so we use a static list
        def _generate_with_static_groups(
            image_size: int, dpi: float
        ) -> Float[Array, " h w"]:
            return generate_usaf_pattern(
                image_size=image_size, groups=[0, 1, 2], dpi=dpi
            )

        jitted_fn = jax.jit(_generate_with_static_groups)

        pattern: Float[Array, " h w"] = jitted_fn(512, self.default_dpi)

        chex.assert_shape(pattern, (512, 512))
        chex.assert_type(pattern, float)

    @chex.variants(with_jit=True, without_jit=True)
    def test_generate_usaf_pattern_deterministic(self) -> None:
        """Test that pattern generation is deterministic."""
        var_generate_usaf_pattern = self.variant(generate_usaf_pattern)
        image_size: int = 256
        groups = [0, 1]
        dpi: float = 300.0

        pattern1: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=dpi
        )
        pattern2: Float[Array, " h w"] = var_generate_usaf_pattern(
            image_size=image_size, groups=groups, dpi=dpi
        )

        # Should produce identical results
        chex.assert_trees_all_equal(pattern1, pattern2)

    @chex.variants(with_jit=True, without_jit=True)
    def test_create_bar_triplet_horizontal_vs_vertical(self) -> None:
        """Test horizontal and vertical bar orientations are different."""
        var_create_bar_triplet = self.variant(create_bar_triplet)
        width: float = 10.0
        length: float = 50.0
        spacing: float = 10.0

        h_pattern: Float[Array, " h w"] = var_create_bar_triplet(
            width, length, spacing, horizontal=True
        )
        v_pattern: Float[Array, " h w"] = var_create_bar_triplet(
            width, length, spacing, horizontal=False
        )

        # Shapes should be transposed
        chex.assert_equal(h_pattern.shape[0], v_pattern.shape[1])
        chex.assert_equal(h_pattern.shape[1], v_pattern.shape[0])

    def test_create_element_increasing_resolution(self) -> None:
        """Test that elements get smaller with increasing element number."""
        # Within a group, higher element numbers have higher resolution
        # (smaller features)
        group: int = 0

        elem1: Float[Array, " h w"] = create_element(
            group, 1, self.default_scale_factor, self.default_dpi
        )
        elem6: Float[Array, " h w"] = create_element(
            group, 6, self.default_scale_factor, self.default_dpi
        )

        # Element 6 should have smaller features than element 1
        # This means element 6 should be smaller overall
        size1: int = elem1.shape[0] * elem1.shape[1]
        size6: int = elem6.shape[0] * elem6.shape[1]

        # Element 6 (higher resolution) should be smaller
        chex.assert_scalar_positive(size1 - size6)
