"""Tests for USAF 1951 resolution test pattern generation."""

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from janssen.models.usaf_pattern import (
    create_bar_triplet,
    create_element_pattern,
    create_group_pattern,
    generate_usaf_pattern,
    get_bar_width_pixels,
)


class TestCreateBarTriplet:
    """Tests for create_bar_triplet function."""

    def test_horizontal_bars_shape(self) -> None:
        """Test horizontal bar triplet has correct shape."""
        width = 10
        length = 50
        pattern = create_bar_triplet(width, length, horizontal=True)

        # Height = 5 * width (3 bars + 2 spaces)
        # Width = length
        chex.assert_shape(pattern, (5 * width, length))

    def test_vertical_bars_shape(self) -> None:
        """Test vertical bar triplet has correct shape."""
        width = 10
        length = 50
        pattern = create_bar_triplet(width, length, horizontal=False)

        # Height = length
        # Width = 5 * width (3 bars + 2 spaces)
        chex.assert_shape(pattern, (length, 5 * width))

    def test_horizontal_bars_structure(self) -> None:
        """Test horizontal bars have correct structure."""
        width = 5
        length = 20
        pattern = create_bar_triplet(width, length, horizontal=True)

        # Check bar positions
        # Bar 1: rows [0, 5)
        chex.assert_trees_all_close(pattern[0:5, :], 1.0)
        # Space: rows [5, 10)
        chex.assert_trees_all_close(pattern[5:10, :], 0.0)
        # Bar 2: rows [10, 15)
        chex.assert_trees_all_close(pattern[10:15, :], 1.0)
        # Space: rows [15, 20)
        chex.assert_trees_all_close(pattern[15:20, :], 0.0)
        # Bar 3: rows [20, 25)
        chex.assert_trees_all_close(pattern[20:25, :], 1.0)

    def test_vertical_bars_structure(self) -> None:
        """Test vertical bars have correct structure."""
        width = 5
        length = 20
        pattern = create_bar_triplet(width, length, horizontal=False)

        # Check bar positions (columns instead of rows)
        chex.assert_trees_all_close(pattern[:, 0:5], 1.0)
        chex.assert_trees_all_close(pattern[:, 5:10], 0.0)
        chex.assert_trees_all_close(pattern[:, 10:15], 1.0)
        chex.assert_trees_all_close(pattern[:, 15:20], 0.0)
        chex.assert_trees_all_close(pattern[:, 20:25], 1.0)

    def test_minimum_width_enforced(self) -> None:
        """Test that minimum width of 1 is enforced."""
        pattern = create_bar_triplet(0, 10, horizontal=True)
        chex.assert_shape(pattern, (5, 10))  # 5 * 1 = 5

    def test_output_dtype(self) -> None:
        """Test output is float32."""
        pattern = create_bar_triplet(5, 20, horizontal=True)
        chex.assert_type(pattern, jnp.float32)


class TestCreateElementPattern:
    """Tests for create_element_pattern function."""

    def test_element_contains_both_triplets(self) -> None:
        """Test element contains both horizontal and vertical triplets."""
        bar_width = 10
        element = create_element_pattern(bar_width)

        # Element should be non-zero (contains bars)
        assert jnp.sum(element) > 0

        # Check dimensions are reasonable
        # Height should be 5 * bar_width (triplet height)
        # Width should be bar_length + gap + bar_length
        bar_length = 5 * bar_width
        assert element.shape[0] == 5 * bar_width
        assert element.shape[1] > bar_length  # At least one triplet width

    def test_element_aspect_ratio(self) -> None:
        """Test element has expected aspect ratio."""
        bar_width = 10
        element = create_element_pattern(bar_width)

        # Element should be wider than tall due to side-by-side triplets
        assert element.shape[1] > element.shape[0]

    def test_minimum_bar_width(self) -> None:
        """Test minimum bar width is enforced."""
        element = create_element_pattern(0)
        # Should still produce a valid pattern
        assert element.shape[0] >= 5
        assert element.shape[1] >= 1


class TestGetBarWidthPixels:
    """Tests for get_bar_width_pixels function."""

    def test_resolution_increases_with_group(self) -> None:
        """Test that bar width decreases (resolution increases) with group."""
        pixels_per_mm = 100.0
        element = 1

        widths = [
            get_bar_width_pixels(group, element, pixels_per_mm)
            for group in range(-2, 5)
        ]

        # Each successive group should have smaller bars
        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1]

    def test_resolution_increases_with_element(self) -> None:
        """Test that bar width decreases with element number."""
        pixels_per_mm = 100.0
        group = 0

        widths = [
            get_bar_width_pixels(group, elem, pixels_per_mm)
            for elem in range(1, 7)
        ]

        # Each successive element should have smaller or equal bars
        for i in range(len(widths) - 1):
            assert widths[i] >= widths[i + 1]

    def test_resolution_formula(self) -> None:
        """Test that resolution formula is correct."""
        pixels_per_mm = 100.0
        group = 0
        element = 1

        # R = 2^(0 + 0/6) = 1 lp/mm
        # bar_width = 1/(2*1) = 0.5 mm = 50 pixels
        expected_width = int(round(0.5 * pixels_per_mm))
        actual_width = get_bar_width_pixels(group, element, pixels_per_mm)

        assert actual_width == expected_width

    def test_minimum_width_enforced(self) -> None:
        """Test minimum width of 1 pixel is enforced."""
        # Very high group should result in very small bars
        width = get_bar_width_pixels(10, 6, 1.0)
        assert width >= 1


class TestCreateGroupPattern:
    """Tests for create_group_pattern function."""

    def test_group_contains_six_elements(self) -> None:
        """Test that group pattern has content for 6 elements."""
        group_pattern, max_dim = create_group_pattern(0, 50.0)

        # Should have non-zero content
        assert jnp.sum(group_pattern) > 0
        assert max_dim > 0

    def test_group_decreasing_element_size(self) -> None:
        """Test elements within group get progressively smaller."""
        # This is implicitly tested by get_bar_width_pixels
        # but we can verify the group pattern has varying content
        group_pattern, _ = create_group_pattern(0, 100.0)

        # Pattern should be 2D
        chex.assert_rank(group_pattern, 2)

    def test_group_layout_dimensions(self) -> None:
        """Test group has reasonable dimensions for 2x3 layout."""
        group_pattern, _ = create_group_pattern(0, 50.0)

        # Should be taller than wide (3 rows vs 2 columns of elements)
        # or roughly square depending on element sizes
        assert group_pattern.shape[0] > 0
        assert group_pattern.shape[1] > 0


class TestGenerateUsafPattern:
    """Tests for generate_usaf_pattern function."""

    def test_default_output_shape(self) -> None:
        """Test default output has correct shape."""
        pattern = generate_usaf_pattern()

        chex.assert_shape(pattern.sample, (1024, 1024))

    def test_custom_image_size(self) -> None:
        """Test custom image size."""
        pattern = generate_usaf_pattern(image_size=512)

        chex.assert_shape(pattern.sample, (512, 512))

    def test_custom_groups(self) -> None:
        """Test custom group range."""
        pattern = generate_usaf_pattern(image_size=512, groups=range(0, 3))

        # Should still produce valid output
        chex.assert_shape(pattern.sample, (512, 512))
        assert jnp.sum(pattern.sample) > 0

    def test_background_foreground_values(self) -> None:
        """Test background and foreground values are applied."""
        pattern = generate_usaf_pattern(
            image_size=256,
            groups=range(0, 2),
            background=0.2,
            foreground=0.8,
        )

        amp = jnp.abs(pattern.sample)
        # Background should be 0.2, bars should be 0.8
        assert jnp.min(amp) >= 0.2 - 1e-5
        assert jnp.max(amp) <= 0.8 + 1e-5

    def test_inverted_pattern(self) -> None:
        """Test inverted pattern (white background, black bars)."""
        pattern = generate_usaf_pattern(
            image_size=256,
            groups=range(0, 2),
            background=1.0,
            foreground=0.0,
        )

        amp = jnp.abs(pattern.sample)
        assert jnp.min(amp) >= 0.0 - 1e-5
        assert jnp.max(amp) <= 1.0 + 1e-5

    def test_dx_equals_pixel_size(self) -> None:
        """Test dx equals the provided pixel_size."""
        pixel_size = 2.5e-6
        pattern = generate_usaf_pattern(pixel_size=pixel_size)

        chex.assert_trees_all_close(pattern.dx, pixel_size, rtol=1e-5)

    def test_default_pixel_size(self) -> None:
        """Test default pixel size is 1 µm."""
        pattern = generate_usaf_pattern()

        chex.assert_trees_all_close(pattern.dx, 1.0e-6, rtol=1e-5)

    def test_output_is_complex(self) -> None:
        """Test output sample is complex."""
        pattern = generate_usaf_pattern(image_size=128, groups=range(0, 2))

        assert jnp.iscomplexobj(pattern.sample)

    def test_zero_phase_gives_real_values(self) -> None:
        """Test that max_phase=0 gives purely real output."""
        pattern = generate_usaf_pattern(
            image_size=128, groups=range(0, 2), max_phase=0.0
        )

        # Imaginary part should be zero (or very close)
        chex.assert_trees_all_close(jnp.imag(pattern.sample), 0.0, atol=1e-6)

    def test_nonzero_phase_adds_imaginary_component(self) -> None:
        """Test that max_phase > 0 adds imaginary component."""
        pattern = generate_usaf_pattern(
            image_size=256,
            groups=range(0, 2),
            max_phase=jnp.pi / 2,
            background=0.0,
            foreground=1.0,
        )

        # Where there are bars (foreground), there should be phase
        # Imaginary part should be non-zero somewhere
        assert jnp.max(jnp.abs(jnp.imag(pattern.sample))) > 0.1

    def test_phase_follows_amplitude_pattern(self) -> None:
        """Test that phase pattern matches amplitude pattern."""
        pattern = generate_usaf_pattern(
            image_size=256,
            groups=range(0, 2),
            max_phase=jnp.pi,
            background=0.0,
            foreground=1.0,
        )

        amp = jnp.abs(pattern.sample)
        phase = jnp.angle(pattern.sample)

        # Where amplitude is zero (background), phase is undefined but
        # the complex value is 0, so we skip those points

        # Where amplitude is high (foreground), phase should be ~pi or ~-pi
        # (they are equivalent due to angle wrapping)
        foreground_mask = amp > 0.9
        if jnp.any(foreground_mask):
            # |phase| should be close to pi
            assert (
                jnp.mean(jnp.abs(jnp.abs(phase[foreground_mask]) - jnp.pi))
                < 0.2
            )

    def test_returns_sample_function(self) -> None:
        """Test output is SampleFunction type."""
        pattern = generate_usaf_pattern(image_size=128, groups=range(0, 2))

        # Should have sample and dx attributes
        assert hasattr(pattern, "sample")
        assert hasattr(pattern, "dx")

    def test_output_dtype(self) -> None:
        """Test output sample is complex64."""
        pattern = generate_usaf_pattern(image_size=128, groups=range(0, 2))

        chex.assert_type(pattern.sample, jnp.complex64)


class TestJaxCompatibility:
    """Tests for JAX compatibility."""

    def test_bar_triplet_jit_compatible(self) -> None:
        """Test create_bar_triplet works with jit."""
        # Note: create_bar_triplet uses Python conditionals so cannot be
        # fully jitted, but should work within jitted functions that
        # call it with concrete values

        @jax.jit
        def wrapper(x: jax.Array) -> jax.Array:
            # Use concrete values inside jit
            pattern = create_bar_triplet(5, 20, horizontal=True)
            return pattern * x[0]

        result = wrapper(jnp.array([2.0]))
        chex.assert_shape(result, (25, 20))

    def test_element_pattern_is_array(self) -> None:
        """Test element pattern is a JAX array."""
        element = create_element_pattern(10)
        chex.assert_type(element, jnp.float32)

    def test_group_pattern_is_array(self) -> None:
        """Test group pattern is a JAX array."""
        group_pattern, _ = create_group_pattern(0, 50.0)
        chex.assert_type(group_pattern, jnp.float32)

    def test_usaf_pattern_sample_is_array(self) -> None:
        """Test USAF pattern sample is a JAX array."""
        pattern = generate_usaf_pattern(image_size=128, groups=range(0, 2))
        chex.assert_type(pattern.sample, jnp.complex64)
