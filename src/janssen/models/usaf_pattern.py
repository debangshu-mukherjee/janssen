"""USAF 1951 resolution test pattern generation.

Extended Summary
----------------
Generates USAF 1951 resolution test patterns using pure JAX operations.
Supports configurable image sizes, group ranges, and DPI settings for
optical resolution testing.

Routine Listings
----------------
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element : function
    Creates a single element (horizontal + vertical bars)
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern

Notes
-----
All functions use JAX operations and support automatic differentiation.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Iterable, Optional
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from janssen.utils import scalar_bool, scalar_float, scalar_integer


@jaxtyped(typechecker=beartype)
def create_bar_triplet(
    blank_pattern: Int[Array, " h w"],
    width: scalar_integer,
    length: scalar_integer,
    spacing: scalar_integer,
    horizontal: scalar_bool = True,
) -> Int[Array, " h w"]:
    """Create 3 parallel bars (horizontal or vertical) using pre-allocated array.

    Parameters
    ----------
    blank_pattern : Int[Array, " h w"]
        Pre-allocated array to fill with bar pattern
    width : scalar_integer
        Width of the bars
    length : scalar_integer
        Length of the bars
    spacing : scalar_integer
        Spacing between the bars
    horizontal : bool, optional
        Whether to create horizontal bars, by default True

    Returns
    -------
    pattern : Int[Array, " h w"]
        The created pattern (slice of blank_pattern)

    Notes
    -----
    Algorithm:
    - Use pre-allocated blank_pattern to avoid traced shape creation
    - Set the pixels in the array to 1 for the bars
    - Extract and return the relevant slice
    - If horizontal is True, create 3 horizontal bars
    - If horizontal is False, create 3 vertical bars
    - Return the created pattern
    """

    def _horizontal() -> Int[Array, " h w"]:
        """Create 3 horizontal bars using coordinate masks."""
        # Create coordinate grids
        buffer_h: int = blank_pattern.shape[0]
        buffer_w: int = blank_pattern.shape[1]
        y_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_h, dtype=jnp.int32
        )[:, None]
        x_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_w, dtype=jnp.int32
        )[None, :]

        # Create masks for each bar
        # Bar 1: rows [0, width), cols [0, length)
        bar1_mask: Bool[Array, " buffer_h buffer_w"] = (y_coords < width) & (
            x_coords < length
        )
        # Bar 2: rows [width+spacing, 2*width+spacing), cols [0, length)
        bar2_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords >= (width + spacing))
            & (y_coords < ((2 * width) + spacing))
            & (x_coords < length)
        )
        # Bar 3: rows [2*width+2*spacing, 3*width+2*spacing), cols [0, length)
        bar3_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords >= ((2 * width) + (2 * spacing)))
            & (y_coords < ((3 * width) + (2 * spacing)))
            & (x_coords < length)
        )

        # Combine masks and create pattern
        all_bars_mask: Bool[Array, " buffer_h buffer_w"] = (
            bar1_mask | bar2_mask | bar3_mask
        )
        pattern_out: Int[Array, " h w"] = all_bars_mask.astype(jnp.int32)
        return pattern_out

    def _vertical() -> Int[Array, " h w"]:
        """Create 3 vertical bars using coordinate masks."""
        # Create coordinate grids
        buffer_h: int = blank_pattern.shape[0]
        buffer_w: int = blank_pattern.shape[1]
        y_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_h, dtype=jnp.int32
        )[:, None]
        x_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_w, dtype=jnp.int32
        )[None, :]

        # Create masks for each bar
        # Bar 1: rows [0, length), cols [0, width)
        bar1_mask: Bool[Array, " buffer_h buffer_w"] = (y_coords < length) & (
            x_coords < width
        )
        # Bar 2: rows [0, length), cols [width+spacing, 2*width+spacing)
        bar2_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords < length)
            & (x_coords >= (width + spacing))
            & (x_coords < ((2 * width) + spacing))
        )
        # Bar 3: rows [0, length), cols [2*width+2*spacing, 3*width+2*spacing)
        bar3_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords < length)
            & (x_coords >= ((2 * width) + (2 * spacing)))
            & (x_coords < ((3 * width) + (2 * spacing)))
        )

        # Combine masks and create pattern
        all_bars_mask: Bool[Array, " buffer_h buffer_w"] = (
            bar1_mask | bar2_mask | bar3_mask
        )
        pattern_out: Int[Array, " h w"] = all_bars_mask.astype(jnp.int32)
        return pattern_out

    result: Int[Array, " h w"] = jax.lax.cond(
        horizontal, _horizontal, _vertical
    )

    return result


@jaxtyped(typechecker=beartype)
def create_element(
    blank_pattern: Int[Array, " buffer_size buffer_size"],
    group: scalar_integer,
    element: scalar_integer,
    scale_factor: scalar_float,
    dpi: scalar_float,
) -> Float[Array, " buffer_size buffer_size"]:
    """Create a single element (horizontal + vertical bars).

    Parameters
    ----------
    blank_pattern : Int[Array, " buffer_size buffer_size"]
        Pre-allocated buffer array to avoid traced shape creation
    group : scalar_integer
        Group number (can be Python int or JAX scalar)
    element : scalar_integer
        Element number (can be Python int or JAX scalar)
    scale_factor : scalar_float
        Scale factor
    dpi : scalar_float
        Dots per inch

    Returns
    -------
    element_img : Float[Array, " h w"]
        The created element

    Notes
    -----
    Algorithm:
    - Calculate the resolution in line pairs per millimeter
    - Calculate the bar width in pixels
    - Calculate the bar length in pixels
    - Create the horizontal bars using pre-allocated buffer
    - Create the vertical bars using pre-allocated buffer
    - Create the element image
    - Return the element image
    """
    # Convert to JAX scalars if needed
    group_val: Float[Array, " "] = jnp.asarray(group, dtype=jnp.float64)
    element_val: Float[Array, " "] = jnp.asarray(element, dtype=jnp.float64)

    resolution_lp_mm: scalar_float = 2.0 ** (
        group_val + (element_val - 1) / 6.0
    )
    mm_per_inch: float = 25.4
    pixels_per_mm: scalar_float = dpi / mm_per_inch
    pixels_per_lp: scalar_float = pixels_per_mm / resolution_lp_mm
    bar_width: Float[Array, " "] = pixels_per_lp / 2.0 * scale_factor
    bar_width = jnp.maximum(bar_width, 1.0)
    bar_length: Float[Array, " "] = bar_width * 5.0

    # Convert to integers for create_bar_triplet
    bar_width_int: scalar_integer = jnp.round(bar_width).astype(jnp.int32)
    bar_length_int: scalar_integer = jnp.round(bar_length).astype(jnp.int32)
    spacing_int: scalar_integer = jnp.round(bar_width * 0.5).astype(jnp.int32)

    # Use the pre-allocated blank_pattern buffer passed from
    # generate_usaf_pattern
    h_bars: Int[Array, " h w"] = create_bar_triplet(
        blank_pattern,
        bar_width_int,
        bar_length_int,
        bar_width_int,
        horizontal=True,
    )
    v_bars: Int[Array, " h w"] = create_bar_triplet(
        blank_pattern,
        bar_width_int,
        bar_length_int,
        bar_width_int,
        horizontal=False,
    )
    # Get shapes as JAX scalars
    h_height: scalar_integer = h_bars.shape[0]
    h_width: scalar_integer = h_bars.shape[1]
    v_height: scalar_integer = v_bars.shape[0]
    v_width: scalar_integer = v_bars.shape[1]

    # Calculate total size using JAX operations
    total_height: scalar_integer = jnp.maximum(h_height, v_height)

    # Use the buffer for element assembly
    element_buffer: Float[Array, " buffer_size buffer_size"] = (
        blank_pattern.astype(jnp.float32) * 0.0
    )

    # Calculate offsets as int32
    h_offset: scalar_integer = ((total_height - h_height) // 2).astype(
        jnp.int32
    )
    v_offset: scalar_integer = ((total_height - v_height) // 2).astype(
        jnp.int32
    )
    zero_int32: scalar_integer = jnp.int32(0)

    # Place horizontal bars using dynamic_update_slice
    h_bars_float: Float[Array, " h_height h_width"] = h_bars.astype(
        jnp.float32
    )
    element_buffer = jax.lax.dynamic_update_slice(
        element_buffer, h_bars_float, (h_offset, zero_int32)
    )

    # Place vertical bars using dynamic_update_slice
    v_bars_float: Float[Array, " v_height v_width"] = v_bars.astype(
        jnp.float32
    )
    v_x_pos: scalar_integer = (h_width + spacing_int).astype(jnp.int32)
    element_buffer_out: Float[Array, " buffer_size buffer_size"] = (
        jax.lax.dynamic_update_slice(
            element_buffer, v_bars_float, (v_offset, v_x_pos)
        )
    )

    # Return the buffer - the caller will handle extracting
    # the relevant portion. We cannot use dynamic_slice here because
    # total_height and total_width are traced values
    return element_buffer_out


@jaxtyped(typechecker=beartype)
def generate_usaf_pattern(
    image_size: int = 1024,
    groups: Optional[Iterable[int]] = None,
    dpi: scalar_float = 300,
) -> Float[Array, " image_size image_size"]:
    """
    Generate USAF 1951 resolution test pattern using pure JAX.

    Parameters
    ----------
    image_size : int, optional
        Size of the output image (square), by default 1024
    groups : Iterable[int], optional
        Range of groups to include, by default range(-2, 8)
    dpi : scalar_float, optional
        Dots per inch for scaling, by default 300

    Returns
    -------
    pattern : Float[Array, " image_size image_size"]
        JAX array containing the USAF test pattern
    """
    # Handle None at Python time, not JAX trace time
    groups_to_use: Iterable[int] = (
        groups if groups is not None else range(-2, 8)
    )
    groups_array: Float[Array, " n"] = jnp.array(
        list(groups_to_use), dtype=jnp.int32
    )

    canvas: Float[Array, " image_size image_size"] = (
        jnp.ones((image_size, image_size), dtype=jnp.float32) * 0.5
    )
    scale_factor: scalar_float = image_size / 1024.0
    num_groups: int = len(groups_array)
    grid_size: int = int(jnp.ceil(jnp.sqrt(num_groups)))
    cell_size: int = image_size // (grid_size + 1)
    element_spacing: int = int(5 * scale_factor)

    # Create a fixed-size buffer for bar pattern creation
    # Use half the image size as buffer, large enough for elements
    buffer_size: int = max(500, image_size // 2)
    blank_pattern: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
        (buffer_size, buffer_size), dtype=jnp.int32
    )

    def _process_all_groups(
        group_idx: int, canvas_carry: Float[Array, " image_size image_size"]
    ) -> Float[Array, " image_size image_size"]:
        """Process a single group and place all its elements."""
        # Index into groups array - result is a JAX scalar integer
        group: scalar_integer = groups_array[group_idx]

        # Calculate group position in grid (keep as JAX arrays)
        row: Float[Array, " "] = jnp.asarray(group_idx) // grid_size
        col: Float[Array, " "] = jnp.asarray(group_idx) % grid_size
        y_pos: Float[Array, " "] = (row + 0.5) * cell_size
        x_pos: Float[Array, " "] = (col + 0.5) * cell_size

        # Process all 6 elements in this group
        def _process_all_elements(
            elem_idx: int,
            canvas_elem_carry: Float[Array, " image_size image_size"],
        ) -> Float[Array, " image_size image_size"]:
            """Place a single element on the canvas."""
            element: int = elem_idx + 1  # Elements are 1-6

            # Create element
            elem: Float[Array, " h w"] = create_element(
                blank_pattern, group, element, scale_factor, dpi
            )

            # Calculate element position within group
            e_row: Float[Array, " "] = jnp.asarray(elem_idx) // 3
            e_col: Float[Array, " "] = jnp.asarray(elem_idx) % 3

            # elem is buffer_size x buffer_size, we need to place it
            # Calculate position on canvas (as JAX arrays)
            elem_y_float: Float[Array, " "] = y_pos + e_row * element_spacing
            elem_x_float: Float[Array, " "] = x_pos + e_col * element_spacing

            # Clip to bounds and convert to int32 for dynamic_update_slice
            elem_y: scalar_integer = jnp.clip(
                elem_y_float, 0, image_size - buffer_size
            ).astype(jnp.int32)
            elem_x: scalar_integer = jnp.clip(
                elem_x_float, 0, image_size - buffer_size
            ).astype(jnp.int32)

            # Place element on canvas using dynamic_update_slice
            # Since elem is buffer-sized, we directly update
            canvas_updated: Float[Array, " image_size image_size"] = (
                jax.lax.dynamic_update_slice(
                    canvas_elem_carry, elem, (elem_y, elem_x)
                )
            )

            return canvas_updated

        # Loop over all 6 elements
        canvas_with_elements: Float[Array, " image_size image_size"] = (
            jax.lax.fori_loop(0, 6, _process_all_elements, canvas_carry)
        )

        return canvas_with_elements

    # Loop over all groups
    final_canvas: Float[Array, " image_size image_size"] = jax.lax.fori_loop(
        0, num_groups, _process_all_groups, canvas
    )

    return final_canvas
