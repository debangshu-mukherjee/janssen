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
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    jaxtyped,
)

from janssen.utils import (
    SampleFunction,
    make_sample_function,
    scalar_bool,
    scalar_float,
    scalar_integer,
)


@jax.jit
@jaxtyped(typechecker=beartype)
def create_bar_triplet(
    blank_pattern: Int[Array, " h w"],
    width: scalar_integer,
    length: scalar_integer,
    spacing: scalar_integer,
    horizontal: scalar_bool = True,
) -> Int[Array, " h w"]:
    """Create 3 parallel bars (horizontal or vertical).

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

    Implementation Details:
    - Creates coordinate grids for buffer height and width
    - For horizontal bars: creates 3 masks at rows [0, width),
      [width+spacing, 2*width+spacing), [2*width+2*spacing, 3*width+2*spacing)
      all spanning columns [0, length)
    - For vertical bars: creates 3 masks at columns [0, width),
      [width+spacing, 2*width+spacing), [2*width+2*spacing, 3*width+2*spacing)
      all spanning rows [0, length)
    - Combines all masks using logical OR and converts to int32
    """

    def _horizontal() -> Int[Array, " h w"]:
        """Create 3 horizontal bars using coordinate masks."""
        buffer_h: int = blank_pattern.shape[0]
        buffer_w: int = blank_pattern.shape[1]
        y_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_h, dtype=jnp.int32
        )[:, None]
        x_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_w, dtype=jnp.int32
        )[None, :]

        bar1_mask: Bool[Array, " buffer_h buffer_w"] = (y_coords < width) & (
            x_coords < length
        )
        bar2_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords >= (width + spacing))
            & (y_coords < ((2 * width) + spacing))
            & (x_coords < length)
        )
        bar3_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords >= ((2 * width) + (2 * spacing)))
            & (y_coords < ((3 * width) + (2 * spacing)))
            & (x_coords < length)
        )

        all_bars_mask: Bool[Array, " buffer_h buffer_w"] = (
            bar1_mask | bar2_mask | bar3_mask
        )
        pattern_out: Int[Array, " h w"] = all_bars_mask.astype(jnp.int32)
        return pattern_out

    def _vertical() -> Int[Array, " h w"]:
        """Create 3 vertical bars using coordinate masks."""
        buffer_h: int = blank_pattern.shape[0]
        buffer_w: int = blank_pattern.shape[1]
        y_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_h, dtype=jnp.int32
        )[:, None]
        x_coords: Int[Array, " buffer_h buffer_w"] = jnp.arange(
            buffer_w, dtype=jnp.int32
        )[None, :]

        bar1_mask: Bool[Array, " buffer_h buffer_w"] = (y_coords < length) & (
            x_coords < width
        )
        bar2_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords < length)
            & (x_coords >= (width + spacing))
            & (x_coords < ((2 * width) + spacing))
        )
        bar3_mask: Bool[Array, " buffer_h buffer_w"] = (
            (y_coords < length)
            & (x_coords >= ((2 * width) + (2 * spacing)))
            & (x_coords < ((3 * width) + (2 * spacing)))
        )

        all_bars_mask: Bool[Array, " buffer_h buffer_w"] = (
            bar1_mask | bar2_mask | bar3_mask
        )
        pattern_out: Int[Array, " h w"] = all_bars_mask.astype(jnp.int32)
        return pattern_out

    result: Int[Array, " h w"] = jax.lax.cond(
        horizontal, _horizontal, _vertical
    )

    return result


@jax.jit
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

    Implementation Details:
    - Converts group and element to JAX float64 scalars for computation
    - Resolution formula: 2^(group + (element-1)/6) line pairs per mm
    - Bar width = (pixels_per_mm / resolution) / 2 * scale_factor, min 1.0
    - Bar length = bar_width * 5.0
    - Spacing = bar_width * 0.5
    - Creates horizontal and vertical bar triplets using pre-allocated buffer
    - Assembles element by placing bars in buffer with calculated offsets
    - Centers bars vertically using (total_height - bar_height) // 2
    - Horizontal bars placed at x=0, vertical bars at x=h_width+spacing
    - Returns full buffer; caller extracts relevant portion
    - Cannot use dynamic_slice as dimensions are traced values
    """
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

    bar_width_int: scalar_integer = jnp.round(bar_width).astype(jnp.int32)
    bar_length_int: scalar_integer = jnp.round(bar_length).astype(jnp.int32)
    spacing_int: scalar_integer = jnp.round(bar_width * 0.5).astype(jnp.int32)

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
    h_height: scalar_integer = h_bars.shape[0]
    h_width: scalar_integer = h_bars.shape[1]
    v_height: scalar_integer = v_bars.shape[0]

    total_height: scalar_integer = jnp.maximum(h_height, v_height)

    element_buffer: Float[Array, " buffer_size buffer_size"] = (
        blank_pattern.astype(jnp.float32) * 0.0
    )

    h_offset: scalar_integer = ((total_height - h_height) // 2).astype(
        jnp.int32
    )
    v_offset: scalar_integer = ((total_height - v_height) // 2).astype(
        jnp.int32
    )
    zero_int32: scalar_integer = jnp.int32(0)

    h_bars_float: Float[Array, " h_height h_width"] = h_bars.astype(
        jnp.float32
    )
    element_buffer = jax.lax.dynamic_update_slice(
        element_buffer, h_bars_float, (h_offset, zero_int32)
    )

    v_bars_float: Float[Array, " v_height v_width"] = v_bars.astype(
        jnp.float32
    )
    v_x_pos: scalar_integer = (h_width + spacing_int).astype(jnp.int32)
    element_buffer_out: Float[Array, " buffer_size buffer_size"] = (
        jax.lax.dynamic_update_slice(
            element_buffer, v_bars_float, (v_offset, v_x_pos)
        )
    )

    return element_buffer_out


@jaxtyped(typechecker=beartype)
def generate_usaf_pattern(
    image_size: int = 1024,
    groups: Optional[Iterable[int]] = None,
    dpi: scalar_float = 300,
    dx: Optional[scalar_float] = None,
) -> SampleFunction:
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
    dx : scalar_float, optional
        Spatial sampling interval in meters. If None, calculated from dpi.

    Returns
    -------
    pattern : SampleFunction
        SampleFunction PyTree containing the USAF test pattern

    Notes
    -----
    Algorithm:
    - Calculate spatial sampling interval dx from dpi if not provided
    - Initialize canvas to 0.5 (gray background)
    - Calculate grid layout based on number of groups
    - Create fixed-size buffer for element generation
    - Loop over groups and place elements in grid pattern
    - Each group contains 6 elements arranged in 2x3 grid
    - Convert pattern to SampleFunction PyTree

    Implementation Details:
    - Handles None groups parameter at Python time (defaults to range(-2, 8))
    - If dx is None: dx = 25.4e-3 / dpi meters (converts dpi to meters)
    - Scale factor = image_size / 1024.0
    - Grid size = ceil(sqrt(num_groups))
    - Cell size = image_size / (grid_size + 1)
    - Element spacing = 5 * scale_factor pixels
    - Buffer size = max(500, image_size // 2) to avoid traced shape creation
    - Uses nested fori_loops: outer for groups, inner for 6 elements per group
    - Group position: (row, col) in grid, converted to canvas coordinates
    - Element position within group: arranged in 2 rows x 3 columns
    - Elements placed using dynamic_update_slice with clipped coordinates
    - Returns SampleFunction PyTree with pattern as complex array
    """
    groups_to_use: Iterable[int] = (
        groups if groups is not None else range(-2, 8)
    )
    groups_array: Float[Array, " n"] = jnp.array(
        list(groups_to_use), dtype=jnp.int32
    )

    if dx is None:
        mm_per_inch: float = 25.4
        dx_calculated: scalar_float = (mm_per_inch * 1e-3) / dpi
    else:
        dx_calculated = dx

    canvas: Float[Array, " image_size image_size"] = (
        jnp.ones((image_size, image_size), dtype=jnp.float32) * 0.5
    )
    scale_factor: scalar_float = image_size / 1024.0
    num_groups: int = len(groups_array)
    grid_size: int = int(jnp.ceil(jnp.sqrt(num_groups)))
    cell_size: int = image_size // (grid_size + 1)
    element_spacing: int = int(5 * scale_factor)

    buffer_size: int = min(image_size, max(500, image_size // 2))
    blank_pattern: Int[Array, " buffer_size buffer_size"] = jnp.zeros(
        (buffer_size, buffer_size), dtype=jnp.int32
    )

    def _process_all_groups(
        group_idx: int, canvas_carry: Float[Array, " image_size image_size"]
    ) -> Float[Array, " image_size image_size"]:
        """Process a single group and place all its elements."""
        group: scalar_integer = groups_array[group_idx]

        row: Float[Array, " "] = jnp.asarray(group_idx) // grid_size
        col: Float[Array, " "] = jnp.asarray(group_idx) % grid_size
        y_pos: Float[Array, " "] = (row + 0.5) * cell_size
        x_pos: Float[Array, " "] = (col + 0.5) * cell_size

        def _process_all_elements(
            elem_idx: int,
            canvas_elem_carry: Float[Array, " image_size image_size"],
        ) -> Float[Array, " image_size image_size"]:
            """Place a single element on the canvas."""
            element: int = elem_idx + 1

            elem: Float[Array, " h w"] = create_element(
                blank_pattern, group, element, scale_factor, dpi
            )

            e_row: Float[Array, " "] = jnp.asarray(elem_idx) // 3
            e_col: Float[Array, " "] = jnp.asarray(elem_idx) % 3

            elem_y_float: Float[Array, " "] = y_pos + e_row * element_spacing
            elem_x_float: Float[Array, " "] = x_pos + e_col * element_spacing

            elem_y: scalar_integer = jnp.clip(
                elem_y_float, 0, image_size - buffer_size
            ).astype(jnp.int32)
            elem_x: scalar_integer = jnp.clip(
                elem_x_float, 0, image_size - buffer_size
            ).astype(jnp.int32)

            canvas_updated: Float[Array, " image_size image_size"] = (
                jax.lax.dynamic_update_slice(
                    canvas_elem_carry, elem, (elem_y, elem_x)
                )
            )

            return canvas_updated

        canvas_with_elements: Float[Array, " image_size image_size"] = (
            jax.lax.fori_loop(0, 6, _process_all_elements, canvas_carry)
        )

        return canvas_with_elements

    final_canvas: Float[Array, " image_size image_size"] = jax.lax.fori_loop(
        0, num_groups, _process_all_groups, canvas
    )
    created_pattern: SampleFunction = make_sample_function(
        final_canvas, dx_calculated
    )
    return created_pattern
