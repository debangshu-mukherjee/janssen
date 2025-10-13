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
from beartype.typing import Any, Iterable, Optional
from jaxtyping import Array, Float, jaxtyped

from janssen.utils import scalar_float


@jaxtyped(typechecker=beartype)
def create_bar_triplet(
    width: scalar_float,
    length: scalar_float,
    spacing: scalar_float,
    horizontal: bool = True,
) -> Float[Array, " h w"]:
    """Create 3 parallel bars (horizontal or vertical).
    
    Parameters
    ----------
    width : scalar_float
        Width of the bars
    length : scalar_float
        Length of the bars
    spacing : scalar_float
        Spacing between the bars
    horizontal : bool, optional
        Whether to create horizontal bars, by default True

    Returns
    -------
    pattern : Float[Array, " h w"]
        The created pattern

    Notes
    -----
    Algorithm:
    - Create a 2D array of zeros with the given length and width
    - Set the pixels in the array to 1 for the bars
    - Return the created pattern
    - If horizontal is True, create 3 horizontal bars
    - If horizontal is False, create 3 vertical bars
    - Return the created pattern
    """
    bar_width: int = int(width)
    bar_length: int = int(length)
    gap: int = int(spacing)

    def _horizontal() -> Float[Array, " h w"]:
        """Create 3 horizontal bars."""
        height: int = 3 * bar_width + 2 * gap
        pattern: Float[Array, " h w"] = jnp.zeros((height, bar_length))
        pattern = pattern.at[0:bar_width, :].set(1)
        pattern = pattern.at[
            bar_width + gap : 2 * bar_width + gap, :
        ].set(1)
        pattern_out: Float[Array, " h w"] = pattern.at[
            2 * bar_width + 2 * gap : 3 * bar_width + 2 * gap, :
        ].set(1)
        return pattern_out

    def _vertical() -> Float[Array, " h w"]:
        """Create 3 vertical bars."""
        width_total: int = 3 * bar_width + 2 * gap
        pattern: Float[Array, " h w"] = jnp.zeros((bar_length, width_total))
        pattern = pattern.at[:, 0:bar_width].set(1)
        pattern = pattern.at[
            :, bar_width + gap : 2 * bar_width + gap
        ].set(1)
        pattern_out: Float[Array, " h w"] = pattern.at[
            :, 2 * bar_width + 2 * gap : 3 * bar_width + 2 * gap
        ].set(1)
        return pattern_out

    result: Float[Array, " h w"] = jax.lax.cond(
        horizontal, _horizontal, _vertical
    )

    return result


@jaxtyped(typechecker=beartype)
def create_element(
    group: int,
    element: int,
    scale_factor: scalar_float,
    dpi: scalar_float,
) -> Float[Array, " h w"]:
    """Create a single element (horizontal + vertical bars).
    
    Parameters
    ----------
    group : int
        Group number
    element : int
        Element number
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
    - Create the horizontal bars
    - Create the vertical bars
    - Create the element image
    - Return the element image
    """
    resolution_lp_mm: scalar_float = 2.0 ** (group + (element - 1) / 6.0)
    mm_per_inch: float = 25.4
    pixels_per_mm: scalar_float = dpi / mm_per_inch
    pixels_per_lp: scalar_float = pixels_per_mm / resolution_lp_mm
    bar_width: Float[Array, " "] = pixels_per_lp / 2.0 * scale_factor
    bar_width = jnp.maximum(bar_width, 1.0)
    bar_length: Float[Array, " "] = bar_width * 5.0
    h_bars: Float[Array, " h w"] = create_bar_triplet(
        bar_width, bar_length, bar_width, horizontal=True
    )
    v_bars: Float[Array, " h w"] = create_bar_triplet(
        bar_width, bar_length, bar_width, horizontal=False
    )
    gap: int = int(bar_width * 0.5)
    h_height: int
    h_width: int
    h_height, h_width = h_bars.shape
    v_height: int
    v_width: int
    v_height, v_width = v_bars.shape
    total_width: int = h_width + gap + v_width
    total_height: int = max(h_height, v_height)
    element_img: Float[Array, " h w"] = jnp.zeros(
        (total_height, total_width)
    )
    h_offset: int = (total_height - h_height) // 2
    element_img = element_img.at[
        h_offset : h_offset + h_height, :h_width
    ].set(h_bars)
    v_offset: int = (total_height - v_height) // 2
    element_img_out: Float[Array, " h w"] = element_img.at[
        v_offset : v_offset + v_height, h_width + gap :
    ].set(v_bars)
    return element_img_out


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

    def _use_default_groups() -> Float[Array, " n"]:
        """Return default groups range."""
        return jnp.array(list(range(-2, 8)))

    def _use_provided_groups() -> Float[Array, " n"]:
        """Return provided groups."""
        return jnp.array(list(groups))

    groups_array: Float[Array, " n"] = jax.lax.cond(
        groups is None, _use_default_groups, _use_provided_groups
    )
    canvas: Float[Array, " image_size image_size"] = (
        jnp.ones((image_size, image_size)) * 0.5
    )
    scale_factor: scalar_float = image_size / 1024.0
    num_groups: int = len(groups_array)
    grid_size: int = int(jnp.ceil(jnp.sqrt(num_groups)))
    cell_size: int = image_size // (grid_size + 1)
    element_spacing: int = int(5 * scale_factor)

    def _process_group(
        carry: Float[Array, " image_size image_size"],
        idx_group_tuple: Any,
    ) -> tuple[Float[Array, " image_size image_size"], None]:
        """Process a single group and place all its elements on canvas."""
        idx: int
        group: int
        idx, group = idx_group_tuple
        canvas_local: Float[Array, " image_size image_size"] = carry
        row: int = idx // grid_size
        col: int = idx % grid_size
        y_pos: int = int((row + 0.5) * cell_size)
        x_pos: int = int((col + 0.5) * cell_size)
        element_indices: Float[Array, " 6"] = jnp.arange(1, 7)

        def _create_single_element(element: int) -> Float[Array, " h w"]:
            """Create a single element."""
            return create_element(int(group), int(element), scale_factor, dpi)

        group_elements: Float[Array, " 6 h w"] = jax.vmap(
            _create_single_element
        )(element_indices)

        def _place_element(
            canvas_carry: Float[Array, " image_size image_size"],
            e_idx_elem_tuple: Any,
        ) -> tuple[Float[Array, " image_size image_size"], None]:
            """Place a single element on the canvas."""
            e_idx: int
            elem: Float[Array, " h w"]
            e_idx, elem = e_idx_elem_tuple
            canvas_inner: Float[Array, " image_size image_size"] = canvas_carry
            e_row: int = e_idx // 3
            e_col: int = e_idx % 3
            elem_h: int = elem.shape[0]
            elem_w: int = elem.shape[1]
            elem_y: int = (
                y_pos
                + e_row * (int(elem_h * 1.2) + element_spacing)
                - int(elem_h)
            )
            elem_x: int = (
                x_pos
                + e_col * (int(elem_w * 1.2) + element_spacing)
                - int(elem_w * 1.5)
            )
            elem_y = jnp.clip(elem_y, 0, image_size - elem_h)
            elem_x = jnp.clip(elem_x, 0, image_size - elem_w)
            canvas_updated: Float[Array, " image_size image_size"] = (
                canvas_inner.at[
                    elem_y : elem_y + elem_h, elem_x : elem_x + elem_w
                ].set(elem)
            )

            return (canvas_updated, None)

        element_tuples = list(enumerate(group_elements))
        canvas_with_elements: Float[Array, " image_size image_size"]
        canvas_with_elements, _ = jax.lax.scan(
            _place_element, canvas_local, element_tuples
        )
        return (canvas_with_elements, None)

    group_tuples = list(enumerate(groups_array))
    final_canvas: Float[Array, " image_size image_size"]
    final_canvas, _ = jax.lax.scan(_process_group, canvas, group_tuples)

    return final_canvas
