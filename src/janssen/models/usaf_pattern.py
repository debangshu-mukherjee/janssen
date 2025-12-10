"""USAF 1951 resolution test pattern generation.

Extended Summary
----------------
Generates USAF 1951 resolution test patterns using pure JAX operations.
The pattern follows the MIL-STD-150A specification with correctly scaled
and positioned groups and elements.

Routine Listings
----------------
create_bar_triplet : function
    Creates 3 parallel bars (horizontal or vertical)
create_element_pattern : function
    Creates a single element (horizontal + vertical bar triplets)
create_group_pattern : function
    Creates a complete group with 6 elements
get_bar_width_pixels : function
    Calculates bar width in pixels for given group and element
generate_usaf_pattern : function
    Generates USAF 1951 resolution test pattern

Notes
-----
All functions use JAX operations and support automatic differentiation.
The USAF 1951 pattern follows the resolution formula:
    Resolution = 2^(group + (element-1)/6) line pairs per mm

Each successive element increases resolution by a factor of 2^(1/6) ≈ 1.122
Each successive group increases resolution by a factor of 2.
"""

import math

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import (
    Array,
    Float,
    jaxtyped,
)

from janssen.utils import (
    SampleFunction,
    ScalarFloat,
    make_sample_function,
)


@jaxtyped(typechecker=beartype)
def create_bar_triplet(
    width: int,
    length: int,
    horizontal: bool = True,
) -> Float[Array, "..."]:
    """Create 3 parallel bars (horizontal or vertical).

    Parameters
    ----------
    width : int
        Width of each bar in pixels (minimum 1)
    length : int
        Length of each bar in pixels (minimum 1)
    horizontal : bool, optional
        Whether to create horizontal bars, by default True

    Returns
    -------
    pattern : Float[Array, "..."]
        The bar triplet pattern. Shape depends on orientation:
        - Horizontal: (5*width, length)
        - Vertical: (length, 5*width)

    Notes
    -----
    Creates three bars following USAF specification where bar spacing
    equals bar width. Total extent is 5 × bar_width (3 bars + 2 spaces).

    Pattern structure (for horizontal):
    - Bar 1: rows [0, width)
    - Space: rows [width, 2*width)
    - Bar 2: rows [2*width, 3*width)
    - Space: rows [3*width, 4*width)
    - Bar 3: rows [4*width, 5*width)
    """
    width_val: int = max(1, width)
    length_val: int = max(1, length)

    # Total extent: 3 bars + 2 spaces = 5 × width
    total_bar_extent: int = 5 * width_val

    if horizontal:
        # Horizontal bars: height = 5*width, width = length
        h: int = total_bar_extent
        w: int = length_val
        y_coords: Float[Array, " h 1"] = jnp.arange(h, dtype=jnp.float32)[
            :, None
        ]

        # Bars at: [0, width), [2*width, 3*width), [4*width, 5*width)
        bar1: Float[Array, " h 1"] = (y_coords < width_val).astype(jnp.float32)
        bar2: Float[Array, " h 1"] = (
            (y_coords >= 2 * width_val) & (y_coords < 3 * width_val)
        ).astype(jnp.float32)
        bar3: Float[Array, " h 1"] = (y_coords >= 4 * width_val).astype(
            jnp.float32
        )

        pattern: Float[Array, " h w"] = jnp.broadcast_to(
            bar1 + bar2 + bar3, (h, w)
        )
    else:
        # Vertical bars: height = length, width = 5*width
        h = length_val
        w = total_bar_extent
        x_coords: Float[Array, " 1 w"] = jnp.arange(w, dtype=jnp.float32)[
            None, :
        ]

        bar1 = (x_coords < width_val).astype(jnp.float32)
        bar2 = (
            (x_coords >= 2 * width_val) & (x_coords < 3 * width_val)
        ).astype(jnp.float32)
        bar3 = (x_coords >= 4 * width_val).astype(jnp.float32)

        pattern = jnp.broadcast_to(bar1 + bar2 + bar3, (h, w))

    return pattern


@jaxtyped(typechecker=beartype)
def create_element_pattern(
    bar_width_px: int,
    gap_factor: float = 0.5,
) -> Float[Array, "..."]:
    """Create a single USAF element (horizontal + vertical bar triplets).

    Parameters
    ----------
    bar_width_px : int
        Width of each bar in pixels (minimum 1)
    gap_factor : float, optional
        Gap between triplets as fraction of bar_width, by default 0.5

    Returns
    -------
    element : Float[Array, "..."]
        The complete element pattern with both triplets

    Notes
    -----
    Each USAF element consists of:
    - 3 horizontal bars (triplet) on the left
    - 3 vertical bars (triplet) on the right
    Bar length is 5× the bar width per USAF specification.

    The element is composed as:
    [horizontal triplet] [gap] [vertical triplet]
    """
    bar_width: int = max(1, bar_width_px)
    bar_length: int = 5 * bar_width  # Standard 5:1 aspect ratio

    # Create triplets
    h_triplet: Float[Array, " hh hw"] = create_bar_triplet(
        bar_width, bar_length, horizontal=True
    )
    v_triplet: Float[Array, " vh vw"] = create_bar_triplet(
        bar_width, bar_length, horizontal=False
    )

    gap: int = max(1, int(bar_width * gap_factor))

    h_height: int = h_triplet.shape[0]
    h_width: int = h_triplet.shape[1]
    v_height: int = v_triplet.shape[0]
    v_width: int = v_triplet.shape[1]

    element_height: int = max(h_height, v_height)
    element_width: int = h_width + gap + v_width

    # Create element canvas
    element: Float[Array, " eh ew"] = jnp.zeros(
        (element_height, element_width), dtype=jnp.float32
    )

    # Center and place horizontal triplet
    h_y_offset: int = (element_height - h_height) // 2
    element = element.at[h_y_offset : h_y_offset + h_height, :h_width].set(
        h_triplet
    )

    # Center and place vertical triplet
    v_y_offset: int = (element_height - v_height) // 2
    v_x_offset: int = h_width + gap
    element = element.at[
        v_y_offset : v_y_offset + v_height, v_x_offset : v_x_offset + v_width
    ].set(v_triplet)

    return element


@jaxtyped(typechecker=beartype)
def get_bar_width_pixels(
    group: int,
    element: int,
    pixels_per_mm: float,
) -> int:
    """Calculate bar width in pixels for a given group and element.

    Parameters
    ----------
    group : int
        Group number (typically -2 to 7)
    element : int
        Element number (1 to 6)
    pixels_per_mm : float
        Pixel density in pixels per millimeter

    Returns
    -------
    bar_width : int
        Bar width in pixels (minimum 1)

    Notes
    -----
    Resolution formula per MIL-STD-150A:
        R = 2^(group + (element-1)/6) line pairs per mm

    One line pair = one bar + one space = 2 × bar_width
    Therefore: bar_width_mm = 1 / (2 × R)
    """
    resolution_lp_mm: float = 2.0 ** (group + (element - 1) / 6.0)
    bar_width_mm: float = 1.0 / (2.0 * resolution_lp_mm)
    bar_width_px: int = int(round(bar_width_mm * pixels_per_mm))
    return max(1, bar_width_px)


@jaxtyped(typechecker=beartype)
def create_group_pattern(
    group: int,
    pixels_per_mm: float,
) -> Tuple[Float[Array, "..."], int]:
    """Create a complete group with 6 elements in 2×3 layout.

    Parameters
    ----------
    group : int
        Group number
    pixels_per_mm : float
        Pixel density in pixels per millimeter

    Returns
    -------
    group_pattern : Float[Array, "..."]
        The complete group pattern
    max_dimension : int
        Maximum dimension of the group

    Notes
    -----
    Elements are arranged in 2 columns × 3 rows:
    - Column 1: Elements 1, 2, 3 (top to bottom)
    - Column 2: Elements 4, 5, 6 (top to bottom)

    Elements within a group progressively decrease in size
    following the 2^((element-1)/6) scaling.
    """
    elements: list[Float[Array, "..."]] = []

    for elem in range(1, 7):
        bar_width: int = get_bar_width_pixels(group, elem, pixels_per_mm)
        element: Float[Array, "..."] = create_element_pattern(bar_width)
        elements.append(element)

    # Find dimensions for layout
    max_elem_width: int = max(int(e.shape[1]) for e in elements)

    # Spacing between elements within group
    elem_spacing: int = max(2, int(max_elem_width * 0.2))

    # Layout: 2 columns × 3 rows
    col1_elements: list[Float[Array, "..."]] = elements[0:3]
    col2_elements: list[Float[Array, "..."]] = elements[3:6]

    col1_height: int = (
        sum(int(e.shape[0]) for e in col1_elements) + elem_spacing * 2
    )
    col2_height: int = (
        sum(int(e.shape[0]) for e in col2_elements) + elem_spacing * 2
    )
    total_height: int = max(col1_height, col2_height)

    col1_width: int = max(int(e.shape[1]) for e in col1_elements)
    col2_width: int = max(int(e.shape[1]) for e in col2_elements)
    col_gap: int = max(2, int(col1_width * 0.4))
    total_width: int = col1_width + col_gap + col2_width

    group_pattern: Float[Array, " h w"] = jnp.zeros(
        (total_height, total_width), dtype=jnp.float32
    )

    # Place column 1 (elements 1, 2, 3)
    y_pos: int = 0
    for elem in col1_elements:
        eh: int = elem.shape[0]
        ew: int = elem.shape[1]
        x_offset: int = (col1_width - ew) // 2
        group_pattern = group_pattern.at[
            y_pos : y_pos + eh, x_offset : x_offset + ew
        ].set(elem)
        y_pos += eh + elem_spacing

    # Place column 2 (elements 4, 5, 6)
    y_pos = 0
    x_base: int = col1_width + col_gap
    for elem in col2_elements:
        eh = elem.shape[0]
        ew = elem.shape[1]
        x_offset = x_base + (col2_width - ew) // 2
        group_pattern = group_pattern.at[
            y_pos : y_pos + eh, x_offset : x_offset + ew
        ].set(elem)
        y_pos += eh + elem_spacing

    max_dimension: int = max(total_height, total_width)
    return group_pattern, max_dimension


@jaxtyped(typechecker=beartype)
def generate_usaf_pattern(
    image_size: int = 1024,
    groups: Optional[range] = None,
    dpi: ScalarFloat = 300.0,
    dx: Optional[ScalarFloat] = None,
    background: float = 0.0,
    foreground: float = 1.0,
) -> SampleFunction:
    """Generate USAF 1951 resolution test pattern.

    Parameters
    ----------
    image_size : int, optional
        Size of the output image (square), by default 1024
    groups : range, optional
        Range of groups to include, by default range(-2, 8)
    dpi : ScalarFloat, optional
        Dots per inch for scaling, by default 300.0
    dx : ScalarFloat, optional
        Spatial sampling interval in meters. If None, calculated from dpi.
    background : float, optional
        Background value, by default 0.0 (black)
    foreground : float, optional
        Foreground (bar) value, by default 1.0 (white)

    Returns
    -------
    pattern : SampleFunction
        SampleFunction PyTree containing the USAF test pattern

    Notes
    -----
    The USAF 1951 test pattern consists of groups arranged in a grid.
    Each group contains 6 elements of progressively higher resolution.

    Resolution formula per MIL-STD-150A:
        Resolution = 2^(group + (element-1)/6) line pairs per mm

    Standard groups range from -2 (coarsest) to 7 (finest).

    Each element consists of:
    - 3 horizontal bars (bar triplet)
    - 3 vertical bars (bar triplet)
    with bar length = 5 × bar width.

    Examples
    --------
    >>> from janssen.models import generate_usaf_pattern
    >>> pattern = generate_usaf_pattern(image_size=1024, dpi=300)
    >>> pattern.amplitude.shape
    (1024, 1024)

    >>> # White background with black bars (typical)
    >>> pattern = generate_usaf_pattern(background=1.0, foreground=0.0)

    >>> # Specific group range
    >>> pattern = generate_usaf_pattern(groups=range(0, 5))
    """
    if groups is None:
        groups = range(-2, 8)
    groups_list: list[int] = list(groups)

    # Calculate spatial sampling
    mm_per_inch: float = 25.4
    if dx is None:
        dx_calculated: ScalarFloat = (mm_per_inch * 1e-3) / float(dpi)
    else:
        dx_calculated = dx

    pixels_per_mm: float = float(dpi) / mm_per_inch

    # Create canvas
    canvas: Float[Array, " h w"] = jnp.full(
        (image_size, image_size), background, dtype=jnp.float32
    )

    num_groups: int = len(groups_list)
    grid_cols: int = int(math.ceil(math.sqrt(num_groups)))
    grid_rows: int = int(math.ceil(num_groups / grid_cols))

    margin: int = image_size // 20
    usable_size: int = image_size - 2 * margin
    cell_width: int = usable_size // grid_cols
    cell_height: int = usable_size // grid_rows

    for idx, group in enumerate(groups_list):
        row: int = idx // grid_cols
        col: int = idx % grid_cols

        group_pattern, _ = create_group_pattern(group, pixels_per_mm)
        gh: int = group_pattern.shape[0]
        gw: int = group_pattern.shape[1]

        # Scale group if it doesn't fit in cell
        max_scale: float = min(cell_width / gw, cell_height / gh) * 0.85
        if max_scale < 1.0:
            scaled_ppm: float = pixels_per_mm * max_scale
            group_pattern, _ = create_group_pattern(group, scaled_ppm)
            gh = group_pattern.shape[0]
            gw = group_pattern.shape[1]

        # Calculate position (centered in cell)
        cell_x: int = margin + col * cell_width
        cell_y: int = margin + row * cell_height
        x_pos: int = cell_x + (cell_width - gw) // 2
        y_pos: int = cell_y + (cell_height - gh) // 2

        # Bounds check
        if y_pos + gh > image_size:
            gh = image_size - y_pos
        if x_pos + gw > image_size:
            gw = image_size - x_pos

        if gh > 0 and gw > 0 and y_pos >= 0 and x_pos >= 0:
            # Scale pattern values from [0, 1] to [background, foreground]
            scaled_pattern: Float[Array, " gh gw"] = (
                background + group_pattern[:gh, :gw] * (foreground - background)
            )
            canvas = canvas.at[y_pos : y_pos + gh, x_pos : x_pos + gw].set(
                scaled_pattern
            )

    # Create SampleFunction PyTree
    pattern: SampleFunction = make_sample_function(canvas, dx_calculated)

    return pattern