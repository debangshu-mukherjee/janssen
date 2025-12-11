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

    Both horizontal and vertical patterns are computed, then selected
    based on the horizontal flag. This is JAX-safe since the flag is
    a static Python bool that doesn't change during tracing.
    """
    width_val: int = max(1, width)
    length_val: int = max(1, length)
    total_bar_extent: int = 5 * width_val
    h_h: int = total_bar_extent
    h_w: int = length_val
    y_coords: Float[Array, " h 1"] = jnp.arange(h_h, dtype=jnp.float32)[
        :, None
    ]
    bar1_h: Float[Array, " h 1"] = (y_coords < width_val).astype(jnp.float32)
    bar2_h: Float[Array, " h 1"] = (
        (y_coords >= 2 * width_val) & (y_coords < 3 * width_val)
    ).astype(jnp.float32)
    bar3_h: Float[Array, " h 1"] = (y_coords >= 4 * width_val).astype(
        jnp.float32
    )
    pattern_h: Float[Array, " h w"] = jnp.broadcast_to(
        bar1_h + bar2_h + bar3_h, (h_h, h_w)
    )
    v_h: int = length_val
    v_w: int = total_bar_extent
    x_coords: Float[Array, " 1 w"] = jnp.arange(v_w, dtype=jnp.float32)[
        None, :
    ]
    bar1_v: Float[Array, " 1 w"] = (x_coords < width_val).astype(jnp.float32)
    bar2_v: Float[Array, " 1 w"] = (
        (x_coords >= 2 * width_val) & (x_coords < 3 * width_val)
    ).astype(jnp.float32)
    bar3_v: Float[Array, " 1 w"] = (x_coords >= 4 * width_val).astype(
        jnp.float32
    )
    pattern_v: Float[Array, " h w"] = jnp.broadcast_to(
        bar1_v + bar2_v + bar3_v, (v_h, v_w)
    )
    pattern: Float[Array, "..."] = pattern_h if horizontal else pattern_v
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

    Triplets are centered vertically within the element canvas.
    """
    bar_width: int = max(1, bar_width_px)
    bar_length: int = 5 * bar_width
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
    element: Float[Array, " eh ew"] = jnp.zeros(
        (element_height, element_width), dtype=jnp.float32
    )
    h_y_offset: int = (element_height - h_height) // 2
    element = element.at[
        h_y_offset : h_y_offset + h_height, :h_width
    ].set(h_triplet)
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

    The loop over 6 elements is unrolled at trace time since
    the element count is fixed.
    """
    elements: list[Float[Array, "..."]] = []
    element_heights: list[int] = []
    element_widths: list[int] = []
    for elem in range(1, 7):
        bar_width: int = get_bar_width_pixels(group, elem, pixels_per_mm)
        element: Float[Array, "..."] = create_element_pattern(bar_width)
        elements.append(element)
        element_heights.append(int(element.shape[0]))
        element_widths.append(int(element.shape[1]))
    max_elem_width: int = max(element_widths)
    elem_spacing: int = max(2, int(max_elem_width * 0.2))
    col1_heights: list[int] = element_heights[0:3]
    col2_heights: list[int] = element_heights[3:6]
    col1_widths: list[int] = element_widths[0:3]
    col2_widths: list[int] = element_widths[3:6]
    col1_height: int = sum(col1_heights) + elem_spacing * 2
    col2_height: int = sum(col2_heights) + elem_spacing * 2
    total_height: int = max(col1_height, col2_height)
    col1_width: int = max(col1_widths)
    col2_width: int = max(col2_widths)
    col_gap: int = max(2, int(col1_width * 0.4))
    total_width: int = col1_width + col_gap + col2_width
    group_pattern: Float[Array, " h w"] = jnp.zeros(
        (total_height, total_width), dtype=jnp.float32
    )
    y_pos: int = 0
    for i in range(3):
        elem = elements[i]
        eh: int = element_heights[i]
        ew: int = element_widths[i]
        x_offset: int = (col1_width - ew) // 2
        group_pattern = group_pattern.at[
            y_pos : y_pos + eh, x_offset : x_offset + ew
        ].set(elem)
        y_pos += eh + elem_spacing
    y_pos = 0
    x_base: int = col1_width + col_gap
    for i in range(3, 6):
        elem = elements[i]
        eh = element_heights[i]
        ew = element_widths[i]
        x_offset = x_base + (col2_width - ew) // 2
        group_pattern = group_pattern.at[
            y_pos : y_pos + eh, x_offset : x_offset + ew
        ].set(elem)
        y_pos += eh + elem_spacing
    max_dimension: int = max(total_height, total_width)
    return group_pattern, max_dimension


@jaxtyped(typechecker=beartype)
def calculate_usaf_group_range(
    image_size: int,
    pixel_size: float,
    min_bar_pixels: int = 2,
    grid_fill_fraction: float = 0.95,
) -> dict:
    """Calculate the viable USAF group range for given parameters.

    Parameters
    ----------
    image_size : int
        Image size in pixels (square)
    pixel_size : float
        Pixel size in meters
    min_bar_pixels : int, optional
        Minimum bar width in pixels for visibility, by default 2
    grid_fill_fraction : float, optional
        Scale factor for fitting largest group, by default 0.95

    Returns
    -------
    result : dict
        Dictionary containing:
        - max_group: finest group where bars are still >= min_bar_pixels
        - min_group: coarsest group that fits in image
        - recommended_range: suggested range() for generate_usaf_pattern
        - num_groups: how many groups in recommended range
        - pixels_per_mm: calculated pixel density
        - group_info: dict with bar width info for each group

    Notes
    -----
    This function maximizes the number of groups that can fit in the image
    using variable-density row packing (smaller groups pack more per row).

    Examples
    --------
    >>> result = calculate_usaf_group_range(
    ...     image_size=8192,
    ...     pixel_size=0.5e-6,
    ... )
    >>> print(f"Use: groups=range({result['min_group']}, {result['max_group'] + 1})")
    """
    pixels_per_mm: float = 1e-3 / pixel_size

    # Max group: bars must be at least min_bar_pixels wide
    # For element 6 (finest in group): bar_width = pixels_per_mm / (2 * 2^(g + 5/6))
    max_group: int = int(math.floor(
        math.log2(pixels_per_mm / (2 * min_bar_pixels)) - 5 / 6
    ))

    def get_group_size(g: int, ppm: float) -> Tuple[int, int]:
        """Estimate group pattern size in pixels.
        
        Returns (height, width) of the group pattern.
        """
        bar_width: float = ppm / (2 * (2 ** g))
        bar_width = max(1, bar_width)
        elem_height: float = 5 * bar_width
        elem_width: float = 10.5 * bar_width
        elem_spacing: float = 0.2 * elem_width
        col_height: float = 3 * elem_height + 2 * elem_spacing
        col_gap: float = 0.4 * elem_width
        total_width: float = 2 * elem_width + col_gap
        return max(1, int(col_height)), max(1, int(total_width))

    def simulate_packing(groups: list[int], ppm: float, img_size: int) -> int:
        """Simulate row-packing and return how many groups fit."""
        margin: int = img_size // 40
        usable: int = img_size - 2 * margin
        spacing_h: int = max(5, img_size // 200)
        spacing_v: int = max(20, img_size // 80)
        
        # Check if largest group fits
        largest_g: int = min(groups)
        largest_h, largest_w = get_group_size(largest_g, ppm)
        
        effective_ppm: float = ppm
        if largest_h > usable or largest_w > usable:
            scale: float = min(usable / largest_h, usable / largest_w) * grid_fill_fraction
            effective_ppm = ppm * scale
        
        # Simulate packing
        current_x: int = margin
        current_y: int = margin
        row_max_h: int = 0
        count: int = 0
        
        for g in groups:
            gh, gw = get_group_size(g, effective_ppm)
            
            if current_x + gw > img_size - margin:
                current_x = margin
                current_y += row_max_h + spacing_v
                row_max_h = 0
            
            if current_y + gh > img_size - margin:
                break
            
            current_x += gw + spacing_h
            row_max_h = max(row_max_h, gh)
            count += 1
        
        return count

    # Find min_group that maximizes the number of groups that fit
    best_min_group: int = max_group
    best_count: int = 1
    
    for candidate_min in range(-10, max_group + 1):
        groups_to_try: list[int] = list(range(candidate_min, max_group + 1))
        count: int = simulate_packing(groups_to_try, pixels_per_mm, image_size)
        
        if count >= len(groups_to_try):
            # All groups fit
            if len(groups_to_try) > best_count:
                best_count = len(groups_to_try)
                best_min_group = candidate_min
    
    min_group: int = best_min_group

    # Build group info
    group_info: dict = {}
    for g in range(min_group, max_group + 1):
        bar_width_e1: float = pixels_per_mm / (2 * (2 ** g))
        bar_width_e6: float = pixels_per_mm / (2 * (2 ** (g + 5 / 6)))
        gh, gw = get_group_size(g, pixels_per_mm)
        group_info[g] = {
            "bar_width_element1": round(bar_width_e1, 1),
            "bar_width_element6": round(bar_width_e6, 1),
            "group_size_approx": f"{gh}x{gw}",
        }

    recommended_range: range = range(min_group, max_group + 1)

    return {
        "max_group": max_group,
        "min_group": min_group,
        "recommended_range": recommended_range,
        "num_groups": len(recommended_range),
        "pixels_per_mm": pixels_per_mm,
        "group_info": group_info,
    }


@jaxtyped(typechecker=beartype)
def generate_usaf_pattern(
    image_size: int = 1024,
    groups: Optional[range] = None,
    pixel_size: ScalarFloat = 1.0e-6,
    background: float = 0.0,
    foreground: float = 1.0,
    max_phase: float = 0.0,
    auto: bool = False,
    min_bar_pixels: int = 2,
) -> SampleFunction:
    """Generate USAF 1951 resolution test pattern.

    Parameters
    ----------
    image_size : int, optional
        Size of the output image (square), by default 1024
    groups : range, optional
        Range of groups to include, by default range(-2, 8).
        Ignored if auto=True.
    pixel_size : ScalarFloat, optional
        Physical size of each pixel in meters, by default 1.0e-6 (1 µm)
    background : float, optional
        Background value, by default 0.0 (black)
    foreground : float, optional
        Foreground (bar) value, by default 1.0 (white)
    max_phase : float, optional
        Maximum phase shift in radians applied to the bars, by default 0.0.
        The phase pattern follows the same structure as the amplitude,
        scaling from 0 (at background) to max_phase (at foreground).
    auto : bool, optional
        If True, automatically calculate the optimal group range to
        fill the image based on image_size and pixel_size. Overrides
        the groups parameter. By default False.
    min_bar_pixels : int, optional
        Minimum bar width in pixels for visibility when auto=True,
        by default 2. Ignored if auto=False.

    Returns
    -------
    pattern : SampleFunction
        SampleFunction PyTree containing the USAF test pattern as a
        complex array with both amplitude and phase information.

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

    The output is a complex field: amplitude * exp(i * phase), where
    the phase follows the same spatial pattern as the amplitude.

    The loop over groups is unrolled at Python trace time since
    groups_list is known before tracing. Python-level conditionals
    for bounds checking, scaling, and phase normalization are evaluated
    at trace time since all controlling values are Python scalars.
    
    A global scale factor is computed from the largest (coarsest) group
    to ensure all groups fit within their grid cells while preserving
    the correct relative size ratios between groups.

    Examples
    --------
    >>> from janssen.models import generate_usaf_pattern
    >>> pattern = generate_usaf_pattern(image_size=1024, pixel_size=1e-6)
    >>> pattern.sample.shape
    (1024, 1024)

    >>> # Auto mode: fill the image optimally
    >>> pattern = generate_usaf_pattern(image_size=8192, pixel_size=0.5e-6, auto=True)

    >>> # Camera with 6.5 µm pixels
    >>> pattern = generate_usaf_pattern(pixel_size=6.5e-6)

    >>> # White background with black bars (typical)
    >>> pattern = generate_usaf_pattern(background=1.0, foreground=0.0)

    >>> # Phase object with π phase shift on bars
    >>> pattern = generate_usaf_pattern(max_phase=jnp.pi)

    >>> # Specific group range
    >>> pattern = generate_usaf_pattern(groups=range(0, 5))
    """
    # Handle auto mode
    if auto:
        range_info = calculate_usaf_group_range(
            image_size=image_size,
            pixel_size=float(pixel_size),
            min_bar_pixels=min_bar_pixels,
        )
        groups_list: list[int] = list(range_info["recommended_range"])
    else:
        groups_list = (
            list(groups) if groups is not None else list(range(-2, 8))
        )
    dx_calculated: ScalarFloat = float(pixel_size)
    pixels_per_mm: float = 1.0e-3 / float(pixel_size)
    canvas: Float[Array, " h w"] = jnp.full(
        (image_size, image_size), background, dtype=jnp.float32
    )
    
    margin: int = image_size // 40  # Smaller margin for more space
    usable_size: int = image_size - 2 * margin
    spacing_h: int = max(5, image_size // 200)  # Horizontal spacing (tight)
    
    # Helper to estimate group size without generating full pattern
    def estimate_group_size(group: int, ppm: float) -> Tuple[int, int]:
        """Estimate group size without generating full pattern."""
        bar_width: int = get_bar_width_pixels(group, 1, ppm)
        bar_length: int = 5 * bar_width
        elem_height: int = 5 * bar_width
        elem_width: int = bar_length + max(1, int(bar_width * 0.5)) + 5 * bar_width
        elem_spacing: int = max(2, int(elem_width * 0.2))
        col_height: int = 3 * elem_height + 2 * elem_spacing
        col_gap: int = max(2, int(elem_width * 0.4))
        total_width: int = 2 * elem_width + col_gap
        return col_height, total_width
    
    # Check if largest (coarsest) group fits in usable area
    largest_group: int = min(groups_list)
    largest_h, largest_w = estimate_group_size(largest_group, pixels_per_mm)
    
    # Only scale if the largest group doesn't fit at all
    effective_ppm: float = pixels_per_mm
    if largest_h > usable_size or largest_w > usable_size:
        scale: float = min(usable_size / largest_h, usable_size / largest_w) * 0.95
        effective_ppm = pixels_per_mm * scale
    
    # First pass: determine rows and their heights (dry run)
    rows: list[list[Tuple[int, int, int]]] = []  # Each row: list of (group, gh, gw)
    current_row: list[Tuple[int, int, int]] = []
    current_x: int = margin
    
    for group in groups_list:
        gh, gw = estimate_group_size(group, effective_ppm)
        
        if current_x + gw > image_size - margin and current_row:
            # Start new row
            rows.append(current_row)
            current_row = []
            current_x = margin
        
        current_row.append((group, gh, gw))
        current_x += gw + spacing_h
    
    if current_row:
        rows.append(current_row)
    
    # Calculate row heights
    row_heights: list[int] = []
    for row in rows:
        max_h = max(gh for _, gh, _ in row)
        row_heights.append(max_h)
    
    # Calculate total row height and distribute vertical space equally
    total_row_height: int = sum(row_heights)
    num_gaps: int = len(rows) + 1  # gaps above first row, between rows, after last row
    total_free_space: int = image_size - total_row_height
    spacing_v: int = total_free_space // num_gaps if num_gaps > 0 else margin
    
    # Second pass: actually place the groups with calculated spacing
    current_y: int = spacing_v
    
    for row_idx, row in enumerate(rows):
        row_height: int = row_heights[row_idx]
        current_x = margin
        
        # Calculate total row width to center the row
        row_width: int = sum(gw for _, _, gw in row) + spacing_h * (len(row) - 1)
        current_x = (image_size - row_width) // 2  # Center the row
        
        for group, gh_est, gw_est in row:
            # Check if we've run out of vertical space
            if current_y + row_height > image_size - spacing_v // 2:
                break
            
            # Generate actual pattern
            pattern, _ = create_group_pattern(group, effective_ppm)
            gh: int = int(pattern.shape[0])
            gw: int = int(pattern.shape[1])
            
            # Vertically center within row
            y_offset: int = (row_height - gh) // 2
            x_pos: int = current_x
            y_pos: int = current_y + y_offset
            
            # Clip if necessary
            gh_clipped: int = min(gh, image_size - y_pos)
            gw_clipped: int = min(gw, image_size - x_pos)
            
            if gh_clipped > 0 and gw_clipped > 0 and y_pos >= 0 and x_pos >= 0:
                clipped_pattern = pattern[:gh_clipped, :gw_clipped]
                scaled_pattern: Float[Array, " gh gw"] = (
                    background + clipped_pattern * (foreground - background)
                )
                canvas = canvas.at[
                    y_pos : y_pos + gh_clipped, x_pos : x_pos + gw_clipped
                ].set(scaled_pattern)
            
            current_x += gw + spacing_h
        
        current_y += row_height + spacing_v
    
    # Normalize canvas to [0, 1] for phase calculation
    # Use Python conditional since foreground/background are known at trace time
    if foreground != background:
        normalized_pattern: Float[Array, " h w"] = (canvas - background) / (
            foreground - background
        )
    else:
        normalized_pattern = jnp.zeros_like(canvas)
    phase_pattern: Float[Array, " h w"] = normalized_pattern * max_phase
    complex_field = canvas.astype(jnp.complex64) * jnp.exp(
        1j * phase_pattern.astype(jnp.complex64)
    )
    pattern: SampleFunction = make_sample_function(
        complex_field, dx_calculated
    )
    return pattern
