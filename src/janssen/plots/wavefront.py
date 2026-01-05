"""Wavefront plotting functions.

Extended Summary
----------------
Functions for visualizing optical wavefronts including amplitude, phase,
and complex field representations using matplotlib.
This module is NOT JAX-accelerated.

Routine Listings
----------------
plot_amplitude : function
    Plot the amplitude of an optical wavefront
plot_complex_wavefront : function
    Plot a complex optical wavefront using HSV color mapping
plot_intensity : function
    Plot the intensity of an optical wavefront
plot_phase : function
    Plot the phase of an optical wavefront using HSV color mapping
_plot_field : function, internal
    Internal function for plotting the field of an optical wavefront

Notes
-----
All plotting functions use matplotlib and matplotlib-scalebar for
publication-quality figures with proper scale annotations.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype
from beartype.typing import List, Literal, Optional, Tuple, Union
from jaxtyping import Complex, Float
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from numpy import ndarray as NDArray

from janssen.types import OpticalWavefront


def _plot_field(
    wavefront: OpticalWavefront,
    plot_fn: Callable[[Axes, Complex[NDArray, " hh ww"]], AxesImage],
    figsize: Tuple[float, float],
    scalebar_length: float | None,
    scalebar_units: str,
    title: Optional[str],
    colorbar_location: Optional[str] = None,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Tuple[Axes, Axes]]]:
    """Handle plotting logic for both scalar and polarized fields."""
    field: Complex[NDArray, " hh ww"] = np.asarray(wavefront.field)
    dx: float = float(wavefront.dx)
    is_polarized: bool = field.ndim == 3  # noqa: PLR2004

    if is_polarized:
        polarized_figsize: Tuple[float, float] = (figsize[0] * 2, figsize[1])
        fig: Figure
        ax1: Axes
        ax2: Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=polarized_figsize)
        axes: Tuple[Axes, Axes] = (ax1, ax2)
        fields: List[Complex[NDArray, " hh ww"]] = [
            field[:, :, 0],
            field[:, :, 1],
        ]
        labels: List[str] = ["Ex", "Ey"]

        ax: Axes
        f: Complex[NDArray, " hh ww"]
        label: str
        for ax, f, label in zip(axes, fields, labels, strict=True):
            im = plot_fn(ax, f)
            ax.axis("off")
            ax.set_title(label)

            if colorbar_location:
                divider: AxesDivider = make_axes_locatable(ax)
                is_vertical: bool = colorbar_location in ("left", "right")
                orientation: str = "vertical" if is_vertical else "horizontal"
                cax: Axes = divider.append_axes(
                    colorbar_location, size="5%", pad=0.05
                )
                fig.colorbar(im, cax=cax, orientation=orientation)

            scalebar: ScaleBar = ScaleBar(
                dx,
                units=scalebar_units,
                length_fraction=0.25,
                location="lower right",
                color="white",
                box_alpha=0.5,
                fixed_value=scalebar_length,
            )
            ax.add_artist(scalebar)

        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()
        return fig, axes

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=figsize)

    im = plot_fn(ax, field)
    ax.axis("off")

    if colorbar_location:
        divider: AxesDivider = make_axes_locatable(ax)
        is_vertical: bool = colorbar_location in ("left", "right")
        orientation: str = "vertical" if is_vertical else "horizontal"
        cax: Axes = divider.append_axes(colorbar_location, size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation=orientation)

    scalebar: ScaleBar = ScaleBar(
        dx,
        units=scalebar_units,
        length_fraction=0.25,
        location="lower right",
        color="white",
        box_alpha=0.5,
        fixed_value=scalebar_length,
    )
    ax.add_artist(scalebar)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    return fig, ax


@beartype
def plot_complex_wavefront(
    wavefront: OpticalWavefront,
    figsize: Tuple[float, float] = (6, 5),
    scalebar_length: float | None = None,
    scalebar_units: str = "m",
    title: Optional[str] = None,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Tuple[Axes, Axes]]]:
    """Plot a complex optical wavefront using HSV color mapping.

    Parameters
    ----------
    wavefront : OpticalWavefront
        The optical wavefront to plot. Contains field, wavelength, dx,
        z_position, and polarization attributes.
    figsize : Tuple[float, float], optional
        Figure size in inches (width, height). Default is (6, 5).
    scalebar_length : Optional[float], optional
        Length of the scalebar in the units specified by scalebar_units.
        If None, matplotlib-scalebar will choose automatically.
    scalebar_units : str, optional
        Units for the scalebar. Default is "m" (meters).
    title : Optional[str], optional
        Title for the figure. If None, no title is added.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object.
    ax : Axes or Tuple[Axes, Axes]
        The matplotlib Axes object. For polarized wavefronts, returns a tuple
        of two Axes (Ex, Ey).

    Notes
    -----
    The function creates a single plot using HSV color mapping where:

    - Hue: Represents the phase arg(U(x,y)), mapped from [-π, π] to [0, 1]
    - Saturation: Set to 1 (fully saturated colors)
    - Value: Represents the amplitude |U(x,y)|, normalized to [0, 1]

    This representation allows simultaneous visualization of both amplitude
    and phase in a single image. Bright colors indicate high amplitude,
    while dark regions indicate low amplitude. The color itself encodes the
    phase.

    A scalebar is added based on the wavefront's dx (pixel size) parameter.

    For polarized wavefronts (3D field with shape [H, W, 2]), two side-by-side
    plots are created showing Ex and Ey components.
    """

    def _plot_complex(ax: Axes, f: Complex[NDArray, " hh ww"]) -> AxesImage:
        amplitude: Float[NDArray, " hh ww"] = np.abs(f)
        phase: Float[NDArray, " hh ww"] = np.angle(f)
        amplitude_normalized: Float[NDArray, " hh ww"] = amplitude / (
            np.max(amplitude) + 1e-10
        )

        hue: Float[NDArray, " hh ww"] = (phase + np.pi) / (2 * np.pi)
        saturation: Float[NDArray, " hh ww"] = np.ones_like(
            amplitude_normalized
        )
        value: Float[NDArray, " hh ww"] = amplitude_normalized

        hsv_image: Float[NDArray, " hh ww 3"] = np.stack(
            [hue, saturation, value], axis=-1
        )
        rgb_image: Float[NDArray, " hh ww 3"] = hsv_to_rgb(hsv_image)

        return ax.imshow(rgb_image, origin="lower")

    return _plot_field(
        wavefront,
        _plot_complex,
        figsize,
        scalebar_length,
        scalebar_units,
        title,
        colorbar_location=None,
    )


@beartype
def plot_amplitude(
    wavefront: OpticalWavefront,
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "gray",
    colorbar_location: Literal["top", "bottom", "left", "right"] = "right",
    colorbar_min: float | None = None,
    scalebar_length: float | None = None,
    scalebar_units: str = "m",
    title: Optional[str] = None,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Tuple[Axes, Axes]]]:
    """Plot the amplitude of an optical wavefront.

    Parameters
    ----------
    wavefront : OpticalWavefront
        The optical wavefront to plot. Contains field, wavelength, dx,
        z_position, and polarization attributes.
    figsize : Tuple[float, float], optional
        Figure size in inches (width, height). Default is (6, 5).
    cmap : str, optional
        Colormap for the amplitude plot. Default is "gray".
    colorbar_location : Literal["top", "bottom", "left", "right"], optional
        Location of the colorbar. Default is "right".
    colorbar_min : float | None, optional
        Minimum value for the colorbar. If None, uses the minimum value
        of the amplitude data. Default is None.
    scalebar_length : Optional[float], optional
        Length of the scalebar in the units specified by scalebar_units.
        If None, matplotlib-scalebar will choose automatically.
    scalebar_units : str, optional
        Units for the scalebar. Default is "m" (meters).
    title : Optional[str], optional
        Title for the figure. If None, no title is added.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object.
    ax : Axes or Tuple[Axes, Axes]
        The matplotlib Axes object. For polarized wavefronts, returns a tuple
        of two Axes (Ex, Ey).

    Notes
    -----
    The function plots the amplitude |U(x,y)| of the complex field.
    A colorbar is included to show the amplitude scale, sized to match
    the image dimensions.

    A scalebar is added based on the wavefront's dx (pixel size) parameter.

    For polarized wavefronts (3D field with shape [H, W, 2]), two side-by-side
    plots are created showing Ex and Ey components.
    """

    def _plot_amp(ax: Axes, f: Complex[NDArray, " hh ww"]) -> AxesImage:
        amplitude: Float[NDArray, " hh ww"] = np.abs(f)
        vmin: float
        if colorbar_min is not None:
            vmin = colorbar_min
        else:
            vmin = float(np.min(amplitude))
        return ax.imshow(amplitude, cmap=cmap, origin="lower", vmin=vmin)

    return _plot_field(
        wavefront,
        _plot_amp,
        figsize,
        scalebar_length,
        scalebar_units,
        title,
        colorbar_location=colorbar_location,
    )


@beartype
def plot_intensity(
    wavefront: OpticalWavefront,
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "gray",
    colorbar_location: Literal["top", "bottom", "left", "right"] = "right",
    colorbar_min: float | None = None,
    scalebar_length: float | None = None,
    scalebar_units: str = "m",
    title: Optional[str] = None,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Tuple[Axes, Axes]]]:
    """Plot the intensity of an optical wavefront.

    Parameters
    ----------
    wavefront : OpticalWavefront
        The optical wavefront to plot. Contains field, wavelength, dx,
        z_position, and polarization attributes.
    figsize : Tuple[float, float], optional
        Figure size in inches (width, height). Default is (6, 5).
    cmap : str, optional
        Colormap for the intensity plot. Default is "gray".
    colorbar_location : Literal["top", "bottom", "left", "right"], optional
        Location of the colorbar. Default is "right".
    colorbar_min : float | None, optional
        Minimum value for the colorbar. If None, uses the minimum value
        of the intensity data. Default is None.
    scalebar_length : Optional[float], optional
        Length of the scalebar in the units specified by scalebar_units.
        If None, matplotlib-scalebar will choose automatically.
    scalebar_units : str, optional
        Units for the scalebar. Default is "m" (meters).
    title : Optional[str], optional
        Title for the figure. If None, no title is added.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object.
    ax : Axes or Tuple[Axes, Axes]
        The matplotlib Axes object. For polarized wavefronts, returns a tuple
        of two Axes (Ex, Ey).

    Notes
    -----
    The function plots the intensity |U(x,y)|² of the complex field.
    A colorbar is included to show the intensity scale, sized to match
    the image dimensions.

    A scalebar is added based on the wavefront's dx (pixel size) parameter.

    For polarized wavefronts (3D field with shape [H, W, 2]), two side-by-side
    plots are created showing Ex and Ey components.
    """

    def _plot_int(ax: Axes, f: Complex[NDArray, " hh ww"]) -> AxesImage:
        intensity: Float[NDArray, " hh ww"] = np.abs(f) ** 2
        vmin: float
        if colorbar_min is not None:
            vmin = colorbar_min
        else:
            vmin = float(np.min(intensity))
        return ax.imshow(intensity, cmap=cmap, origin="lower", vmin=vmin)

    return _plot_field(
        wavefront,
        _plot_int,
        figsize,
        scalebar_length,
        scalebar_units,
        title,
        colorbar_location=colorbar_location,
    )


@beartype
def plot_phase(
    wavefront: OpticalWavefront,
    figsize: Tuple[float, float] = (6, 5),
    scalebar_length: float | None = None,
    scalebar_units: str = "m",
    title: Optional[str] = None,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, Tuple[Axes, Axes]]]:
    """Plot the phase of an optical wavefront using HSV color mapping.

    Parameters
    ----------
    wavefront : OpticalWavefront
        The optical wavefront to plot. Contains field, wavelength, dx,
        z_position, and polarization attributes.
    figsize : Tuple[float, float], optional
        Figure size in inches (width, height). Default is (6, 5).
    scalebar_length : Optional[float], optional
        Length of the scalebar in the units specified by scalebar_units.
        If None, matplotlib-scalebar will choose automatically.
    scalebar_units : str, optional
        Units for the scalebar. Default is "m" (meters).
    title : Optional[str], optional
        Title for the figure. If None, no title is added.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object.
    ax : Axes or Tuple[Axes, Axes]
        The matplotlib Axes object. For polarized wavefronts, returns a tuple
        of two Axes (Ex, Ey).

    Notes
    -----
    The function plots the phase arg(U(x,y)) using HSV color mapping where:

    - Hue: Represents the phase, mapped from [-π, π] to [0, 1]
    - Saturation: Set to 1 (fully saturated colors)
    - Value: Set to 1 (full brightness)

    This creates a uniform brightness image where only the color encodes
    the phase information. The cyclic nature of HSV hue naturally represents
    the cyclic nature of phase.

    A scalebar is added based on the wavefront's dx (pixel size) parameter.

    For polarized wavefronts (3D field with shape [H, W, 2]), two side-by-side
    plots are created showing Ex and Ey components.
    """

    def _plot_ph(ax: Axes, f: Complex[NDArray, " hh ww"]) -> AxesImage:
        phase: Float[NDArray, " hh ww"] = np.angle(f)

        hue: Float[NDArray, " hh ww"] = (phase + np.pi) / (2 * np.pi)
        saturation: Float[NDArray, " hh ww"] = np.ones_like(hue)
        value: Float[NDArray, " hh ww"] = np.ones_like(hue)

        hsv_image: Float[NDArray, " hh ww 3"] = np.stack(
            [hue, saturation, value], axis=-1
        )
        rgb_image: Float[NDArray, " hh ww 3"] = hsv_to_rgb(hsv_image)

        return ax.imshow(rgb_image, origin="lower")

    return _plot_field(
        wavefront,
        _plot_ph,
        figsize,
        scalebar_length,
        scalebar_units,
        title,
        colorbar_location=None,
    )
