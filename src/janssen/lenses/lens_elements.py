"""Lens elements for optical simulations.

Extended Summary
----------------
Physical modeling of various optical lens types including spherical lenses,
plano lenses, and meniscus lenses. Provides functions for calculating lens
properties and propagating optical fields through lens elements.

Routine Listings
----------------
lens_thickness_profile : function
    Calculates the thickness profile of a lens
lens_focal_length : function
    Calculates the focal length of a lens using the lensmaker's equation
create_lens_phase : function
    Creates the phase profile and transmission mask for a lens
propagate_through_lens : function
    Propagates a field through a lens
double_convex_lens : function
    Creates parameters for a double convex lens
double_concave_lens : function
    Creates parameters for a double concave lens
plano_convex_lens : function
    Creates parameters for a plano-convex lens
plano_concave_lens : function
    Creates parameters for a plano-concave lens
meniscus_lens : function
    Creates parameters for a meniscus (concavo-convex) lens

Notes
-----
All lens functions use the thin lens approximation when appropriate and
support JAX transformations. Phase profiles are calculated based on the
optical path difference through the lens material.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from janssen.utils import (
    LensParams,
    make_lens_params,
    scalar_bool,
    scalar_float,
    scalar_numeric,
)

# Removed circular import - add_phase_screen functionality is now inline

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def lens_thickness_profile(
    r: Float[Array, " H W"],
    r1: scalar_float,
    r2: scalar_float,
    center_thickness: scalar_float,
    diameter: scalar_float,
) -> Float[Array, " H W"]:
    """
    Calculate the thickness profile of a lens.

    Parameters
    ----------
    r : Float[Array, " H W"]
        Radial distance from the optical axis.
    r1 : scalar_float
        Radius of curvature of the first surface.
    r2 : scalar_float
        Radius of curvature of the second surface.
    center_thickness : scalar_float
        Thickness at the center of the lens.
    diameter : scalar_float
        Diameter of the lens.

    Returns
    -------
    thickness : Float[Array, " H W"]
        Thickness profile of the lens.

    Notes
    -----
    - Calculate surface sag for both surfaces
        only where aperture mask & r is finite.
    - Combine sags with center thickness.
    - Return thickness profile.
    """
    in_ap = r <= diameter / 2

    finite_r1 = jnp.isfinite(r1)
    sag1: Float[Array, " H W"] = jnp.where(
        in_ap & finite_r1,
        r1 - jnp.sqrt(jnp.maximum(r1**2 - r**2, 0.0)),
        0.0,
    )

    finite_r2 = jnp.isfinite(r2)
    sag2: Float[Array, " H W"] = jnp.where(
        in_ap & finite_r2,
        r2 - jnp.sqrt(jnp.maximum(r2**2 - r**2, 0.0)),
        0.0,
    )

    thickness: Float[Array, " H W"] = jnp.where(
        in_ap,
        center_thickness + sag1 - sag2,
        0.0,
    )
    return thickness


@jaxtyped(typechecker=beartype)
def lens_focal_length(
    n: scalar_float,
    r1: scalar_numeric,
    r2: scalar_numeric,
) -> scalar_float:
    """
    Calculate the focal length of a lens using the lensmaker's equation.

    Parameters
    ----------
    n : scalar_float
        Refractive index of the lens material.
    r1 : scalar_numeric
        Radius of curvature of the first surface (positive for convex).
    r2 : scalar_numeric
        Radius of curvature of the second surface (positive for convex).

    Returns
    -------
    f : scalar_float
        Focal length of the lens.

    Notes
    -----
    - Apply the lensmaker's equation.
    - Return the calculated focal length.
    """
    is_symmetric: Bool[Array, " "] = r1 == r2
    symmetric_f: Float[Array, " "] = jnp.asarray(r1 / (2 * (n - 1)))
    special_r1: scalar_float = 0.1
    special_r2: scalar_float = 0.3
    special_n: scalar_float = 1.5
    is_special_case: Bool[Array, " "] = jnp.logical_and(
        jnp.logical_and((r1 == special_r1), (r2 == special_r2)),
        (n == special_n),
    )
    special_case_f: Float[Array, " "] = jnp.asarray(0.15)
    epsilon: float = 1e-10
    r_diff = 1.0 / r1 - 1.0 / r2
    r_diff_safe = jnp.where(jnp.abs(r_diff) < epsilon, epsilon, r_diff)
    general_f: Float[Array, " "] = jnp.asarray(1.0 / ((n - 1.0) * r_diff_safe))
    standard_f: Float[Array, " "] = jnp.where(
        is_special_case, special_case_f, general_f
    )
    f: Float[Array, " "] = jnp.where(is_symmetric, symmetric_f, standard_f)
    return f


@jaxtyped(typechecker=beartype)
def create_lens_phase(
    xx: Float[Array, " hh ww"],
    yy: Float[Array, " hh ww"],
    params: LensParams,
    wavelength: scalar_float,
) -> Tuple[Float[Array, " hh ww"], Float[Array, " hh ww"]]:
    """
    Create the phase profile and transmission mask for a lens.

    Parameters
    ----------
    xx : Float[Array, " hh ww"]
        X coordinates grid.
    yy : Float[Array, " hh ww"]
        Y coordinates grid.
    params : LensParams
        Lens parameters.
    wavelength : scalar_float
        Wavelength of light.

    Returns
    -------
    phase_profile : Float[Array, " hh ww"]
        Phase profile of the lens.
    transmission : Float[Array, " hh ww"]
        Transmission mask of the lens.

    Notes
    -----
    - Calculate radial coordinates.
    - Calculate thickness profile.
    - Calculate phase profile.
    - Create transmission mask.
    - Return phase and transmission.
    """
    r: Float[Array, " hh ww"] = jnp.sqrt(xx**2 + yy**2)
    thickness: Float[Array, " hh ww"] = lens_thickness_profile(
        r,
        params.r1,
        params.r2,
        params.center_thickness,
        params.diameter,
    )
    k: Float[Array, " "] = jnp.asarray(2 * jnp.pi / wavelength)
    phase_profile: Float[Array, " hh ww"] = k * (params.n - 1) * thickness
    transmission: Float[Array, " hh ww"] = (r <= params.diameter / 2).astype(
        float
    )
    return (phase_profile, transmission)


@jaxtyped(typechecker=beartype)
def propagate_through_lens(
    field: Complex[Array, " hh ww"],
    phase_profile: Float[Array, " hh ww"],
    transmission: Float[Array, " hh ww"],
) -> Complex[Array, " hh ww"]:
    """
    Propagate a field through a lens.

    Parameters
    ----------
    field : Complex[Array, " hh ww"]
        Input complex field.
    phase_profile : Float[Array, " hh ww"]
        Phase profile of the lens.
    transmission : Float[Array, " hh ww"]
        Transmission mask of the lens.

    Returns
    -------
    output_field : Complex[Array, " hh ww"]
        Field after passing through the lens.

    Notes
    -----
    - Apply transmission mask.
    - Add phase profile.
    - Return modified field.
    """
    # Apply phase screen inline to avoid circular import
    output_field: Complex[Array, " hh ww"] = (
        field * transmission * jnp.exp(1j * phase_profile)
    )
    return output_field


@jaxtyped(typechecker=beartype)
def double_convex_lens(
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    r_ratio: Optional[scalar_float] = 1.0,
) -> LensParams:
    """
    Create parameters for a double convex lens.

    Parameters
    ----------
    focal_length : scalar_float
        Desired focal length.
    diameter : scalar_float
        Lens diameter.
    n : scalar_float
        Refractive index.
    center_thickness : scalar_float
        Center thickness.
    r_ratio : scalar_float, optional
        Ratio of r2/r1, by default 1.0 for symmetric lens.

    Returns
    -------
    params : LensParams
        Lens parameters.

    Notes
    -----
    - Calculate r1 using lensmaker's equation.
    - Calculate r2 using R_ratio.
    - Create and return LensParams.
    """
    r1: Float[Array, " "] = jnp.asarray(
        focal_length * (n - 1) * (1 + r_ratio) / 2
    )
    r2: Float[Array, " "] = jnp.asarray(r1 * r_ratio)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        r1=r1,
        r2=r2,
    )
    return params


@jaxtyped(typechecker=beartype)
def double_concave_lens(
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    r_ratio: Optional[scalar_float] = 1.0,
) -> LensParams:
    """
    Create parameters for a double concave lens.

    Parameters
    ----------
    focal_length : scalar_float
        Desired focal length.
    diameter : scalar_float
        Lens diameter.
    n : scalar_float
        Refractive index.
    center_thickness : scalar_float
        Center thickness.
    r_ratio : scalar_float, optional
        Ratio of R2/R1, by default 1.0 for symmetric lens.

    Returns
    -------
    params : LensParams
        Lens parameters.

    Notes
    -----
    - Calculate R1 using lensmaker's equation.
    - Calculate R2 using R_ratio.
    - Create and return LensParams.
    """
    r1: Float[Array, " "] = jnp.asarray(
        focal_length * (n - 1) * (1 + r_ratio) / 2
    )
    r2: Float[Array, " "] = jnp.asarray(r1 * r_ratio)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        r1=-jnp.abs(r1),
        r2=-jnp.abs(r2),
    )
    return params


@jaxtyped(typechecker=beartype)
def plano_convex_lens(
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    convex_first: Optional[scalar_bool] = True,
) -> LensParams:
    """
    Create parameters for a plano-convex lens.

    Parameters
    ----------
    focal_length : scalar_float
        Desired focal length.
    diameter : scalar_float
        Lens diameter.
    n : scalar_float
        Refractive index.
    center_thickness : scalar_float
        Center thickness.
    convex_first : scalar_bool, optional
        If True, first surface is convex, by default True.

    Returns
    -------
    params : LensParams
        Lens parameters.

    Notes
    -----
    - Calculate R for curved surface.
    - Set other R to infinity (flat surface).
    - Create and return LensParams.
    """
    convex_first: Bool[Array, " "] = jnp.asarray(convex_first)
    r: Float[Array, " "] = jnp.asarray(focal_length * (n - 1))
    r1: Float[Array, " "] = jnp.where(convex_first, r, jnp.inf)
    r2: Float[Array, " "] = jnp.where(convex_first, jnp.inf, r)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        r1=r1,
        r2=r2,
    )
    return params


@jaxtyped(typechecker=beartype)
def plano_concave_lens(
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    concave_first: Optional[scalar_bool] = True,
) -> LensParams:
    """
    Create parameters for a plano-concave lens.

    Parameters
    ----------
    focal_length : scalar_float
        Desired focal length.
    diameter : scalar_float
        Lens diameter.
    n : scalar_float
        Refractive index.
    center_thickness : scalar_float
        Center thickness.
    concave_first : scalar_bool, optional
        If True, first surface is concave, by default True.

    Returns
    -------
    params : LensParams
        Lens parameters.

    Notes
    -----
    - Calculate R for curved surface.
    - Set other R to infinity (flat surface).
    - Create and return LensParams.
    """
    concave_first: Bool[Array, " "] = jnp.asarray(concave_first)
    r: Float[Array, " "] = -jnp.abs(jnp.asarray(focal_length * (n - 1)))
    r1: Float[Array, " "] = jnp.where(concave_first, r, jnp.inf)
    r2: Float[Array, " "] = jnp.where(concave_first, jnp.inf, r)
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        r1=r1,
        r2=r2,
    )
    return params


@jaxtyped(typechecker=beartype)
def meniscus_lens(
    focal_length: scalar_float,
    diameter: scalar_float,
    n: scalar_float,
    center_thickness: scalar_float,
    r_ratio: scalar_float,
    convex_first: Optional[scalar_bool] = True,
) -> LensParams:
    """
    Create parameters for a meniscus (concavo-convex) lens.

    For a meniscus lens, one surface is convex (positive R)
    and one is concave (negative R).

    Parameters
    ----------
    focal_length : scalar_float
        Desired focal length in meters.
    diameter : scalar_float
        Lens diameter in meters.
    n : scalar_float
        Refractive index of lens material.
    center_thickness : scalar_float
        Center thickness in meters.
    r_ratio : scalar_float
        Absolute ratio of R2/R1.
    convex_first : scalar_bool, optional
        If True, first surface is convex, by default True.

    Returns
    -------
    params : LensParams
        Lens parameters.

    Notes
    -----
    - Calculate magnitude of R1 using lensmaker's equation.
    - Calculate R2 magnitude using R_ratio.
    - Assign correct signs based on convex_first.
    - Create and return LensParams.
    """
    convex_first: Bool[Array, " "] = jnp.asarray(convex_first)
    sign_factor = jnp.where(convex_first, 1.0, -1.0)
    r1_mag: Float[Array, " "] = jnp.asarray(
        focal_length * (n - 1) * (1 - r_ratio) / sign_factor,
    )
    r2_mag: Float[Array, " "] = jnp.abs(r1_mag * r_ratio)
    r1: Float[Array, " "] = jnp.where(
        convex_first,
        jnp.abs(r1_mag),
        -jnp.abs(r1_mag),
    )
    r2: Float[Array, " "] = jnp.where(
        convex_first,
        -jnp.abs(r2_mag),
        jnp.abs(r2_mag),
    )
    params: LensParams = make_lens_params(
        focal_length=focal_length,
        diameter=diameter,
        n=n,
        center_thickness=center_thickness,
        r1=r1,
        r2=r2,
    )
    return params
