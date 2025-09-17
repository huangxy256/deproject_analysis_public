import numpy as np
from jampy.mge_half_light_isophote import mge_half_light_isophote


def get_sigma_e(surf_lum, sigma_lum, qobs_lum, jam, xbin, ybin):
    """calculate velocity dispersion within the half-light radius from a jam model

    Args:
        surf_lum (_type_): peak of surface luminosity MGE
        sigma_lum (_type_): sigma of surface luminocity MGE
        qobs_lum (_type_): array of the projected axis raio of the surface luminosity MGEs
        jam (_type_): jam model, a jampy.jam_axi_proj instance
        xbin (_type_): x coordinate to sample the velocity dispersion
        ybin (_type_): y coordinate to sample the velocity dispersion


    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    ifu_dim = int(np.sqrt(len(xbin)))
    if np.all(qobs_lum <= 1):
        flux = jam.flux
    elif np.all(qobs_lum > 1):
        flux = np.reshape(jam.flux, (ifu_dim, ifu_dim)).T
        flux = flux.flatten()
    else:
        raise ValueError("Apparent axis ratio must be constant with radius!")

    reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(
        surf_lum, sigma_lum, qobs_lum
    )

    w = xbin**2 + ybin**2 < reff**2

    model = jam.model

    sig_e = np.sqrt((flux[w] * model[w] ** 2).sum() / flux[w].sum())

    return sig_e
