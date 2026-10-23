"""Recalculate the Milky Way environment values in the paper's Table 1."""

from astropy import constants as const
from astropy import units as u

from captus.utils.misc import get_MW_params


RADII_KPC = (8.0, 2.2, 0.05)  # [kpc]


def main() -> None:
    print("r [kpc]  rho_DM [GeV cm^-3]  v_c [km s^-1]  sigma_DM [km s^-1]")
    for radius_kpc in RADII_KPC:
        params = get_MW_params(radius_kpc)
        density_gev_cm3 = (
            params["DM_density"].to(u.kg / u.m**3) * const.c**2
        ).to_value(u.GeV / u.cm**3)
        print(
            f"{radius_kpc:6.2f}  {density_gev_cm3:17.3f}  "
            f"{params['v_circ']:13.2f}  {params['DM_dispersion']:18.2f}"
        )


if __name__ == "__main__":
    main()