
import rebound
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
from astropy.table import Table
import logging
import sys
import pandas as pd
from astropy import constants as const
from astropy import units as u  

# pathToData = '/data/a.saricaoglu/repo/COMPAS/Files/msstarvsx_prefirstMT.fits'
# with fits.open(pathToData) as hdul:
#     data = hdul[1].data
#     subset = data[:]
#     print(data.columns)
#     comp_mass = subset["Companion_mass"]
#     msstar_mass = subset["MSstar_mass"]
#     comp_radius  = subset["Companion_radius"]
#     comp_radius = (comp_radius * const.R_sun).to(u.au).value
#     msstar_radius  = subset["MSstar_radius"]
#     msstar_radius = (msstar_radius * const.R_sun).to(u.au).value
#     semajax = subset["Semimajor_axis"]
#     orbperiod = subset["Orbital_period"]
#     orbperiod = (orbperiod * u.day).to(u.yr).value
#     # print(comp_mass)
#     # print(msstar_mass)
#     # print(semaj)
#     binary_df = pd.DataFrame()
#     print(len(binary_df))
#     cmas = []
#     msmas = []
#     crad = []
#     msrad = []
#     semaj = []
#     orbper = []

# mass_ratios = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Define the mass ratio groups
# population_limit = 10
# mass_ratio_groups = {ratio: [] for ratio in mass_ratios}  # Dictionary to track populations

# for cm, mm, cr, mr, a, t in zip(comp_mass, msstar_mass, comp_radius, msstar_radius, semajax, orbperiod):
#     if (cm > 0) and (mm > 0) and (mm > cm):
#         mass_ratio = mm / cm  # Calculate the mass ratio
#         assigned = False

#         # Check if the mass ratio falls within ±0.05 of any group
#         for target_ratio in mass_ratios:
#             if target_ratio - 0.05 <= mass_ratio <= target_ratio + 0.05:
#                 if len(mass_ratio_groups[target_ratio]) < population_limit:
#                     # Add the binary to the group
#                     mass_ratio_groups[target_ratio].append((cm, mm, cr, mr, a, t))
#                     assigned = True
#                     break  # Stop checking once assigned

#         if not assigned:
#             continue  # Skip this binary if it doesn't fit into any group

# print("Mass ratio groups:")
# for target_ratio, binaries in mass_ratio_groups.items():
#     print(f"Target ratio {target_ratio}: {len(binaries)} binaries")
# # Flatten the grouped binaries and add a mass ratio group column
# group_labels = []  # To store the mass ratio group for each binary
# for target_ratio, binaries in mass_ratio_groups.items():
#     for cm, mm, cr, mr, a, t in binaries:
#         cmas.append(cm)
#         msmas.append(mm)
#         crad.append(cr)
#         msrad.append(mr)
#         semaj.append(a)
#         orbper.append(t)
#         group_labels.append(target_ratio)  # Add the mass ratio group label

# # Create a dictionary for the FITS table columns
# binary_data = {
#     "Mass_Ratio_Group": group_labels,  # Add the mass ratio group as a column
#     "Companion_mass": cmas,
#     "MSstar_mass": msmas,
#     "Companion_radius": crad,
#     "MSstar_radius": msrad,
#     "Semimajor_axis": semaj,
#     "Orbital_period": orbper,
# }

# # Convert the dictionary to an Astropy Table
# binary_table = Table(binary_data)

# # Write the table to a FITS file
# output_fits_path = "/data/a.saricaoglu/repo/COMPAS/Files/binary_data_prefirstMT.fits"
# binary_table.write(output_fits_path, format="fits", overwrite=True)

# print(f"Binary data saved to {output_fits_path}")
# #Read the FITS file
# with fits.open("/data/a.saricaoglu/repo/COMPAS/Files/binary_data_prefirstMT.fits") as hdul:
#     data = hdul[1].data  # Access the binary table
#     print(data.columns)  # Print the column names

#     # Filter binaries for a specific mass ratio group (e.g., 1.0)
#     target_ratio = 32.0
#     filtered_binaries = data[data["Mass_Ratio_Group"] == target_ratio]

#     print(f"Binaries in mass ratio group {target_ratio}:")
#     for binary in filtered_binaries:
#         print(binary)

# pathToData = '/data/a.saricaoglu/repo/COMPAS/Files/msstarvsx_pstlastMT.fits'
# with fits.open(pathToData) as hdul:
#     data = hdul[1].data
#     subset = data[:]
#     print(data.columns)
#     comp_mass = subset["Companion_mass"]
#     msstar_mass = subset["MSstar_mass"]
#     comp_radius  = subset["Companion_radius"]
#     comp_radius = (comp_radius * const.R_sun).to(u.au).value
#     msstar_radius  = subset["MSstar_radius"]
#     msstar_radius = (msstar_radius * const.R_sun).to(u.au).value
#     semajax = subset["Semimajor_axis"]
#     orbperiod = subset["Orbital_period"]
#     orbperiod = (orbperiod * u.day).to(u.yr).value
#     # print(comp_mass)
#     # print(msstar_mass)
#     # print(semaj)
#     binary_df = pd.DataFrame()
#     print(len(binary_df))
#     cmas = []
#     msmas = []
#     crad = []
#     msrad = []
#     semaj = []
#     orbper = []

# mass_ratios = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Define the mass ratio groups
# population_limit = 10
# mass_ratio_groups = {ratio: [] for ratio in mass_ratios}  # Dictionary to track populations

# for cm, mm, cr, mr, a, t in zip(comp_mass, msstar_mass, comp_radius, msstar_radius, semajax, orbperiod):
#     if (cm > 0) and (mm > 0) and (mm > cm):
#         mass_ratio = mm / cm  # Calculate the mass ratio
#         assigned = False

#         # Check if the mass ratio falls within ±0.05 of any group
#         for target_ratio in mass_ratios:
#             if target_ratio - 0.05 <= mass_ratio <= target_ratio + 0.05:
#                 if len(mass_ratio_groups[target_ratio]) < population_limit:
#                     # Add the binary to the group
#                     mass_ratio_groups[target_ratio].append((cm, mm, cr, mr, a, t))
#                     assigned = True
#                     break  # Stop checking once assigned

#         if not assigned:
#             continue  # Skip this binary if it doesn't fit into any group

# print("Mass ratio groups:")
# for target_ratio, binaries in mass_ratio_groups.items():
#     print(f"Target ratio {target_ratio}: {len(binaries)} binaries")
# # Flatten the grouped binaries and add a mass ratio group column
# group_labels = []  # To store the mass ratio group for each binary
# for target_ratio, binaries in mass_ratio_groups.items():
#     for cm, mm, cr, mr, a, t in binaries:
#         cmas.append(cm)
#         msmas.append(mm)
#         crad.append(cr)
#         msrad.append(mr)
#         semaj.append(a)
#         orbper.append(t)
#         group_labels.append(target_ratio)  # Add the mass ratio group label

# # Create a dictionary for the FITS table columns
# binary_data = {
#     "Mass_Ratio_Group": group_labels,  # Add the mass ratio group as a column
#     "Companion_mass": cmas,
#     "MSstar_mass": msmas,
#     "Companion_radius": crad,
#     "MSstar_radius": msrad,
#     "Semimajor_axis": semaj,
#     "Orbital_period": orbper,
# }

# # Convert the dictionary to an Astropy Table
# binary_table = Table(binary_data)

# # Write the table to a FITS file
# output_fits_path = "/data/a.saricaoglu/repo/COMPAS/Files/binary_data_pstlastMT.fits"
# binary_table.write(output_fits_path, format="fits", overwrite=True)

# print(f"Binary data saved to {output_fits_path}")
# #Read the FITS file
# with fits.open("/data/a.saricaoglu/repo/COMPAS/Files/binary_data_pstlastMT.fits") as hdul:
#     data = hdul[1].data  # Access the binary table
#     print(data.columns)  # Print the column names

#     # Filter binaries for a specific mass ratio group (e.g., 1.0)
#     target_ratio = 32.0
#     filtered_binaries = data[data["Mass_Ratio_Group"] == target_ratio]

#     print(f"Binaries in mass ratio group {target_ratio}:")
#     for binary in filtered_binaries:
#         print(binary)
# pathToData = '/data/a.saricaoglu/repo/COMPAS/Files/msstarvscomp_SP.fits'
# with fits.open(pathToData) as hdul:
#     data = hdul[1].data
#     subset = data[:]
#     print(data.columns)
#     comp_mass = subset["Companion_mass"]
#     msstar_mass = subset["MSstar_mass"]
#     comp_radius  = subset["Companion_radius"]
#     comp_radius = (comp_radius * const.R_sun).to(u.au).value
#     msstar_radius  = subset["MSstar_radius"]
#     msstar_radius = (msstar_radius * const.R_sun).to(u.au).value
#     semajax = subset["Semimajor_axis"]
#     orbperiod = subset["Orbital_period"]
#     orbperiod = (orbperiod * u.day).to(u.yr).value
#     # print(comp_mass)
#     # print(msstar_mass)
#     # print(semaj)
#     binary_df = pd.DataFrame()
#     print(len(binary_df))
#     cmas = []
#     msmas = []
#     crad = []
#     msrad = []
#     semaj = []
#     orbper = []

# mass_ratios = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]  # Define the mass ratio groups
# population_limit = 10
# mass_ratio_groups = {ratio: [] for ratio in mass_ratios}  # Dictionary to track populations

# for cm, mm, cr, mr, a, t in zip(comp_mass, msstar_mass, comp_radius, msstar_radius, semajax, orbperiod):
#     if (cm > 0) and (mm > 0) :
#         mass_ratio = mm / cm  # Calculate the mass ratio
#         assigned = False

#         # Check if the mass ratio falls within ±0.05 of any group
#         for target_ratio in mass_ratios:
#             if target_ratio - 0.05 <= mass_ratio <= target_ratio + 0.05:
#                 if len(mass_ratio_groups[target_ratio]) < population_limit:
#                     # Add the binary to the group
#                     mass_ratio_groups[target_ratio].append((cm, mm, cr, mr, a, t))
#                     assigned = True
#                     break  # Stop checking once assigned

#         if not assigned:
#             continue  # Skip this binary if it doesn't fit into any group

# print("Mass ratio groups:")
# for target_ratio, binaries in mass_ratio_groups.items():
#     print(f"Target ratio {target_ratio}: {len(binaries)} binaries")
# # Flatten the grouped binaries and add a mass ratio group column
# group_labels = []  # To store the mass ratio group for each binary
# for target_ratio, binaries in mass_ratio_groups.items():
#     for cm, mm, cr, mr, a, t in binaries:
#         cmas.append(cm)
#         msmas.append(mm)
#         crad.append(cr)
#         msrad.append(mr)
#         semaj.append(a)
#         orbper.append(t)
#         group_labels.append(target_ratio)  # Add the mass ratio group label

# # Create a dictionary for the FITS table columns
# binary_data = {
#     "Mass_Ratio_Group": group_labels,  # Add the mass ratio group as a column
#     "Companion_mass": cmas,
#     "MSstar_mass": msmas,
#     "Companion_radius": crad,
#     "MSstar_radius": msrad,
#     "Semimajor_axis": semaj,
#     "Orbital_period": orbper,
# }

# # Convert the dictionary to an Astropy Table
# binary_table = Table(binary_data)

# # Write the table to a FITS file
# output_fits_path = "/data/a.saricaoglu/repo/COMPAS/Files/binary_data_msstarvscomp_SP.fits"
# binary_table.write(output_fits_path, format="fits", overwrite=True)

# print(f"Binary data saved to {output_fits_path}")
# #Read the FITS file
# with fits.open("/data/a.saricaoglu/repo/COMPAS/Files/binary_data_msstarvscomp_SP.fits") as hdul:
#     data = hdul[1].data  # Access the binary table
#     print(data.columns)  # Print the column names

#     # Filter binaries for a specific mass ratio group (e.g., 1.0)
#     target_ratio = 64.0
#     filtered_binaries = data[data["Mass_Ratio_Group"] == target_ratio]

#     print(f"Binaries in mass ratio group {target_ratio}:")
#     for binary in filtered_binaries:
#         print(binary)

def pick(number, ratio):
    # with fits.open("/data/a.saricaoglu/repo/COMPAS/Files/binary_data_msstarvscomp_SP.fits") as hdul:
    #     data = hdul[1].data  # Access the binary table
    #     filtered_binaries = data[data["Mass_Ratio_Group"] == ratio]
        #return filtered_binaries[number]  # Return the first 'number' binaries
    return [64, 0.0009543, 1.0, (0.000477895*(const.R_sun)).to(u.au).value, (const.R_sun).to(u.au).value, 5.2, 11.89 ]