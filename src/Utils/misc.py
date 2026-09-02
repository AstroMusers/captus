import os
import pytextable as ptx
import numpy as np
import astropy.constants as const
import pandas as pd
import src.Utils.calculations as calc
def _resolve_repo_root() -> str:
    # Optional override for admins/users
    env_root = os.getenv("CAPTUS_REPO_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    else:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return REPO_ROOT
    
def create_analysis_latex_table(analysis_list, rows, row_labels, ci_lists=None, unit_change=None, fmt='.6f', cols_labels=None, filename=None):
    """
    Create a LaTeX table from the catalog data using pytextable.
    - cols: list of column names (keys in catalog[v]['occurrences'])
    - rows: list of velocity keys (e.g., ['v17', 'v18'])
    - row_labels: list of labels for the rows
    - ci_lists: list of confidence interval lists
    - cols_labels: list of labels for the columns
    - filename: output .tex file path
    """
    def sci_notation_latex(x, pos=None):
        """Format a number as 2.3 × 10^4 for axis labels (LaTeX style)."""
        if x == 0:
            return "0"
        else:
            exponent = int(np.floor(np.log10(abs(x))))
            coeff = x / 10**exponent
            # Use LaTeX formatting for matplotlib
            if ((coeff - 1.00) < 1e-3):
                return r"10$^{{{}}}$".format(exponent)
            # elif (coeff - 10.00) < 1e-3:
            #     return r"10$^{{{}}}$".format(exponent + 1)
            else:
                return r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)
    if cols_labels is None:
        cols_labels_list = []
        for l in analysis_list:
            catalog = l.results_dictionary
            if cols_labels is None:
                labl = catalog['mC'] / const.M_sun.value
                print(f'Adding column label: {labl:.0e} M_sun')
                cols_labels_list.append(fr'{sci_notation_latex(labl)} M$_{{\odot}}$')
        cols_labels = cols_labels_list
    rows_data = []
    for unit_change, v, vl in zip(unit_change, rows, row_labels):
        row_data = []
        if  unit_change is not None:
            uc = unit_change
        else:
            uc = 1
        print(f'Processing row: {vl} with key: {v}')
        # row_data.append(vl)
        for l in analysis_list:
            catalog = l.results_dictionary
            # if len(v) == 1:
            #     d = sci_notation_latex(catalog[v[0]])
            #     row_data.append(d)
            # elif len(v) == 2:
            #     d = sci_notation_latex(catalog[v[0]][v[1]])
            #     row_data.append(d)
            # elif len(v) == 3:
            #     d = sci_notation_latex(catalog[v[0]][v[1]][v[2]])
            #     row_data.append(d)
            if len(v) == 1:
                d = catalog[v[0]]*uc
            elif len(v) == 2:
                d = catalog[v[0]][v[1]]*uc
            elif len(v) == 3:
                d = catalog[v[0]][v[1]][v[2]]*uc

            if ci_list is not None:
                ci_key = ci_list[rows.index(v)]
                ci_value = catalog['errors'][ci_key]
                row_data.append(f'{d:.2e} ({np.round(ci_value[0], 2):.2e}, {ci_value[1]:.2e})')

        rows_data.append(row_data)
    print(f'Rows data for table: {rows_data}')
    print(f'Column labels for table: {cols_labels_list}')
    table = ptx.tostring(rows_data,fmt=fmt, header=cols_labels)
    return table             
   


def round_uncertainty_and_value(value, error, sig_figs_err=None, sc_notation=False):
    """
    Round uncertainty to significant figures, then round value to the same
    decimal place. Optionally return a shared scientific-notation form.

    Returns
    -------
    If sc_notation=False:
        value_str, error_str

    If sc_notation=True:
        value_str, error_str, exponent
        meaning: (value_str ± error_str) × 10^{exponent}
    """
    value = float(value)
    error = float(error)

    if not np.isfinite(value) or not np.isfinite(error):
        if sc_notation:
            return str(value), str(error), 0
        return str(value), str(error)

    if error == 0:
        if sc_notation:
            exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
            scaled_value = value / 10**exponent
            return f"{scaled_value:.3g}", "0", exponent
        return f"{value:.3g}", "0"

    if sig_figs_err is None:
        sig_figs_err = 1

    # Scientific notation branch
    if sc_notation:
        # Use exponent of the value, so value looks like 2.10 rather than 0.06
        exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else int(np.floor(np.log10(abs(error))))

        scale = 10**exponent
        value_scaled = value / scale
        error_scaled = error / scale

        val_str, err_str = round_uncertainty_and_value(
            value_scaled,
            error_scaled,
            sig_figs_err=sig_figs_err,
            sc_notation=False
        )

        return val_str, err_str, exponent

    # Regular decimal branch
    exponent = np.floor(np.log10(abs(error)))
    decimals = int(-exponent + (sig_figs_err - 1))

    if decimals >= 0:
        err_rounded = round(error, decimals)
        val_rounded = round(value, decimals)

        val_str = f"{val_rounded:.{decimals}f}"
        err_str = f"{err_rounded:.{decimals}f}"
    else:
        factor = 10**(-decimals)
        err_rounded = round(error / factor) * factor
        val_rounded = round(value / factor) * factor

        val_str = f"{val_rounded:.0f}"
        err_str = f"{err_rounded:.0f}"

    return val_str, err_str
   
def create_analysis_pandas_table(analysis_list, rows, row_labels, decimal_places=[2,3], symmetric_errors=False, ci_list=None, qval_list=None, err_list=None, err_sigma=1.96, average_columns=False, skip_columns=None, unit_change=None, sc_notation=None, ci_unit_change=None, cols_labels=None, filename=None):
    """
    Create a LaTeX table from the catalog data using pytextable.
    - cols: list of column names (keys in catalog[v]['occurrences'])
    - rows: list of velocity keys (e.g., ['v17', 'v18'])
    - row_labels: list of labels for the rows
    - cols_labels: list of labels for the columns
    - filename: output .tex file path
    """
    def round_to_sig_figs(x, sig_figs=1):
        if x == 0:
            return 0
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (sig_figs - 1))

    averaged_columns_data = []
    averaged_columns_labels = []
    # if decimal places is none, set it to minimal decimal places for each column
    if decimal_places is None:
        decimal_places = []
        for v in rows:
            max_decimals = 0
            for l in analysis_list:
                catalog = l.results_dictionary
                if len(v) == 1:
                    d = catalog[v[0]]
                elif len(v) == 2:
                    d = catalog[v[0]][v[1]]
                elif len(v) == 3:
                    d = catalog[v[0]][v[1]][v[2]]
                decimals = -int(np.floor(np.log10(abs(d)))) + 1 if d != 0 else 0
                max_decimals = min(max_decimals, decimals)
            decimal_places.append(max_decimals)
    if cols_labels is None:
        cols_labels_list = []
        for k, l in enumerate(analysis_list):
            catalog = l.results_dictionary
            labl = catalog['mC'] / const.M_sun.value
            if average_columns and k in average_columns:
                averaged_columns_labels.append(fr'{labl}')
                if len(averaged_columns_labels) == len(average_columns):
                    cols_labels_list.append(fr'Average of ' + ', '.join(averaged_columns_labels) + ' M$_{{\odot}}$')
                    averaged_columns_labels.clear()
                else:
                    continue
            else:
                cols_labels_list.append(fr'{labl}')
        cols_labels = cols_labels_list
    rows_data = []
    for unit_change, v, vl in zip(unit_change, rows, row_labels):
        row_data = []
        if  unit_change is not None:
            uc = unit_change
        else:
            uc = 1
        # print(f'Processing row: {vl} with key: {v}')
        # row_data.append(vl)
        for j, l in enumerate(analysis_list):
            catalog = l.results_dictionary
            if len(v) == 1:
                d = catalog[v[0]]*uc
            elif len(v) == 2:
                d = catalog[v[0]][v[1]]*uc
            elif len(v) == 3:
                d = catalog[v[0]][v[1]][v[2]]*uc

            
            if ci_list is not None:
                ci_key = ci_list[rows.index(v)]
                ci_uc = ci_unit_change[rows.index(v)]

                if ci_key is not None:
                    ci_value_l = round_to_sig_figs(catalog['errors'][ci_key][0]*ci_uc, 1)
                    ci_value_u = round_to_sig_figs(catalog['errors'][ci_key][1]*ci_uc, 1)
                    d = round_to_sig_figs(d, 1)
                    row_data.append(fr'{d} $_{{-{ci_value_l}}}^{{+{ci_value_u}}}$')
            elif qval_list is not None:
                qval_key = qval_list[rows.index(v)]
                qval_uc = ci_unit_change[rows.index(v)]
                if qval_key is not None:
                    qval_l = round_to_sig_figs(catalog['errors'][qval_key][0]*qval_uc, 1)
                    qval_u = round_to_sig_figs(catalog['errors'][qval_key][2]*qval_uc, 1)
                    row_data.append(f'{d} (q={qval_l}, q={qval_u})')
            elif err_list is not None:
                err_key = err_list[rows.index(v)]
                err_uc = ci_unit_change[rows.index(v)]
                if err_key is not None:
                    # If catalog['errors'][ci_key][0] is the lower CI bound:
                    err_value = catalog['errors'][err_key]*err_uc
                    ci_value_sym = err_value * err_sigma
                    if average_columns and j in average_columns:
                            averaged_columns_data.append((d, ci_value_sym))
                            if len(averaged_columns_data) == len(average_columns):
                                # d_comp = np.sum([x[0]/x[1]**2 for x in averaged_columns_data])
                                # ci_comp = np.sum([1/x[1]**2 for x in averaged_columns_data])
                                # avg_d = d_comp / ci_comp
                                # avg_ci = ci_comp**(-0.5)
                                # avg_d = np.mean([x[0] for x in averaged_columns_data])
                                # avg_ci 
                                # d = avg_d
                                # ci_value_sym = avg_ci
                                # row_data.append(fr'{avg_d:.2f} $\pm$ {avg_ci:.2f}')
                                averaged_columns_data.clear()
                            else:
                                continue
                        
                    if sc_notation is not None and sc_notation[rows.index(v)]:
                        mean_str, err_str, exponent = round_uncertainty_and_value(d, ci_value_sym, sig_figs_err=1, sc_notation=True)
                        row_data.append(fr'$({mean_str} \pm {err_str}) \times 10^{{{exponent}}}$')
                    else:
                        mean_str, err_str = round_uncertainty_and_value(d, ci_value_sym, sig_figs_err=1, sc_notation=False)
                        row_data.append(fr'{mean_str} $\pm$ {err_str}')

            else:
                d = round_to_sig_figs(d, 1)
                row_data.append(d)

        rows_data.append(row_data)
    if skip_columns is not None:
        for i in range(len(rows_data)):
            rows_data[i] = [d for j, d in enumerate(rows_data[i]) if not skip_columns[j]]
        cols_labels_list = [label for j, label in enumerate(cols_labels) if not skip_columns[j]] 
    print(f'Rows data for table: {rows_data}')
    print(f'Column labels for table: {cols_labels_list}')
    table = pd.DataFrame(rows_data, columns=cols_labels_list, index=row_labels)
    return table  

def get_MW_params(r=8.0):
    # Milky Way parameters
    DM_density = calc.dm_density_profile_milkyway(r)
    v_circ = calc.circular_velocity_milkyway(r)
    DM_mass, DM_vel = calc.dark_matter_mass_and_velocity_milkyway(r)
    DM_dispersion = v_circ / np.sqrt(2)  # Assuming isotropic velocity distribution
    dictionary = {
        'DM_density': DM_density,
        'v_circ': v_circ,
        'DM_mass': DM_mass,
        'DM_vel': DM_vel,
        'DM_dispersion': DM_dispersion
    }
    return dictionary


# def get_integrated_Neq_and_params(analysis, r_array, dr, M_star=1.0, fraction=1.0, save=False):
    
#     """Calculate integrated N_eq for a range of radii and return a dictionary of results.
    
#     Parameters
#     ----------
#     analysis : Analysis object or tuple of Analysis objects
#         The analysis or analyses to use for calculating N_eq.
#     r_array : array-like
#         Array of radii at which to calculate N_eq.
#     M_star : float, optional
#         Average stellar mass to use in N_eq calculation (default is 1.0).
#     fraction : float, optional
#         Fraction of stars to consider in N_eq calculation (default is 1.0).
#     Returns
#     -------
#     dict
#         A dictionary containing the integrated N_eq values and corresponding parameters for each radius.
#     """
#     all_results = {}
#     for r in r_array:
#         r_dict = {}
#         a_list = []
#         if isinstance(analysis, tuple):
#             for a1, a2 in zip(analysis[0], analysis[1]):
#                 combined = a1.merged_copy_for_error_analysis(a2)
#                 combined.update_results_dictionary(r)
#                 a_list.append(combined)
#         else:
#             analysis.update_results_dictionary(r)
#             a_list.append(analysis)

#         N_eq_full = []
#         Neq_pbh_bound = []
#         for a in a_list:
#             N_eq_full.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_full'])
#             Neq_pbh_bound.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_bound'])

#         N_stars_at_r = calc.star_number_at_r(r, dr, M_star=M_star)
#         N_alike_systems_at_r = calc.star_number_at_r(r, dr, M_star=M_star, fraction=fraction)

#         N_eq_full_stars = N_eq_full[0] * N_stars_at_r
#         Neq_pbh_bound_stars = Neq_pbh_bound[0] * N_stars_at_r

#         N_eq_full_alike = N_eq_full[0] * N_alike_systems_at_r
#         Neq_pbh_bound_alike = Neq_pbh_bound[0] * N_alike_systems_at_r

#         r_dict['N_eq_full'] = N_eq_full
#         r_dict['N_eq_pbh_bound'] = Neq_pbh_bound
#         r_dict['N_stars_at_r'] = N_stars_at_r
#         r_dict['N_alike_systems_at_r'] = N_alike_systems_at_r
#         r_dict['N_eq_full_stars'] = N_eq_full_stars
#         r_dict['N_eq_pbh_bound_stars'] = Neq_pbh_bound_stars
#         r_dict['N_eq_full_alike'] = N_eq_full_alike
#         r_dict['N_eq_pbh_bound_alike'] = Neq_pbh_bound_alike

#         all_results[r] = r_dict
#         print(f"Calculated N_eq values for r={r} kpc: N_eq_full={N_eq_full}, Neq_pbh_bound={Neq_pbh_bound}, N_stars_at_r={N_stars_at_r}, N_alike_systems_at_r={N_alike_systems_at_r}")
    
#     # Now integrate N_eq over radius
#     integrated_results = {}
#     N_eq_full_stars_integrated = calc.integrate_gauss_legendre(r_array, [all_results[r]['N_eq_full_stars'] for r in r_array])
#     Neq_pbh_bound_stars_integrated = calc.integrate_gauss_legendre(r_array, [all_results[r]['N_eq_pbh_bound_stars'] for r in r_array])
#     N_eq_full_alike_integrated = calc.integrate_gauss_legendre(r_array, [all_results[r]['N_eq_full_alike'] for r in r_array])
#     Neq_pbh_bound_alike_integrated = calc.integrate_gauss_legendre(r_array, [all_results[r]['N_eq_pbh_bound_alike'] for r in r_array])
#     integrated_results['N_eq_full_stars_integrated'] = N_eq_full_stars_integrated
#     integrated_results['Neq_pbh_bound_stars_integrated'] = Neq_pbh_bound_stars_integrated
#     integrated_results['N_eq_full_alike_integrated'] = N_eq_full_alike_integrated
#     integrated_results['Neq_pbh_bound_alike_integrated'] = Neq_pbh_bound_alike_integrated


#     # Save the dictionary as a dataframe 
#     if save:
#         import pickle

#         data = {'all_results': all_results, 'integrated_results': integrated_results}
#         with open('integrated_neq_results.pkl', 'wb') as f:
#             pickle.dump(data, f)

#     return all_results, integrated_results
def get_integrated_Neq_and_params(analysis, r_array, dr, M_star=1.0, fraction=1.0, save=False):
    
    """Calculate integrated N_eq for a range of radii and return a dictionary of results.
    
    Parameters
    ----------
    analysis : Analysis object or tuple of Analysis objects
        The analysis or analyses to use for calculating N_eq.
    r_array : array-like
        Array of radii at which to calculate N_eq.
    M_star : float, optional
        Average stellar mass to use in N_eq calculation (default is 1.0).
    fraction : float, optional
        Fraction of stars to consider in N_eq calculation (default is 1.0).
    Returns
    -------
    dict
        A dictionary containing the integrated N_eq values and corresponding parameters for each radius.
    """
    all_results = {}
    for i, r in enumerate(r_array):
        r_dict = {}
        a_list = []
        if isinstance(analysis, tuple):
            for a1, a2 in zip(analysis[0], analysis[1]):
                combined = a1.merged_copy_for_error_analysis(a2)
                combined.update_results_dictionary(r)
                a_list.append(combined)
        else:
            # Create fresh list for each radius to avoid accumulation
            a_list = list(analysis) if not isinstance(analysis, list) else analysis.copy()
            for a in a_list:
                a.update_results_dictionary(r)

        N_eq_full = []
        Neq_pbh_bound = []
        for a in a_list:
            N_eq_full.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_full'])
            Neq_pbh_bound.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_bound'])

        # torus_volume_at_r = 2 * np.pi * r * np.pi * dr**2  # Volume of the cylindrical shell at radius r
        N_stars_at_r = calc.star_number_at_r(r, dr[i], M_star=M_star)
        N_alike_systems_at_r = calc.star_number_at_r(r, dr[i], M_star=M_star, fraction=fraction)

        N_eq_full_stars = np.asarray(N_eq_full) * (N_stars_at_r/dr[i])  # Convert to number density by dividing by dr
        Neq_pbh_bound_stars = np.asarray(Neq_pbh_bound) * (N_stars_at_r/dr[i])

        N_eq_full_alike = np.asarray(N_eq_full) * (N_alike_systems_at_r/dr[i])
        Neq_pbh_bound_alike = np.asarray(Neq_pbh_bound) * (N_alike_systems_at_r/dr[i])

        r_dict['N_eq_full'] = N_eq_full
        r_dict['N_eq_pbh_bound'] = Neq_pbh_bound
        r_dict['N_stars_at_r'] = N_stars_at_r
        r_dict['N_alike_systems_at_r'] = N_alike_systems_at_r
        r_dict['N_eq_full_stars'] = N_eq_full_stars
        r_dict['N_eq_pbh_bound_stars'] = Neq_pbh_bound_stars
        r_dict['N_eq_full_alike'] = N_eq_full_alike
        r_dict['N_eq_pbh_bound_alike'] = Neq_pbh_bound_alike

        all_results[r] = r_dict
    
    integrated_results = {}
    # Now integrate N_eq over radius
    # Instead of looping through all_results multiple times:
    # Pre-extract the arrays you need
    for i in range(len(a_list)):
        integrated_results[i] = {}
        N_eq_full_stars_array = np.array([all_results[r]['N_eq_full_stars'][i] for r in r_array])
        Neq_pbh_bound_stars_array = np.array([all_results[r]['N_eq_pbh_bound_stars'][i] for r in r_array])
        N_eq_full_alike_array = np.array([all_results[r]['N_eq_full_alike'][i] for r in r_array])
        Neq_pbh_bound_alike_array = np.array([all_results[r]['N_eq_pbh_bound_alike'][i] for r in r_array])

        # Then integrate once
        N_eq_full_stars_integrated = calc.integrate_gauss_legendre(r_array, N_eq_full_stars_array)
        Neq_pbh_bound_stars_integrated = calc.integrate_gauss_legendre(r_array, Neq_pbh_bound_stars_array)
        N_eq_full_alike_integrated = calc.integrate_gauss_legendre(r_array, N_eq_full_alike_array)
        Neq_pbh_bound_alike_integrated = calc.integrate_gauss_legendre(r_array, Neq_pbh_bound_alike_array)
        # N_eq_full_stars_integrated = np.sum(N_eq_full_stars_array)
        # Neq_pbh_bound_stars_integrated = np.sum(Neq_pbh_bound_stars_array)
        # N_eq_full_alike_integrated = np.sum(N_eq_full_alike_array)
        # Neq_pbh_bound_alike_integrated = np.sum(Neq_pbh_bound_alike_array)

        integrated_results[i]['N_eq_full_stars_integrated'] = N_eq_full_stars_integrated
        integrated_results[i]['Neq_pbh_bound_stars_integrated'] = Neq_pbh_bound_stars_integrated
        integrated_results[i]['N_eq_full_alike_integrated'] = N_eq_full_alike_integrated
        integrated_results[i]['Neq_pbh_bound_alike_integrated'] = Neq_pbh_bound_alike_integrated


    # Save the dictionary as a dataframe 
    if save:
        import pickle

        data = {'all_results': all_results, 'integrated_results': integrated_results}
        with open('integrated_neq_results.pkl', 'wb') as f:
            pickle.dump(data, f)

    return all_results, integrated_results

def get_shell_sum_Neq_and_params(analysis, r_edges, M_star=1.0, fraction=1.0, save=False):
    
    """Calculate integrated N_eq for a range of radii and return a dictionary of results.
    
    Parameters
    ----------
    analysis : Analysis object or tuple of Analysis objects
        The analysis or analyses to use for calculating N_eq.
    r_array : array-like
        Array of radii at which to calculate N_eq.
    M_star : float, optional
        Average stellar mass to use in N_eq calculation (default is 1.0).
    fraction : float, optional
        Fraction of stars to consider in N_eq calculation (default is 1.0).
    Returns
    -------
    dict
        A dictionary containing the integrated N_eq values and corresponding parameters for each radius.
    """
    all_results = {}
    r_edges = np.asarray(r_edges)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr_array = np.diff(r_edges)

    for i, r_mid in enumerate(r_centers):
        r_dict = {}
        r_in = r_edges[i]
        r_out = r_edges[i + 1]
        dr_i = dr_array[i]

        
        if isinstance(analysis, tuple):
            for a1, a2 in zip(analysis[0], analysis[1]):
                combined = a1.merged_copy_for_error_analysis(a2)
                combined.update_results_dictionary(r_mid)
                a_list.append(combined)
        else:
            # Create fresh list for each radius to avoid accumulation
            a_list = list(analysis) if not isinstance(analysis, list) else analysis.copy()
            for a in a_list:
                a.update_results_dictionary(r_mid)

        N_eq_full = []
        Neq_pbh_bound = []
        n_pbh_density = []
        for a in a_list:
            N_eq_full.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_full'])
            Neq_pbh_bound.append(a.results_dictionary['total_pbh_neq_n']['Neq_pbh_f_bound'])
            n_pbh_density.append(a.results_dictionary['total_pbh_neq_n']['number_density_in_range_kpc3'])

        # torus_volume_at_r = 2 * np.pi * r * np.pi * dr**2  # Volume of the cylindrical shell at radius r
        N_stars_at_r = calc.star_number_at_r(r_mid, dr_i, M_star=M_star)
        N_alike_systems_at_r = calc.star_number_at_r(r_mid, dr_i, M_star=M_star, fraction=fraction)

        N_eq_full_stars = np.asarray(N_eq_full) * (N_stars_at_r)  # Convert to number density by dividing by dr
        Neq_pbh_bound_stars = np.asarray(Neq_pbh_bound) * (N_stars_at_r)

        N_eq_full_alike = np.asarray(N_eq_full) * (N_alike_systems_at_r)
        Neq_pbh_bound_alike = np.asarray(Neq_pbh_bound) * (N_alike_systems_at_r)

        r_dict['n_pbh_density'] = n_pbh_density
        r_dict['N_eq_full'] = N_eq_full
        r_dict['N_eq_pbh_bound'] = Neq_pbh_bound
        r_dict['N_stars_at_r'] = N_stars_at_r
        r_dict['N_alike_systems_at_r'] = N_alike_systems_at_r
        r_dict['N_eq_full_stars'] = N_eq_full_stars
        r_dict['N_eq_pbh_bound_stars'] = Neq_pbh_bound_stars
        r_dict['N_eq_full_alike'] = N_eq_full_alike
        r_dict['N_eq_pbh_bound_alike'] = Neq_pbh_bound_alike

        all_results[r_mid] = r_dict
    
    integrated_results = {}
    # Now integrate N_eq over radius
    # Instead of looping through all_results multiple times:
    # Pre-extract the arrays you need
    for i in range(len(a_list)):
        integrated_results[i] = {}
        N_stars_total_array = np.array([all_results[r]['N_stars_at_r'] for r in r_centers])
        N_eq_full_stars_array = np.array([all_results[r]['N_eq_full_stars'][i] for r in r_centers])
        Neq_pbh_bound_stars_array = np.array([all_results[r]['N_eq_pbh_bound_stars'][i] for r in r_centers])
        N_eq_full_alike_array = np.array([all_results[r]['N_eq_full_alike'][i] for r in r_centers])
        Neq_pbh_bound_alike_array = np.array([all_results[r]['N_eq_pbh_bound_alike'][i] for r in r_centers])

        # Then integrate once
        # N_eq_full_stars_integrated = calc.integrate_gauss_legendre(r_array, N_eq_full_stars_array)
        # Neq_pbh_bound_stars_integrated = calc.integrate_gauss_legendre(r_array, Neq_pbh_bound_stars_array)
        # N_eq_full_alike_integrated = calc.integrate_gauss_legendre(r_array, N_eq_full_alike_array)
        # Neq_pbh_bound_alike_integrated = calc.integrate_gauss_legendre(r_array, Neq_pbh_bound_alike_array)
        N_eq_full_stars_integrated = np.sum(N_eq_full_stars_array)
        Neq_pbh_bound_stars_integrated = np.sum(Neq_pbh_bound_stars_array)
        N_eq_full_alike_integrated = np.sum(N_eq_full_alike_array)
        Neq_pbh_bound_alike_integrated = np.sum(Neq_pbh_bound_alike_array)
        N_stars_total_integrated = np.sum(N_stars_total_array)

        integrated_results[i]['N_eq_full_stars_integrated'] = N_eq_full_stars_integrated
        integrated_results[i]['Neq_pbh_bound_stars_integrated'] = Neq_pbh_bound_stars_integrated
        integrated_results[i]['N_eq_full_alike_integrated'] = N_eq_full_alike_integrated
        integrated_results[i]['Neq_pbh_bound_alike_integrated'] = Neq_pbh_bound_alike_integrated
        integrated_results[i]['N_stars_total_integrated'] =  N_stars_total_integrated


    # Save the dictionary as a dataframe 
    if save:
        import pickle

        data = {'all_results': all_results, 'integrated_results': integrated_results}
        with open('integrated_neq_results.pkl', 'wb') as f:
            pickle.dump(data, f)

    return all_results, integrated_results
