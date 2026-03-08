import os
import pytextable as ptx
import numpy as np
import astropy.constants as const

def _resolve_repo_root() -> str:
    # Optional override for admins/users
    env_root = os.getenv("CAPTUS_REPO_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    else:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return REPO_ROOT
    
def create_analysis_latex_table(analysis_list, rows, row_labels, unit_change=None, fmt='.6f', cols_labels=None, filename=None):
    """
    Create a LaTeX table from the catalog data using pytextable.
    - cols: list of column names (keys in catalog[v]['occurrences'])
    - rows: list of velocity keys (e.g., ['v17', 'v18'])
    - row_labels: list of labels for the rows
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
                row_data.append(d)
            elif len(v) == 2:
                d = catalog[v[0]][v[1]]*uc
                row_data.append(d)
            elif len(v) == 3:
                d = catalog[v[0]][v[1]][v[2]]*uc
                row_data.append(d)

        rows_data.append(row_data)
    print(f'Rows data for table: {rows_data}')
    print(f'Column labels for table: {cols_labels_list}')
    table = ptx.tostring(rows_data,fmt=fmt, header=cols_labels)
    return table             
   
