import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from captus.utils.plotting_utils import sci_notation_latex
import os
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

save_directory = os.path.join(os.path.dirname(__file__), '../../results/plots/GW_Capture/')
os.makedirs(save_directory, exist_ok=True)
mpl.rc('font',family='DeJavu Serif')
font = FontProperties(family='DeJavu Serif')
plt.rcParams['savefig.dpi'] = 600  # Set the resolution for saved figures
# Set the number of lines to plot and the colormap
n_lines = 10
cmap = mpl.colormaps['plasma']
# Take colors at regular intervals spanning the colormap.
colors = cmap(np.linspace(0, 1, n_lines))

def plot_capture_cross_section(m_pbh_values_in_Msun, v_inf_values_kms, cross_section_grid, star, r):
    """
    Plot the capture cross-section grid for a given star and radius.
    """
    vmin = float(cross_section_grid[cross_section_grid > 0].min())
    vmax = float(cross_section_grid.max())

    plt.figure(figsize=(3.5, 3))
    plt.fill_betweenx(np.array(v_inf_values_kms), 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, label='Prohibited Region')
    plt.fill_betweenx(np.array(v_inf_values_kms)[np.array(v_inf_values_kms) < 550], 
                    1.0, 
                    np.max(m_pbh_values_in_Msun),
                    color='cyan', alpha=0.4, label='Prohibited Region')

    im = plt.contourf(m_pbh_values_in_Msun, v_inf_values_kms, cross_section_grid,
                      cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), levels=np.logspace(np.log10(vmin), np.log10(vmax), 7))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'M$_{P}$ [M$_{\odot}$]')
    plt.ylabel(r'v$_{\infty}$ [km s$^{-1}$]')
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\sigma_{cap}^{GW}$ [AU$^2$]')
    cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar.get_ticks()])
    
    plt.title(f'Capture Cross-Section at r={r} kpc')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_directory + f'{star}_crosssection_{r}kpc.png', bbox_inches='tight')

def plot_n_eqs(m_pbh_values_in_Msun, v_inf_values_kms, n_eqs_grid, star, r):
    """
    Plot the N_eq grid for a given star and radius.
    """
    n_eqs_grid = np.ma.masked_invalid(n_eqs_grid)
    n_eqs_grid = np.ma.masked_less_equal(n_eqs_grid, 0.0)
    vmin = float(n_eqs_grid.min())
    vmax = float(n_eqs_grid.max())

    v_above_550 = np.array(v_inf_values_kms)[np.array(v_inf_values_kms) >= 550]

    plt.figure(figsize=(3.5, 3))
    plt.fill_betweenx(np.array(v_inf_values_kms), 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, label='Prohibited Region')
    # plt.fill_betweenx(np.array(v_inf_values_kms)[np.array(v_inf_values_kms) < 550], 
    #                 1.0, 
    #                 np.max(m_pbh_values_in_Msun),
    #                 color='cyan', alpha=0.4, label='Prohibited Region')

    im = plt.contourf(m_pbh_values_in_Msun, v_inf_values_kms, n_eqs_grid, 
                      cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), 
                      levels=np.logspace(np.log10(vmin), np.log10(vmax), 7))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'M$_{P}$ [M$_{\odot}$]')
    plt.ylabel(r'v$_{\infty}$ [km s$^{-1}$]')
    cbar = plt.colorbar(im)
    cbar.set_label(r'$d$N$_{eq}/d$v$_{\infty}$ ')
    cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar.get_ticks()])
    
    plt.title(f'N_eq at r={r} kpc')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_directory + f'{star}_neq_{r}kpc.png', bbox_inches='tight')

def plot_coalescence_time(m_pbh_values_in_Msun, v_inf_values_kms, coalescence_grid, star, r):
    """
    Plot the coalescence time grid for a given star and radius.
    """
    coal_grid = np.where(coalescence_grid == 0, np.nan, coalescence_grid)
    coal_plot = np.ma.masked_invalid(coal_grid)
    coal_plot = np.ma.masked_less_equal(coal_plot, 0.0)
    vmin = float(coal_plot[coal_plot > 0].min())
    vmax = float(coal_plot.max())

    plt.figure(figsize=(3.5, 3))
    plt.fill_betweenx(np.array(v_inf_values_kms), 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, label='Prohibited Region')
    plt.fill_betweenx(np.array(v_inf_values_kms)[np.array(v_inf_values_kms) < 550], 
                    1.0, 
                    np.max(m_pbh_values_in_Msun),
                    color='cyan', alpha=0.4, label='Prohibited Region')

    im = plt.contourf(
        m_pbh_values_in_Msun,
        v_inf_values_kms,
        coal_plot,
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 7),
        cmap="plasma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'M$_{P}$ [M$_{\odot}$]')
    plt.ylabel(r'v$_{\infty}$ [km s$^{-1}$]')
    cbar = plt.colorbar(im)
    cbar.set_label(r'Coalescence time [yr]')
    cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar.get_ticks()])
    
    plt.title(f'Coalescence Time at r={r} kpc')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_directory + f'{star}_coalescence_time_{r}kpc.png', bbox_inches='tight')

def plot_all_grids(m_pbh_values_in_Msun, v_inf_values_kms, cross_section_grid, n_eqs_grid, coalescence_grid, star, r):
    """
    Plot all the grids for a given star and radius.
    """

    coal_grid = np.where(cross_section_grid == 0, np.nan, coalescence_grid)
    coal_plot = np.ma.masked_invalid(coal_grid)
    coal_plot = np.ma.masked_less_equal(coal_plot, 0.0)
    # Create 2x1 subplot with shared x-axis
    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(3.5, 7), sharex=True)
    vmin = float(cross_section_grid[cross_section_grid > 0].min())
    vmax = float(cross_section_grid.max())
    # ===== TOP PLOT: Cross-section grid =====
    v_below_550 = np.array(v_inf_values_kms)[np.array(v_inf_values_kms) < 550]

    ax1.fill_betweenx(np.array(v_inf_values_kms), 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, zorder=0)
    ax1.fill_betweenx(v_below_550, 
                    1.0, 
                    np.max(m_pbh_values_in_Msun),
                    color='cyan', alpha=0.4, zorder=0)

    im1 = ax1.contourf(m_pbh_values_in_Msun, v_inf_values_kms, cross_section_grid,
                    cmap='plasma', norm=plt.cm.colors.LogNorm(), levels=np.logspace(np.log10(vmin), np.log10(vmax), 7), zorder=1)
    ax1.set_yscale('log')
    # ax1.set_ylabel(r'v$_{\infty}$ [km s$^{-1}$]')
    ax1.text(0.03, 0.7, r'Prohibited', fontsize=10, ha='left', va='center', rotation=55,
            transform=ax1.transAxes)
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=1, aspect=20)
    cbar1.set_label(r'$\sigma_{cap}^{GW}$ [AU$^2$]')
    cbar1.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar1.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar1.get_ticks()])

    # ===== BOTTOM PLOT: N_eq grid =====
    n_eqs_grid = np.ma.masked_invalid(n_eqs_grid)
    n_eqs_grid = np.ma.masked_less_equal(n_eqs_grid, 0.0)
    vmin = float(n_eqs_grid.min())
    vmax = float(n_eqs_grid.max())

    v_above_550 = np.array(v_inf_values_kms)[np.array(v_inf_values_kms) >= 550]

    ax2.fill_betweenx(v_below_550, 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, zorder=0)
    ax2.fill_betweenx(v_below_550, 
                    1.0, 
                    np.max(m_pbh_values_in_Msun),
                    color='cyan', alpha=0.4, zorder=0)
    ax2.fill_between(m_pbh_values_in_Msun, v_above_550[0], np.max(v_inf_values_kms), 
                    alpha=0.4, color='gray', label='Prohibited Region', zorder=0)

    im2 = ax2.contourf(m_pbh_values_in_Msun, v_inf_values_kms, n_eqs_grid, 
                    cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), 
                    levels=np.logspace(np.log10(vmin), np.log10(vmax), 7), zorder=1)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'M$_{P}$ [M$_{\odot}$]')
    # ax2.set_ylabel(r'v$_{\infty}$ [km s$^{-1}$]')

    ax2.hlines(550, np.min(m_pbh_values_in_Msun), np.max(m_pbh_values_in_Msun), 
            color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(0.03, 0.7, r'Prohibited', fontsize=10, ha='left', va='center', rotation=55,
            transform=ax2.transAxes)

    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=1, aspect=20)
    cbar2.set_label(r'$d$N$_{eq}/d$v$_{\infty}$ ')
    cbar2.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar2.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar2.get_ticks()])

    # Set log scale on top plot as well
    ax1.set_xscale('log')

    vmin = float(coal_plot[coal_plot > 0].min())
    vmax = float(coal_plot.max())
    ax3.fill_betweenx(np.array(v_inf_values_kms), 
                    np.min(m_pbh_values_in_Msun), 
                    1.0,
                    color='gray', alpha=0.4, zorder=0)
    im3 = ax3.contourf(
        m_pbh_values_in_Msun,
        v_inf_values_kms,
        coal_plot,
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 7),
        cmap="plasma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax3.text(0.03, 0.7, r'Prohibited', fontsize=10, ha='left', va='center', rotation=55,
            transform=ax3.transAxes)
    ax3.set_ylabel(r'v$_{\infty}$ [km s$^{-1}$]')

    cbar3 = plt.colorbar(im3, ax=ax3, pad=0.02, shrink=1, aspect=20)
    cbar3.set_label(r'Coalescence time [yr]')
    cbar3.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), 7))
    cbar3.set_ticklabels([f"{sci_notation_latex(x)}" for x in cbar3.get_ticks()])

    ax3.set_yscale('log')
    ax3.set_xscale('log')


    fig.subplots_adjust(hspace=0.08)
    plt.savefig(save_directory + f'{star}_crosssection_coal_dneq_{r}kpc.png', bbox_inches='tight')