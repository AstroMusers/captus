import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from itertools import groupby
from operator import itemgetter
import os
from collections.abc import Iterable
import captus.utils.plotting_utils as plu
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.ticker as mt
from astropy import units as u
from astropy import constants as const
from scipy.stats import gaussian_kde
import corner as cr
import pandas as pd
import seaborn as sns
import tdpy as td

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
class Plots:
    def __init__(self, name, analysis, analysis_dictionaries=None, **kwargs):

        self.name = name

        plots_dir = os.path.join(REPO_ROOT, f'plots/{self.name}/Plots')

        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        self.plots_dir = plots_dir

        # Accept single analysis or iterable of analyses
        if isinstance(analysis, Iterable) and not isinstance(analysis, (str, bytes)):
            self.analysis_list = list(analysis)
        else:
            self.analysis_list = [analysis]
        
        if analysis_dictionaries is not None:
            self.analysis_dicts = analysis_dictionaries
        else:
            self.analysis_dicts = self._get_analysis_dictionaries()
            
        self.time_series_dicts = {}
        self.zero_capture_excluded = kwargs.get('exclude_zero_capture', True)

    def _get_analysis_dictionaries(self):
        """Retrieve combined dictionaries from all analyses."""
        analysis_dicts = {}
        for analysis in self.analysis_list:
            name = analysis.get_name()
            combined_dict = analysis.get_combined_dictionary()
            analysis_dicts[name] = combined_dict
        return analysis_dicts

    def add_analysis(self, analysis):
        """Add analysis object to the manager."""
        if isinstance(analysis, Iterable) and not isinstance(analysis, (str, bytes)):
            self.analysis_list.extend(list(analysis))
        else:
            self.analysis_list.append(analysis)
        self.analysis_dicts = self._get_analysis_dictionaries()

    def list_analyses(self):
        """Return current analysis objects."""
        return [a.name for a in self.analysis_list]
    
    def get_analysis(self, name):
        """Retrieve analysis object analysis object by name."""
        for analysis in self.analysis_list:
            if analysis.get_name() == name:
                return analysis
        return None

    
    
    # def _create_dataframes(self, analysis):
    #     """Convert analysis results to pandas DataFrames for easier plotting."""

    #     data_frames = {}
    #     for v_key, results in analysis.get_combined_dictionary().items():
    #         mc_data = results["mc"]["data"] if results["mc"] is not None else None
    #         if mc_data is not None:
    #             df_mc = pd.from_dict(mc_data, orient='index')
    #         orbsim_data = results["rebound"]["data"] if results["rebound"] is not None else None
    #         if orbsim_data is not None:
    #             df_orbsim = pd.from_dict(orbsim_data, orient='index')
    #         data_frames[v_key] = df_mc.join(df_orbsim, how='outer', lsuffix='_mc', rsuffix='_orbsim')
        
    #     return data_frames

    def _figure(self, nrows=1, ncols=1, sharex=False, figsize=(3.5, 2.5)):
        """
        Create a figure with common style.
        """
        plt.rcParams.update({'font.family': 'serif'})
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, figsize=figsize)
        return fig, axes

    def plot_metric_wrt_v(self, metric_list, metric_ylabel, analysis_name, metric_masks=None, metric_labels=None, fig_size=None, save=True):
        """
        Expect each analysis to expose:
          - analysis.v_vals_kms: 1D array of v∞ [km/s]
          - analysis.metrics[metric_name]: 1D array aligned with v_vals_kms
        """
        fig, ax = self._figure(figsize=(3.5, 2.5))
        cmap = plt.get_cmap('plasma', len(metric_list) + 1)
        analysis_dict = self.analysis_dicts.get(analysis_name, None)
        

        if analysis_dict is None:
            print(f"Analysis '{analysis_name}' not found.")
            return fig
        
        for metric_name in metric_list:
            metric_array = []
            v_array = []
            for v in analysis_dict.keys():
                if 'V' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                    continue
                v_inf = mc['v_inf']

                metric = self._get_metric_from_sources(analysis_dict, v, metric_name)

                if metric is None:
                    print(f"Metric '{metric_name}' not found for v={v}.")
                    continue

                if isinstance(metric, float) or isinstance(metric, int):
                    metric_array.append(metric)

                elif isinstance(metric[0], (list, np.ndarray)):
                    metric_array.append(float(metric[0]))
                else:
                    print(f"Unrecognized metric format for '{metric_name}' at v={v}.")
                    continue

                
                v_array.append(v_inf / 1e3)



            v = np.asarray(v_array)
            y = np.asarray(metric_array)
            order = np.argsort(v)

            metric_label = None
            if metric_labels:
                metric_label = metric_labels[metric_list.index(metric_name)]

            ax.plot(v[order], y[order], marker='o', linestyle='--', alpha=0.9, color=cmap(metric_list.index(metric_name)), label=metric_label, markersize=2, linewidth=1)

        ax.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        ax.set_ylabel(metric_ylabel)
        ax.legend(frameon=False)
        figname = f"{analysis_name}_{metric_name}_vs_v.png"

        if save:
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig

    def plot_metric_wrt_v_twinaxis(self, metric_lists, metric_ylabels, analysis_name, metric_masks=None, metric_labels=None, fig_size=None, save=True):
        """
        Plot two metrics on twin y-axes against v∞.
        Expect each analysis to expose:
          - analysis.v_vals_kms: 1D array of v∞ [km/s]
          - analysis.metrics[metric_name]: 1D array aligned with v_vals_kms
        """
        fig, ax1 = self._figure(figsize=(3.5, 2.5))
        ax2 = ax1.twinx()
        cmap = plt.get_cmap('plasma', len(metric_lists[0]) + 1)
        analysis_dict = self.analysis_dicts.get(analysis_name, None)
        
        if analysis_dict is None:
            print(f"Analysis '{analysis_name}' not found.")
            return fig
        

        for i, metric_list in enumerate(metric_lists):
            for metric_name in metric_list:
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    rb = analysis_dict[v]['rebound']
                    mc = analysis_dict[v]['mc']
                    oc = analysis_dict[v]['occurrences']
                    v_inf = mc['v_inf']
                    
                    try:
                        metric = rb[metric_name] if metric_name in rb else (mc[metric_name] if metric_name in mc else oc[metric_name])
                    except Exception as e:
                        print(f"Error retrieving metric '{metric_name}' for v={v}: {e}")
                        continue

                    if isinstance(metric, (list, np.ndarray)):
                        metric = np.mean(metric)  # or other summary statistic

                    metric_array.append(metric)
                    v_array.append(v_inf / 1e3)  # convert to km/s


                v = np.asarray(v_array)
                y = np.asarray(metric_array)
                order = np.argsort(v)

                metric_label = None
                
                if i == 0:
                    if metric_labels:
                        metric_label = metric_labels[i][metric_list.index(metric_name)]
                        key0 = plu.latex_label_key(metric_ylabels[0])  # -> "capture"
                        key1 = plu.latex_label_key(metric_ylabels[1])  # -> "termination"
                        metricsname = f"{key0}_{key1}"
                    if isinstance(metric, (list, np.ndarray)):
                        figname = f"{analysis_name}_mean_{metric_name}_vs_v.png"
                    else:
                        figname = f"{analysis_name}_{metric_name}_vs_v.png"

                    ax1.plot(v[order], y[order], marker='o', linestyle='dotted', color=cmap(metric_list.index(metric_name)), label=metric_label, markersize=3, linewidth=1)
                else:
                    ax2.plot(v[order], y[order], marker='s', linestyle='dashed', color=cmap(metric_list.index(metric_name)), label=metric_label, markersize=3, linewidth=1)

        ax1.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        ax1.set_ylabel(metric_ylabels[0])
        ax2.set_ylabel(metric_ylabels[1])
        ax1.legend(frameon=False, loc='upper center', ncol=2)
        # ax2.legend(frameon=False, loc='upper right')


        if save:
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig

    def plot_metric_array_wrt_v(self, metric_list, metric_ylabel, analysis_name, metric_masks=None, metric_labels=None, fig_size=None, save=True):
        """
        Expect each analysis to expose:
          - analysis.v_vals_kms: 1D array of v∞ [km/s]
          - analysis.metrics[metric_name]: 1D array aligned with v_vals_kms
        """
        fig, ax = self._figure(figsize=(7, 2.5))
        cmap = plt.get_cmap('plasma', 7)
        analysis_dict = self.analysis_dicts.get(analysis_name, None)
        analysis = self.get_analysis(analysis_name)

        if analysis_dict is None:
            print(f"Analysis '{analysis_name}' not found.")
            return fig

        for i, metric_name in enumerate(metric_list):
            datasets = []
            v_array = []
            metric_mask = None
            sorted_keys = sorted(analysis_dict.keys(), key=lambda x: analysis_dict[x]['mc']['v_inf'] if 'mc' in analysis_dict[x] and analysis_dict[x]['mc'] is not None else float('inf'))
            for v in sorted_keys:
                if 'V' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                    continue
                v_inf = mc['v_inf']

                metric = self._get_metric_from_sources(analysis_dict, v, metric_name)

                if metric_masks is not None:
                    metric_mask = metric_masks[i]

                if isinstance(metric, (list, tuple)):
                    # flatten valid arrays to 1D lists; drop empties
                    if metric_mask is not None:
                        mask = self.resolve_mask(analysis, metric_mask, v)
                        mask = np.asarray(mask)
                        print(f"Applying mask for metric '{metric_name}' at v={v}: with sum {np.sum(mask)}")
                        metric = [m for j, m in enumerate(metric) if mask[j]]

                    data_v = [float(np.mean(m)) for m in metric if np.size(m) > 0 and float(np.mean(m)) > 0]
                    # if we have multiple arrays per v, concatenate them
                    if len(data_v) == 0:
                        continue
    

                    datasets.append(data_v)
                    v_array.append(v_inf / 1e3)
                else:
                    # Not analysis_dict array-of-arrays; skip for violin
                    continue

            if len(datasets) == 0:
                print(f"No array data for '{metric_name}' to plot as violins.")
                continue

            # sort positions and datasets together
            v_array = np.asarray(v_array)
            order = np.argsort(v_array)
            pos = v_array[order]
            data_sorted = [datasets[i] for i in order]

            vp = ax.violinplot(
                dataset=data_sorted,
                positions=pos,
                showmeans=True,
                showmedians=False,
                showextrema=True,
                widths=30
            )
            # optional styling
            for body in vp['bodies']:
                body.set_zorder(0)
                body.set_alpha(0.3)
                body.set_facecolor(cmap(4 + 8*i))
                # body.set_edgecolor(cmap(2))
                # body.set_hatch('xxx')
                # body.set_hatch_linewidth(0.3)
            for partname in ('cbars','cmins','cmaxes', 'cmeans'):
                vp[partname].set_zorder(1)
                vp[partname].set_alpha(0.8)
                vp[partname].set_edgecolor(cmap(4 + 8*i))
                vp[partname].set_linewidth(1.5)

            vp['cmeans'].set_zorder(2)
            vp['cmeans'].set_alpha(1)

        ax.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        ax.set_ylabel(metric_ylabel)
        ax.set_yscale('log')
        ax.legend(frameon=False)
        figname = f"{analysis_name}_mean_{metric_name}_vs_v.png"
        if save:
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig, data_sorted

    def plot_metric_array_wrt_v_twinaxis(self, metric_lists, metric_ylabels, analysis_name, metric_masks=None, metric_labels=None, fig_size=None, save=True):
        """
        Plot two metrics on twin y-axes against v∞.
        Expect each analysis to expose:
          - analysis.v_vals_kms: 1D array of v∞ [km/s]
          - analysis.metrics[metric_name]: 1D array aligned with v_vals_kms
        """
        fig, ax1 = self._figure(figsize=(7, 2.5))
        ax2 = ax1.twinx()
        cmap = plt.get_cmap('plasma', 15)
        analysis_dict = self.analysis_dicts.get(analysis_name, None)
        analysis = self.get_analysis(analysis_name)
        if analysis_dict is None:
            print(f"Analysis '{analysis_name}' not found.")
            return fig
        

        for i, metric_list in enumerate(metric_lists):
            for metric_name in metric_list:
                datasets = []
                v_array = []
                mask = None
                sorted_keys = sorted(analysis_dict.keys(), key=lambda x: analysis_dict[x]['mc']['v_inf'] if 'mc' in analysis_dict[x] and analysis_dict[x]['mc'] is not None else float('inf'))
                for v in sorted_keys:
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    if metric_masks is not None:
                        metric_mask = metric_masks[i][metric_list.index(metric_name)] if metric_list.index(metric_name) < len(metric_masks[i]) else None
                        mask = self.resolve_mask(analysis, metric_mask, v)

                    if isinstance(metric, (list, tuple)):
                        # flatten valid arrays to 1D lists; drop empties
                        if mask is not None:
                            metric = metric[mask]
                            v_inf = v_inf[mask]
                        data_v = [float(np.mean(m)) for m in metric if np.size(m) > 0 and float(np.mean(m)) > 0]
                        # if we have multiple arrays per v, concatenate them
                        if len(data_v) == 0:
                            continue
        

                        datasets.append(data_v)
                        v_array.append(v_inf / 1e3)
                    else:
                        # Not analysis_dict array-of-arrays; skip for violin
                        continue

                if len(datasets) == 0:
                    print(f"No array data for '{metric_name}' to plot as violins.")
                    continue

                # sort positions and datasets together
                v_array = np.asarray(v_array)
                order = np.argsort(v_array)
                pos = v_array[order]
                data_sorted = [datasets[i] for i in order]

                if i == 0:
                    axobj = ax1
                    side = 'low'
                else:
                    axobj = ax2
                    side = 'high'


                vp = axobj.violinplot(
                    dataset=data_sorted,
                    positions=pos,
                    showmeans=True,
                    showmedians=False,
                    showextrema=True,
                    widths=30,
                    side=side
                )
                # optional styling
                for body in vp['bodies']:
                    body.set_zorder(0)
                    body.set_alpha(0.3)
                    # if i == 0:
                    #     body.set_facecolor('none')
                    #     body.set_edgecolor(cmap(3+8*i))
                    #     body.set_alpha(0.5)
                    # else:
                    body.set_facecolor(cmap(3+8*i))
                    body.set_alpha(0.3)
                    # body.set_edgecolor(cmap(2))
                    # body.set_hatch('xxx')
                    # body.set_hatch_linewidth(0.3)
                for partname in ('cbars','cmins','cmaxes', 'cmeans'):
                    vp[partname].set_zorder(1)
                    vp[partname].set_alpha(0.8)
                    vp[partname].set_edgecolor(cmap(3+8*i))
                    vp[partname].set_linewidth(1.5)

                vp['cmeans'].set_zorder(2)
                vp['cmeans'].set_alpha(1)
                # vp['cmedians'].set_linestyle('dotted')
                # vp['cmedians'].set_zorder(2)



        metricsname = ""
        for key in metric_ylabels:
            key_fmt = plu.latex_label_key(key)
            metricsname += f"{key_fmt}_"
        metricsname = metricsname.rstrip("_")
        figname = f"{analysis_name}_mean_{metricsname}_vs_v.png"
        ax1.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        ax1.set_ylabel(metric_ylabels[0])
        ax2.set_ylabel(metric_ylabels[1])
        ax1.legend(frameon=False, loc='upper left', ncol=2)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        # ax2.legend(frameon=False, loc='upper right')


        if save:
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig

    

    def multiple_analysis_metric_wrt_v_twinaxis(self, metric_lists, metric_ylabels, metric_labels=None, ticks=[None, None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1, 1], match_colors=True, fig_size=None, save=True):
        """
        Plot histogram for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]

        zkey_values = []
        norm_const_list = []

        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey = zkey / const.M_sun.value if 'M$_{\odot}$' in analysis_zlabel else zkey
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a, vkey, normalize)
                norm_const_list.append(normalization_const)
            else:
                norm_const_list.append(1.0)
        

        mC_min, mC_max = min(zkey_values), max(zkey_values)
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=mC_min, vmax=mC_max)
        cmap_full = plt.get_cmap('plasma')
        cmap = LinearSegmentedColormap.from_list(
            'plasma_truncated',
            cmap_full(np.linspace(0, 0.87, 256))  # 0.85 stops before yellow
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        nrows = 1
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        markers1 = ['o', 's', '^', 'X', 'v']
        lines1 = [':', '--', '-', '-.', ':']
        markers2 = ['^', 'P', '*', 'X', 'v']
        lines2 = ['-', '-.', ':', '-', '--']

        # ✅ STEP 1: Collect ALL data to determine global limits
        all_v_data = []
        all_y1_data = []  # Left axis data
        all_y2_data = []  # Right axis data
        
        for i, analysis_dict in enumerate(analysis_list):
            normalization_const = norm_const_list[i]
            
            # Collect left axis data
            for j, metric_name in enumerate(metric_lists[0]):
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    
                    if metric is None:
                        continue
                    
                    metric = metric * unit_change[1]
                    if normalize:
                        metric = metric / normalization_const
                    
                    all_v_data.append(v_inf)
                    all_y1_data.append(metric)
            
            # Collect right axis data
            for j, metric_name in enumerate(metric_lists[1]):
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    if metric is None:
                        continue
                    
                    metric = metric * unit_change[2]
                    if normalize:
                        metric = metric / normalization_const
                    
                    all_y2_data.append(metric)

        # ✅ Compute global limits
        v_array = np.array(all_v_data)
        y1_array = np.array(all_y1_data)
        y2_array = np.array(all_y2_data)
        
        xlim = (v_array.min(), v_array.max())
        
        if scale[1] == 'log':
            y1_valid = y1_array[y1_array > 0]
            ylim_left = (y1_valid.min() * 0.8, y1_valid.max() * 1.2)
        else:
            ylim_left = (y1_array.min() * 0.95, y1_array.max() * 1.05)
        
        if scale[2] == 'log':
            y2_valid = y2_array[y2_array > 0]
            ylim_right = (y2_valid.min() * 0.8, y2_valid.max() * 1.2)
        else:
            ylim_right = (y2_array.min() * 0.95, y2_array.max() * 1.05)

        print(f"✅ Global limits: x={xlim}, y_left={ylim_left}, y_right={ylim_right}")
        ax1 = axs
        ax2 = ax1.twinx()

        if metric_labels:
            handles_left = []
            labels_left = []
            handles_right = []
            labels_right = []

        # plotting settings
        msize = 0
        lwidth = 1
        alpha = 1
        
        # ✅ STEP 2: Plot with global limits
        for i, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[i]
            normalization_const = norm_const_list[i]
            mC = zkey_values[i]
            color = cmap(norm(mC))
            
            # Plot left axis metrics (metric_lists[0])
            for j, metric_name in enumerate(metric_lists[0]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    metric = metric * unit_change[1]

                    if metric is None:
                        metric = 0

                    metric_array.append(metric)
                    v_array.append(v_inf)

                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[0][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax1.set_yscale(scale[1])
                if ticks[1] is not None:    
                    ax1.set_yticks(ticks[1])


                ax1.plot(v, y, marker=markers1[j], linestyle=lines1[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=msize, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_left.append(plt.Line2D([0], [0], marker=markers1[j], linestyle=lines1[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_left.append(metric_label)
            
            # Plot right axis metrics (metric_lists[1])
            for j, metric_name in enumerate(metric_lists[1]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    metric = metric * unit_change[2]
                    if metric is None:
                        metric = 0

                    metric_array.append(metric)
                    v_array.append(v_inf)

                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[1][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax2.set_yscale(scale[2])
                if ticks[2] is not None:
                    ax2.set_yticks(ticks[2])

                ax2.plot(v, y, marker=markers2[j], linestyle=lines2[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=msize, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_right.append(plt.Line2D([0], [0], marker=markers2[j], linestyle=lines2[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_right.append(metric_label)

        # ✅ STEP 3: Apply global limits
        if global_ylim:
            axs.set_xlim(xlim)
            ax1.set_ylim(ylim_left)
            ax2.set_ylim(ylim_right)
            ax2.set_yscale(scale[2])
            # print(f"Formatter function: {plu.sci_notation_latex}") 
            # ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))

        if ticks[1] is not None:    
            ax1.set_yticks(ticks[1])
        if ticks[2] is not None:
            ax2.set_yticks(ticks[2])


        # Set xlabel only on bottom panel
        axs.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Figure-level y-labels
        fig.text(0.007, 0.4, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.94, 0.4, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
        fig.subplots_adjust(hspace=0.0, left=0.15, right=0.8, top=0.8, bottom=0.01)

        # Combined legend on top panel
        if metric_labels:
            all_handles = handles_left + handles_right
            all_labels = labels_left + labels_right
            axs.legend(all_handles, all_labels, frameon=False, fontsize=10, 
                    ncol=1, loc='lower left')
    
        
        cbar = fig.colorbar(
            sm, 
            ax=axs, 
            label=r'M$_{PBH}$ [M$_\odot$]',
            orientation='horizontal',  # ✅ Make it horizontal
            location='top',            # ✅ Place on top
            pad=0.0,                  # ✅ Padding from plot
            aspect=30,                 # ✅ Width-to-height ratio
            shrink=1                 # ✅ Make it shorter than full width
        )

        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v_multipanel.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')


    # def multiple_analysis_metric_wrt_x_twinaxis(self, metric_lists, metric_ylabels, metric_x='mC', metric_labels=None, ticks=[None, None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1, 1], match_colors=True, fig_size=None, save=True):
    #     """
    #     Plot two metrics on twin y-axes against mass (mC).
    #     Collects metrics from all analyses and plots with mC on x-axis.
        
    #     Parameters:
    #     -----------
    #     metric_lists : list of lists
    #         [[left_metrics], [right_metrics]] - metrics for left and right y-axes
    #     metric_ylabels : list of str
    #         [left_ylabel, right_ylabel]
    #     metric_labels : list of lists, optional
    #         [[left_labels], [right_labels]]
    #     scale : list of str
    #         [x_scale, left_y_scale, right_y_scale]
    #     unit_change : list
    #         [mC_unit, left_metric_unit, right_metric_unit]
    #     """
    #     if analysis_key == 'All':
    #         analysis_list = list(self.analysis_dicts.values())
    #     elif isinstance(analysis_key, list):
    #         analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]

    #     zkey_values = []
    #     norm_const_list = []

    #     for a in analysis_list:
    #         zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
    #         zkey = zkey / const.M_sun.value if 'M$_{\odot}$' in analysis_zlabel else zkey
    #         zkey_values.append(zkey)
    #         if normalize:
    #             normalization_const = self._get_metric_from_sources(a, v_key=None, metric_name=normalize)
    #             norm_const_list.append(normalization_const)
    #         else:
    #             norm_const_list.append(1.0)
        
    #     # Sort by mass for better visualization
    #     sorted_indices = np.argsort(zkey_values)
    #     sorted_mC = np.array(zkey_values)[sorted_indices]
    #     sorted_analyses = [analysis_list[i] for i in sorted_indices]
    #     sorted_norm_const = [norm_const_list[i] for i in sorted_indices]

    #     mC_min, mC_max = sorted_mC.min(), sorted_mC.max()
    #     from matplotlib.colors import LogNorm
    #     norm = LogNorm(vmin=mC_min, vmax=mC_max)
    #     cmap_full = plt.get_cmap('plasma')
    #     cmap = LinearSegmentedColormap.from_list(
    #         'plasma_truncated',
    #         cmap_full(np.linspace(0, 0.87, 256))
    #     )

    #     sm = ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])

    #     nrows = 1
    #     ncols = 1
    #     fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
    #     markers1 = ['o', 's', '^', 'X', 'v']
    #     lines1 = [':', '--', '-', '-.', ':']
    #     markers2 = ['^', 'P', '*', 'X', 'v']
    #     lines2 = ['-', '-.', ':', '-', '--']

    #     # ✅ STEP 1: Collect ALL data to determine global limits
    #     all_y1_data = []  # Left axis data
    #     all_y2_data = []  # Right axis data
    #     all_x_data = []   # x-axis data (mC)
    #     for i, analysis_dict in enumerate(sorted_analyses):
    #         normalization_const = sorted_norm_const[i]
    #         metric_x = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_x)
    #         metric_x = metric_x * unit_change[0]
            
    #         # Collect left axis data
    #         for j, metric_name in enumerate(metric_lists[0]):
                
    #             metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
                
    #             if metric is None:
    #                 continue
                
    #             metric = metric * unit_change[1]
    #             if normalize:
    #                 metric = metric / normalization_const
                
    #             all_y1_data.append(metric)
    #             all_x_data.append(metric_x)

    #         # Collect right axis data
    #         for j, metric_name in enumerate(metric_lists[1]):
                    
    #             metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
    #             if metric is None:
    #                 continue
                
    #             metric = metric * unit_change[2]
    #             if normalize:
    #                 metric = metric / normalization_const
                
    #             all_y2_data.append(metric)

    #     # ✅ Compute global limits
    #     y1_array = np.array(all_y1_data)
    #     y2_array = np.array(all_y2_data)
    #     x_array = np.array(all_x_data)
    #     xlim = (x_array.min(), x_array.max())
        
    #     if scale[1] == 'log':
    #         y1_valid = y1_array[y1_array > 0]
    #         ylim_left = (y1_valid.min() * 0.8, y1_valid.max() * 1.2)
    #     else:
    #         ylim_left = (y1_array.min() * 0.95, y1_array.max() * 1.05)
        
    #     if scale[2] == 'log':
    #         y2_valid = y2_array[y2_array > 0]
    #         ylim_right = (y2_valid.min() * 0.8, y2_valid.max() * 1.2)
    #     else:
    #         ylim_right = (y2_array.min() * 0.95, y2_array.max() * 1.05)

    #     print(f"✅ Global limits: x={xlim}, y_left={ylim_left}, y_right={ylim_right}")
    #     ax1 = axs
    #     ax2 = ax1.twinx()

    #     if metric_labels:
    #         handles_left = []
    #         labels_left = []
    #         handles_right = []
    #         labels_right = []

    #     # plotting settings
    #     msize = 0
    #     lwidth = 1
    #     alpha = 1
        
    #     # ✅ STEP 2: Plot all data with global limits
    #     for i, analysis_dict in enumerate(sorted_analyses):
    #         mC = sorted_mC[i]
    #         normalization_const = sorted_norm_const[i]
    #         color = cmap(norm(mC))
            
    #         # Plot left axis metrics (metric_lists[0])
    #         for j, metric_name in enumerate(metric_lists[0]):
    #             metric_array = []
                    
    #             metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
    #             metric = metric * unit_change[1]

    #             if metric is None:
    #                 metric = 0

    #             metric_array.append(metric)
    #             m = np.asarray(all_x_data[j])  # x-axis data (mC)
    #             y = np.asarray(metric_array)

    #             if metric_labels and i == 0:
    #                 metric_label = metric_labels[0][j]
    #             else:
    #                 metric_label = None
                
    #             if normalize:
    #                 y = y / normalization_const
                
    #             ax1.set_yscale(scale[1])
    #             if ticks[1] is not None:    
    #                 ax1.set_yticks(ticks[1])

    #             ax1.plot(m, y, marker=markers1[j], linestyle=lines1[j], alpha=alpha, 
    #                     label=metric_label, color=color, markersize=msize, 
    #                     linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
    #             if metric_labels and i == 0:
    #                 handles_left.append(plt.Line2D([0], [0], marker=markers1[j], linestyle=lines1[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
    #                 labels_left.append(metric_label)
            
    #         # Plot right axis metrics (metric_lists[1])
    #         for j, metric_name in enumerate(metric_lists[1]):
    #             metric_array = []

                  
    #             metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
    #             metric = metric * unit_change[2]
    #             if metric is None:
    #                 metric = 0

    #             metric_array.append(metric)

    #             m = np.asarray(all_x_data[j])  # x-axis data (mC)
    #             y = np.asarray(metric_array)

    #             if metric_labels and i == 0:
    #                 metric_label = metric_labels[1][j]
    #             else:
    #                 metric_label = None
                
    #             if normalize:
    #                 y = y / normalization_const
                
    #             ax2.set_yscale(scale[2])
    #             if ticks[2] is not None:
    #                 ax2.set_yticks(ticks[2])

    #             ax2.plot(m, y, marker=markers2[j], linestyle=lines2[j], alpha=alpha, 
    #                     label=metric_label, color=color, markersize=msize, 
    #                     linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
    #             if metric_labels and i == 0:
    #                 handles_right.append(plt.Line2D([0], [0], marker=markers2[j], linestyle=lines2[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
    #                 labels_right.append(metric_label)

    #     # ✅ STEP 3: Apply global limits
    #     if global_ylim:
    #         axs.set_xlim(xlim)
    #         ax1.set_ylim(ylim_left)
    #         ax2.set_ylim(ylim_right)
    #         ax2.set_yscale(scale[2])

    #     if ticks[1] is not None:    
    #         ax1.set_yticks(ticks[1])
    #     if ticks[2] is not None:
    #         ax2.set_yticks(ticks[2])
        
    #     if scale[0] == 'log':
    #         axs.set_xscale('log')

    #     # Set xlabel and ylabel
    #     axs.set_xlabel(f'{analysis_zlabel[0]} [{analysis_zlabel[1]}]')
        
    #     # Figure-level y-labels
    #     fig.text(0.007, 0.4, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
    #     fig.text(0.94, 0.4, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
    #     fig.subplots_adjust(hspace=0.0, left=0.15, right=0.8, top=0.8, bottom=0.01)

    #     # Combined legend 
    #     if metric_labels:
    #         all_handles = handles_left + handles_right
    #         all_labels = labels_left + labels_right
    #         axs.legend(all_handles, all_labels, frameon=False, fontsize=10, 
    #                 ncol=1, loc='lower left')

        
    #     cbar = fig.colorbar(
    #         sm, 
    #         ax=axs, 
    #         label=r'M$_{PBH}$ [M$_\odot$]',
    #         orientation='horizontal',
    #         location='top',
    #         pad=0.0,
    #         aspect=30,
    #         shrink=1
    #     )

    #     if save:
    #         metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
    #         figname = f'multiple_analysis_{metricsname}_twinaxis_vs_m.png'
    #         out = os.path.join(self.plots_dir, figname)
    #         fig.savefig(out, dpi=300, bbox_inches='tight')
    def multiple_analysis_metric_wrt_x(self, metric_list, metric_ylabel, metric_xname='mC', metric_x_label=None, metric_labels=None, ticks=[None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1], match_colors=True, fig_size=None, save=True, save_as=None):
        """
        Plot one or more metrics on a single y-axis against a custom x-axis metric.
        
        Parameters:
        -----------
        metric_list : list
            Metrics to plot on y-axis
        metric_ylabel : str
            Y-axis label
        metric_xname : str or list
            Metric name (or nested list of keys) for x-axis
        metric_x_label : str, optional
            Label for x-axis (inferred if None)
        metric_labels : list, optional
            Labels for each metric
        scale : list of str
            [x_scale, y_scale]
        unit_change : list
            [x_unit, y_metric_unit]
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]

        zkey_values = []
        norm_const_list = []

        for a in analysis_list:
            zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
            zkey = zkey / const.M_sun.value if 'M$_{\odot}$' in analysis_zlabel else zkey
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a, v_key=None, metric_name=normalize)
                norm_const_list.append(normalization_const)
            else:
                norm_const_list.append(1.0)
        
        # Sort by colorbar key
        sorted_indices = np.argsort(zkey_values)
        sorted_mC = np.array(zkey_values)[sorted_indices]
        sorted_analyses = [analysis_list[i] for i in sorted_indices]
        sorted_norm_const = [norm_const_list[i] for i in sorted_indices]

        mC_min, mC_max = sorted_mC.min(), sorted_mC.max()
        n_analyses = len(sorted_analyses)
        cmap_full = plt.get_cmap('plasma')
        color_indices = np.linspace(0, 1, n_analyses)
        colors = [cmap_full(idx) for idx in color_indices]

        from matplotlib.colors import LogNorm, ListedColormap
        norm_cbar = LogNorm(vmin=sorted_mC.min(), vmax=sorted_mC.max())
        cmap_discrete = ListedColormap(colors)
        sm = ScalarMappable(cmap=cmap_discrete, norm=norm_cbar)
        sm.set_array([])

        fig, ax = self._figure(figsize=fig_size if fig_size is not None else (3.5, 2.5))
        
        markers = ['o', 's', '^', 'X', 'v', 'D', 'p']
        # lines = ['-', '-', '-', '-', '-', '-', ':']
        lines = ['-', ':', '-', '-.', ':', '--', '-.']
        # ✅ STEP 1: Collect ALL data to determine global limits
        all_y_data = []
        all_x_data = []
        data_dict = {}
        
        for i, analysis_dict in enumerate(sorted_analyses):
            normalization_const = sorted_norm_const[i]
            metric_x_val = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_xname)
            metric_x_val = metric_x_val * unit_change[0]
            
            y_vals = []
            # Collect y-axis data
            for j, metric_name in enumerate(metric_list):
                print(f'plotting metric: {metric_name} for analysis {i+1}/{n_analyses}')
                metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
                print(f'raw metric value: {metric}')
                if metric is None:
                    continue
                
                metric = metric * unit_change[1]
                if normalize:
                    metric = metric / normalization_const
                
                all_y_data.append(metric)
                y_vals.append(metric)
            
            all_x_data.append(metric_x_val)
            data_dict[i] = {
                'x': metric_x_val,
                'y': y_vals
            }

        # ✅ Compute global limits
        x_array = np.array(all_x_data)
        y_array = np.array(all_y_data)
        
        xlim = (x_array.min(), x_array.max())
        
        if scale[1] == 'log':
            y_valid = y_array[y_array > 0]
            ylim = (y_valid.min() * 0.8, y_valid.max() * 1.2)
        else:
            ylim = (y_array.min() * 0.95, y_array.max() * 1.05)

        print(f"✅ Global limits: x={xlim}, y={ylim}")

        if metric_labels:
            handles = []
            labels = []

        # plotting settings
        msize = 0
        lwidth = 2
        alpha = 1
        
        # ✅ STEP 2: Plot with global limits
        for i, analysis_dict in enumerate(sorted_analyses):
            mC = sorted_mC[i]
            normalization_const = sorted_norm_const[i]
            color = colors[i]
            x_val = data_dict[i]['x']
            # Plot y-axis metrics
            for j, mtrc in enumerate(data_dict[i]['y']):
                x = np.asarray(x_val)
                y = np.asarray(mtrc)

                if metric_labels and i == 0:
                    label = metric_labels[j]
                else:
                    label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax.set_yscale(scale[1])
                if ticks[1] is not None:    
                    ax.set_yticks(ticks[1])

                ax.plot(x, y, marker=markers[j % len(markers)], linestyle=lines[j % len(lines)], alpha=alpha, 
                        label=label, color=color, markersize=msize, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0, zorder=j+1)
                
                if metric_labels and i == 0:
                    handles.append(plt.Line2D([0], [0], marker=markers[j % len(markers)], linestyle=lines[j % len(lines)], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels.append(label)

        # ✅ STEP 3: Apply global limits
        if global_ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        if ticks[0] is not None:    
            ax.set_xticks(ticks[0])
            ax.set_xticklabels([f'{x:.0f}' if x >= 1 else f'{x:.2f}' for x in ticks[0]])  # Format x-ticks as integers if they are >= 1
        if ticks[1] is not None:
            ax.set_yticks(ticks[1])

        if scale[0] == 'log':
            ax.set_xscale('log')

        # Set labels
        if metric_x_label is None:
            if isinstance(metric_xname, list):
                metric_x_label = ' → '.join(metric_xname)
            else:
                metric_x_label = metric_xname
        ax.set_xlabel(metric_x_label)
        ax.set_ylabel(metric_ylabel)
        
        fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)

        # Legend 
        if metric_labels:
            ax.legend(handles, labels, frameon=False, fontsize=10, 
                    ncol=1, loc='best')

        cbar = fig.colorbar(
            sm, 
            ax=ax, 
            label=r'M$_{PBH}$ [M$_\odot$]',
            orientation='horizontal',
            location='top',
            pad=0.0,
            aspect=40,
        )

        if save:
            if save_as is not None:
                figname = save_as
            else:
                metricsname = "_".join([plu.latex_label_key(metric_ylabel)])
                figname = f'multiple_analysis_{metricsname}_vs_x_.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')

    def multiple_analysis_metric_wrt_x_twinaxis(self, metric_lists, metric_ylabels, metric_xname='mC', metric_x_label=None, metric_labels=None, ticks=[None, None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1, 1], match_colors=True, fig_size=None, save=True):
        """
        Plot two metrics on twin y-axes against a custom x-axis metric.
        
        Parameters:
        -----------
        metric_lists : list of lists
            [[left_metrics], [right_metrics]] - metrics for left and right y-axes
        metric_ylabels : list of str
            [left_ylabel, right_ylabel]
        metric_xname : str or list
            Metric name (or nested list of keys) for x-axis
        metric_x_label : str, optional
            Label for x-axis (inferred if None)
        metric_labels : list of lists, optional
            [[left_labels], [right_labels]]
        scale : list of str
            [x_scale, left_y_scale, right_y_scale]
        unit_change : list
            [x_unit, left_metric_unit, right_metric_unit]
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]

        zkey_values = []
        norm_const_list = []

        for a in analysis_list:
            zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
            zkey = zkey / const.M_sun.value if 'M$_{\odot}$' in analysis_zlabel else zkey
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a, v_key=None, metric_name=normalize)
                norm_const_list.append(normalization_const)
            else:
                norm_const_list.append(1.0)
        
        # Sort by mass for better visualization
        sorted_indices = np.argsort(zkey_values)
        sorted_mC = np.array(zkey_values)[sorted_indices]
        sorted_analyses = [analysis_list[i] for i in sorted_indices]
        sorted_norm_const = [norm_const_list[i] for i in sorted_indices]

        mC_min, mC_max = sorted_mC.min(), sorted_mC.max()
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=mC_min, vmax=mC_max)
        cmap_full = plt.get_cmap('plasma')
        cmap = LinearSegmentedColormap.from_list(
            'plasma_truncated',
            cmap_full(np.linspace(0, 0.87, 256))
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        nrows = 1
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        markers1 = ['o', 's', '^', 'X', 'v']
        lines1 = [':', '--', '-', '-.', ':']
        markers2 = ['^', 'P', '*', 'X', 'v']
        lines2 = ['-', '-.', ':', '-', '--']

        # ✅ STEP 1: Collect ALL data to determine global limits
        all_y1_data = []  # Left axis data
        all_y2_data = []  # Right axis data
        all_x_data = []   # x-axis data (mC)
        data_dict_r = {}  # To store data for each analysis for later plotting
        data_dict_l = {}
        for i, analysis_dict in enumerate(sorted_analyses):
            normalization_const = sorted_norm_const[i]
            metric_x = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_xname)
            metric_x = metric_x * unit_change[0]
            y = []
            # Collect left axis data
            for j, metric_name in enumerate(metric_lists[0]):
                
                metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
                
                if metric is None:
                    continue
                
                metric = metric * unit_change[1]
                if normalize:
                    metric = metric / normalization_const
                
                all_y1_data.append(metric)
                y.append(metric)
            
            all_x_data.append(metric_x)
            data_dict_l[i] = {
                'x': metric_x,
                'y': y
            }
            # Collect right axis data
            y = []
            for j, metric_name in enumerate(metric_lists[1]):
                    
                metric = self._get_metric_from_sources(analysis_dict, v_key=None, metric_name=metric_name)
                if metric is None:
                    continue
                
                metric = metric * unit_change[2]
                if normalize:
                    metric = metric / normalization_const
                
                all_y2_data.append(metric)
                y.append(metric)
            data_dict_r[i] = {
                'x': metric_x,
                'y': y
            }

        # ✅ Compute global limits
        x_array = np.array(all_x_data)
        y1_array = np.array(all_y1_data)
        y2_array = np.array(all_y2_data)
        
        xlim = (x_array.min(), x_array.max())
        
        if scale[1] == 'log':
            y1_valid = y1_array[y1_array > 0]
            ylim_left = (y1_valid.min() * 0.8, y1_valid.max() * 1.2)
        else:
            ylim_left = (y1_array.min() * 0.95, y1_array.max() * 1.05)
        
        if scale[2] == 'log':
            y2_valid = y2_array[y2_array > 0]
            ylim_right = (y2_valid.min() * 0.8, y2_valid.max() * 1.2)
        else:
            ylim_right = (y2_array.min() * 0.95, y2_array.max() * 1.05)

        print(f"✅ Global limits: x={xlim}, y_left={ylim_left}, y_right={ylim_right}")
        ax1 = axs
        ax2 = ax1.twinx()

        if metric_labels:
            handles_left = []
            labels_left = []
            handles_right = []
            labels_right = []

        # plotting settings
        msize = 5
        lwidth = 1
        alpha = 1
        
        # ✅ STEP 2: Plot with global limits
        for i, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[i]
            normalization_const = norm_const_list[i]
            mC = zkey_values[i]
            color = cmap(norm(mC))
            x_array_plot = data_dict_l[i]['x']  # x-axis data for this analysis (already scaled by unit_change[0])
            # Plot left axis metrics (metric_lists[0])
            for j, mtrc in enumerate(data_dict_l[i]['y']):

                x = np.asarray(x_array_plot)
                y = np.asarray(mtrc)

                if metric_labels and i == 0:
                    metric_label = metric_labels[0][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax1.set_yscale(scale[1])
                if ticks[1] is not None:    
                    ax1.set_yticks(ticks[1])

                ax1.plot(x, y, marker=None, linestyle=lines1[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=0, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_left.append(plt.Line2D([0], [0], linestyle=lines1[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_left.append(metric_label)
            
            # Plot right axis metrics (metric_lists[1])
            for j, mtrc in enumerate(data_dict_r[i]['y']):

                x = np.asarray(x_array_plot)
                y = np.asarray(mtrc)

                if metric_labels and i == 0:
                    if len(metric_labels[1]) > 1:
                        metric_label = metric_labels[1][j]
                    else:
                        metric_label =  metric_labels[1][0]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax2.set_yscale(scale[2])
                if ticks[2] is not None:
                    ax2.set_yticks(ticks[2])

                ax2.plot(x, y, marker=markers2[j], linestyle=lines2[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=msize, 
                        linewidth=0, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_right.append(plt.Line2D([0], [0], marker=markers2[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_right.append(metric_label)

        # ✅ STEP 3: Apply global limits
        if global_ylim:
            axs.set_xlim(xlim)
            ax1.set_ylim(ylim_left)
            ax2.set_ylim(ylim_right)
            ax2.set_yscale(scale[2])

        if ticks[1] is not None:    
            ax1.set_yticks(ticks[1])
        if ticks[2] is not None:
            ax2.set_yticks(ticks[2])

        if scale[0] == 'log':
            axs.set_xscale('log')

        # Set xlabel and ylabel
        if metric_x_label is None:
            if isinstance(metric_x, list):
                metric_x_label = ' → '.join(metric_x)
            else:
                metric_x_label = metric_x
        axs.set_xlabel(metric_x_label)
        
        # Figure-level y-labels
        fig.text(0.007, 0.4, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.94, 0.4, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
        fig.subplots_adjust(hspace=0.0, left=0.15, right=0.8, top=0.8, bottom=0.01)

        # Combined legend 
        if metric_labels:
            all_handles = handles_left + handles_right
            all_labels = labels_left + labels_right
            axs.legend(all_handles, all_labels, frameon=False, fontsize=10, 
                    ncol=1, loc='lower left')
    
        
        cbar = fig.colorbar(
            sm, 
            ax=axs, 
            label=r'M$_{PBH}$ [M$_\odot$]',
            orientation='horizontal',
            location='top',
            pad=0.0,
            aspect=30,
            shrink=1
        )

        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_x.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
    
    def multiple_analysis_metric_wrt_v_twinaxis_errorband(self, metric_lists, metric_ylabels, metric_labels=None, ticks=[None, None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1, 1], match_colors=True, fig_size=None, save=True):
        """
        Plot histogram for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]

        zkey_values = []
        norm_const_list = []

        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey = zkey / const.M_sun.value if 'M$_{\odot}$' in analysis_zlabel else zkey
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a, vkey, normalize)
                norm_const_list.append(normalization_const)
            else:
                norm_const_list.append(1.0)
        

        mC_min, mC_max = min(zkey_values), max(zkey_values)
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=mC_min, vmax=mC_max)
        cmap_full = plt.get_cmap('plasma')
        cmap = LinearSegmentedColormap.from_list(
            'plasma_truncated',
            cmap_full(np.linspace(0, 0.87, 256))  # 0.85 stops before yellow
        )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        nrows = 1
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        markers1 = ['o', 's', '^', 'X', 'v']
        lines1 = [':', '--', '-', '-.', ':']
        markers2 = ['^', 'P', '*', 'X', 'v']
        lines2 = ['-', '-.', ':', '-', '--']

        # ✅ STEP 1: Collect ALL data to determine global limits
        all_v_data = []
        all_y1_data = []  # Left axis data
        all_y2_data = []  # Right axis data
        
        for i, analysis_dict in enumerate(analysis_list):
            normalization_const = norm_const_list[i]
            
            # Collect left axis data
            for j, metric_name in enumerate(metric_lists[0]):
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    
                    if metric is None:
                        continue
                    
                    metric = metric * unit_change[1]
                    if normalize:
                        metric = metric / normalization_const
                    
                    all_v_data.append(v_inf)
                    all_y1_data.append(metric)
            
            # Collect right axis data
            for j, metric_name in enumerate(metric_lists[1]):
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    if metric is None:
                        continue
                    
                    metric = metric * unit_change[2]
                    if normalize:
                        metric = metric / normalization_const
                    
                    all_y2_data.append(metric)

        # ✅ Compute global limits
        v_array = np.array(all_v_data)
        y1_array = np.array(all_y1_data)
        y2_array = np.array(all_y2_data)
        
        xlim = (v_array.min(), v_array.max())
        
        if scale[1] == 'log':
            y1_valid = y1_array[y1_array > 0]
            ylim_left = (y1_valid.min() * 0.8, y1_valid.max() * 1.2)
        else:
            ylim_left = (y1_array.min() * 0.95, y1_array.max() * 1.05)
        
        if scale[2] == 'log':
            y2_valid = y2_array[y2_array > 0]
            ylim_right = (y2_valid.min() * 0.8, y2_valid.max() * 1.2)
        else:
            ylim_right = (y2_array.min() * 0.95, y2_array.max() * 1.05)

        print(f"✅ Global limits: x={xlim}, y_left={ylim_left}, y_right={ylim_right}")
        ax1 = axs
        ax2 = ax1.twinx()

        if metric_labels:
            handles_left = []
            labels_left = []
            handles_right = []
            labels_right = []

        # plotting settings
        msize = 0
        lwidth = 1
        alpha = 1
        
        # ✅ STEP 2: Plot with global limits
        for i, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[i]
            normalization_const = norm_const_list[i]
            mC = zkey_values[i]
            color = cmap(norm(mC))
            
            # Plot left axis metrics (metric_lists[0])
            for j, metric_name in enumerate(metric_lists[0]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    metric = metric * unit_change[1]

                    if metric is None:
                        metric = 0

                    metric_array.append(metric)
                    v_array.append(v_inf)

                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[0][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax1.set_yscale(scale[1])
                if ticks[1] is not None:    
                    ax1.set_yticks(ticks[1])


                ax1.plot(v, y, marker=markers1[j], linestyle=lines1[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=msize, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_left.append(plt.Line2D([0], [0], marker=markers1[j], linestyle=lines1[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_left.append(metric_label)
            
            # Plot right axis metrics (metric_lists[1])
            for j, metric_name in enumerate(metric_lists[1]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    metric = metric * unit_change[2]
                    if metric is None:
                        metric = 0

                    metric_array.append(metric)
                    v_array.append(v_inf)

                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[1][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax2.set_yscale(scale[2])
                if ticks[2] is not None:
                    ax2.set_yticks(ticks[2])

                ax2.plot(v, y, marker=markers2[j], linestyle=lines2[j], alpha=alpha, 
                        label=metric_label, color=color, markersize=msize, 
                        linewidth=lwidth, markeredgecolor='none', markeredgewidth=0)
                
                if metric_labels and i == 0:
                    handles_right.append(plt.Line2D([0], [0], marker=markers2[j], linestyle=lines2[j], color='black', markersize=8 if msize>0 else 0, linewidth=2, markeredgecolor='black', markeredgewidth=0.6))
                    labels_right.append(metric_label)

        # ✅ STEP 3: Apply global limits
        if global_ylim:
            axs.set_xlim(xlim)
            ax1.set_ylim(ylim_left)
            ax2.set_ylim(ylim_right)
            ax2.set_yscale(scale[2])
            # print(f"Formatter function: {plu.sci_notation_latex}") 
            # ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))

        if ticks[1] is not None:    
            ax1.set_yticks(ticks[1])
        if ticks[2] is not None:
            ax2.set_yticks(ticks[2])


        # Set xlabel only on bottom panel
        axs.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Figure-level y-labels
        fig.text(0.007, 0.4, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.94, 0.4, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
        fig.subplots_adjust(hspace=0.0, left=0.15, right=0.8, top=0.8, bottom=0.01)

        # Combined legend on top panel
        if metric_labels:
            all_handles = handles_left + handles_right
            all_labels = labels_left + labels_right
            axs.legend(all_handles, all_labels, frameon=False, fontsize=10, 
                    ncol=1, loc='lower left')
    
        
        cbar = fig.colorbar(
            sm, 
            ax=axs, 
            label=r'M$_{PBH}$ [M$_\odot$]',
            orientation='horizontal',  # ✅ Make it horizontal
            location='top',            # ✅ Place on top
            pad=0.0,                  # ✅ Padding from plot
            aspect=30,                 # ✅ Width-to-height ratio
            shrink=1                 # ✅ Make it shorter than full width
        )

        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v_multipanel.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')

    def multiple_analysis_metric_wrt_v_multipanel(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = [r'M$_{PBH}$', r'M$_{\odot}$'], scale = 'log', v_key = 'All', unit_change=[1e-3, 1, 1], normalize='importance_sampling_size', fig_size=None, save=True):
        """
        Plot histogram for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = self.analysis_dicts.values()
        elif isinstance(analysis_key, list):
            for a in self.analysis_list:
                analysis_list = [a for a in self.analysis_dicts if a.get_name() in analysis_key]

        zkey_values = []

        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
            


        nrows = len(analysis_list)
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        markers = ['o', 's', '^', 'X', 'v']
        lines = [':', '--', '-', '-.', ':']
        cmap = plt.get_cmap('plasma', len(metric_list) + 1)

        for i, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[i]
            
            for j, metric_name in enumerate(metric_list):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)

                    if metric is None:
                        metric = 0
                        print(f"Metric '{metric_name}' not found for v={v}.")

                    metric = metric * unit_change[1]
                    if normalize:
                        normalization_const = self._get_metric_from_sources(analysis_dict, v, normalize)
                        metric = metric / normalization_const
                    metric_array.append(metric)
                    v_array.append(v_inf)  # convert to km/s


                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                if scale == 'log':
                    axs[i].set_yscale('log')

                axs[i].plot(v, y, marker=markers[j], linestyle=lines[j], alpha=0.8, 
                        label=metric_label, color=cmap(j), markersize=7, linewidth=1.5,markeredgecolor='black', markeredgewidth=0.6)
                

                
                # Add mC label on right side of each panel
                axs[i].text(0.07, 0.1, fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}', transform=axs[i].transAxes,
                        rotation=0, va='center', fontsize=9)

        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Set ylabel on middle panel (or use fig.supylabel)
        # Option 1: Middle axis
        # axs[len(axs)//2].set_ylabel(metric_ylabel or 'Occurrences')
        
        # Option 2: Figure-level ylabel (better for multi-panel)
        fig.supylabel(metric_ylabel or 'Occurrences', x=0.02)
        fig.subplots_adjust(hspace=0.0, left=0.2, right=0.98, top=0.9, bottom=0.08)

        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False) 
        # for ax in axs:
        #     ax.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
    

        ylims = [ax.get_ylim() for ax in axs]
        global_ylim = (min([y[0] for y in ylims])*0.8, max([y[1] for y in ylims])+0.5*max([y[1] for y in ylims]))
        for ax in axs:
            ax.set_ylim(global_ylim)
        # Legend only on top panel
        axs[0].legend(frameon=False, fontsize=9, ncol=2, 
                      bbox_to_anchor=(0.5, 0.95), loc='lower center')

        if save:
            figname = f'multiple_analysis_{"_".join(metric_list)}_vs_v.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300)       

    def multiple_analysis_metric_wrt_v(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = r'M$_{PBH}$', scale = 'log', v_key = 'All', fig_size=None, save=True):
        """
        Plot histogram for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = self.analysis_dicts.values()
        elif isinstance(analysis_key, list):
            for a in self.analysis_list:
                analysis_list = [a for a in self.analysis_dicts if a.get_name() in analysis_key]
        
        mC_values = []

        for a in analysis_list:
            vkey = list(a.keys())[0]
            mC = self._get_metric_from_sources(a, vkey, analysis_zkey)
            mC_values.append(mC)

        mC_min, mC_max = min(mC_values), max(mC_values)
        norm = Normalize(vmin=mC_min, vmax=mC_max)
        cmap = plt.get_cmap('plasma')
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = self._figure(figsize=(7, 4))
        markers = ['o', 's', '^', '*', 'v']
        lines = [':', '--', '-', '-.', ':']

        for i, analysis_dict in enumerate(analysis_list):
            mC = mC_values[i]
            color = cmap(norm(mC))
            for j, metric_name in enumerate(metric_list):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)

                    if metric is None:
                        metric = 0
                        print(f"Metric '{metric_name}' not found for v={v}.")

                    metric_array.append(metric)
                    v_array.append(v_inf / 1e3)  # convert to km/s


                v = np.asarray(v_array)
                y = np.asarray(metric_array)

                if metric_labels and i == 0:
                    metric_label = metric_labels[j]
                else:
                    metric_label = None

                ax.plot(v, y, marker=markers[j], linestyle=lines[j], alpha=0.8, label=metric_label, color=color, markersize=4, linewidth=1.5)

        ax.set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        ax.set_ylabel(metric_ylabel or 'Occurrences')
        ax.set_yscale('log')
        ax.legend(frameon=False, fontsize=8, ncol=2)
   
        # Add colorbar
        cbar = fig.colorbar(sm, ax=ax, label=r'$m_C$ [M$_\odot$]')
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.update_ticks()

        if save:
            figname = f'multiple_analysis_{"_".join(metric_list)}_vs_v.png'
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)

    def multiple_analysis_metric_wrt_v_twinaxis_multipanel(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, unit_change=[1, 1, 1], match_colors=True, fig_size=None, save=True):
        """
        Plot two sets of metrics on twin y-axes in multi-panel layout.
        Each panel represents one analysis with left and right y-axes.
        
        Parameters:
        -----------
        metric_lists : list of lists
            [[left_metrics], [right_metrics]] - metrics for left and right y-axes
        metric_ylabels : list of str
            [left_ylabel, right_ylabel]
        metric_labels : list of lists, optional
            [[left_labels], [right_labels]]
        scale : list of str
            ['log'/'linear', 'log'/'linear'] for x, left and right y axes
        normalize : bool
            Whether to normalize by importance_sampling_size
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
    
        zkey_values = []
        norm_const_list = []
    
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a, vkey, normalize)
                norm_const_list.append(normalization_const)
            else:
                norm_const_list.append(1.0)
            
    
        nrows = len(analysis_list)
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
        markers = ['o', 's', '^', 'X', 'v']
        lines = ['-', ':', '-', '-.', ':']

        if match_colors:
            cmap = plt.get_cmap('plasma', max(len(metric_lists[0]), len(metric_lists[1])) + 1)
        else:
            cmap = plt.get_cmap('plasma', len(metric_lists[0]) + len(metric_lists[1]) + 1)

        ax2_list = []
        
        for i, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[i]
            normalization_const = norm_const_list[i]
            
            ax1 = axs[i]
            ax2 = ax1.twinx()
            ax2_list.append(ax2)
            
            # Plot left axis metrics (metric_lists[0])
            for j, metric_name in enumerate(metric_lists[0]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)

                    metric = metric * unit_change[1]

                    if metric is None:
                        metric = 0
                        print(f"Metric '{metric_name}' not found for v={v}.")
    
                    metric_array.append(metric)
                    v_array.append(v_inf)
    
                v = np.asarray(v_array)
                y = np.asarray(metric_array)
    
                if metric_labels and i == 0:
                    metric_label = metric_labels[0][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax1.set_yscale(scale[1])
                

                ax1.plot(v, y, marker=markers[0], linestyle=lines[0], alpha=0.8, 
                        label=metric_label, color=cmap(j), markersize=5, 
                        linewidth=2, markeredgecolor='black', markeredgewidth=0.6)
            
            # Plot right axis metrics (metric_lists[1])
            for j, metric_name in enumerate(metric_lists[1]):
                metric_array = []
                v_array = []
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    metric = metric * unit_change[2]
                    if metric is None:
                        metric = 0
                        print(f"Metric '{metric_name}' not found for v={v}.")
    
                    metric_array.append(metric)
                    v_array.append(v_inf)
    
                v = np.asarray(v_array)
                y = np.asarray(metric_array)
    
                if metric_labels and i == 0:
                    metric_label = metric_labels[1][j]
                else:
                    metric_label = None
                
                if normalize:
                    y = y / normalization_const
                
                ax2.set_yscale(scale[2])
    
                ax2.plot(v, y, marker=markers[1], linestyle=lines[1], alpha=0.8, 
                        label=metric_label, color=cmap(j) if match_colors else cmap(len(metric_lists[0]) + j), markersize=5, 
                        linewidth=2, markeredgecolor='black', markeredgewidth=0.6)
            
            # Add analysis label on right side of panel
            ax2.text(0.05, 0.1, fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}', 
                    transform=ax2.transAxes, rotation=0, va='center', fontsize=9)
    
        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Figure-level y-labels
        fig.text(0.007, 0.5, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.99, 0.5, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
        fig.subplots_adjust(hspace=0.0, left=0.15, right=0.88, top=0.9, bottom=0.08)

        # Hide x-tick labels on all but bottom panel
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)
        
        # # Format y-axis ticks for log scale
        # for ax, ax2 in zip(axs, ax2_list):
        #     # Left axis
        #     if scale[1] == 'log':
        #         ax.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
                
        #         # Get current y-limits
        #         ymin, ymax = ax.get_ylim()
                
        #         # Find all powers of 10 within the range
        #         log_min = np.floor(np.log10(ymin))
        #         log_max = np.ceil(np.log10(ymax))
                
        #         # Generate all integer powers of 10 in range
        #         log_ticks = np.arange(log_min, log_max + 1)  # +1 to include log_max
                
        #         # Convert to actual values
        #         tick_values = 10**log_ticks
                
        #         # Filter to only ticks actually within the visible range
        #         tick_values = tick_values[(tick_values >= ymin) & (tick_values <= ymax)]
                
        #         ax.set_yticks(tick_values)
        #     else:
        #         ax.yaxis.set_major_locator(MaxNLocator(3))
            
        #     # Right axis
        #     if scale[2] == 'log':
        #         ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
                
        #         ymin, ymax = ax2.get_ylim()
                
        #         # Find all powers of 10 within the range
        #         log_min = np.floor(np.log10(ymin))
        #         log_max = np.ceil(np.log10(ymax))
                
        #         # Generate all integer powers of 10 in range
        #         log_ticks = np.arange(log_min, log_max + 1)
                
        #         # Convert to actual values
        #         tick_values = 10**log_ticks
                
        #         # Filter to only ticks within visible range
        #         tick_values = tick_values[(tick_values >= ymin) & (tick_values <= ymax)]
                
        #         ax2.set_yticks(tick_values)
        #     else:
        #         ax2.yaxis.set_major_locator(MaxNLocator(3))
        if global_ylim:
            # Synchronize y-limits across panels
            ylims_left = [ax.get_ylim() for ax in axs]
            global_ylim_left = (min([y[0] for y in ylims_left])*0.8, max([y[1] for y in ylims_left])*1.2)
            for ax in axs:
                ax.set_ylim(global_ylim_left)
            
            ylims_right = [ax2.get_ylim() for ax2 in ax2_list]
            global_ylim_right = (min([y[0] for y in ylims_right])*0.8, max([y[1] for y in ylims_right])*1.2)
            for ax2 in ax2_list:
                ax2.set_ylim(global_ylim_right)
    
        # Combined legend on top panel
        if metric_labels:
            handles_left = []
            labels_left = []
            handles_right = []
            labels_right = []
            
            # Create dummy handles for metrics
            from matplotlib.patches import Rectangle
            for j, label in enumerate((metric_labels[0]+ metric_labels[1])):
                handles_left.append(Rectangle((0,0),1,1, facecolor=cmap(j), 
                                             edgecolor='black', linewidth=0.6, alpha=0.8))
                labels_left.append(label.split(' ')[0])  # Remove any unit info in parentheses
            
            # Create dummy handles for axes
            for j, label in enumerate(metric_ylabels):
                handles_right.append(Line2D([0], [0], 
                                    marker=markers[j], 
                                    linestyle=lines[j],
                                    color='white', 
                                    markersize=5,
                                    linewidth=2,
                                    markeredgecolor='black', 
                                    markeredgewidth=0.6,
                                    alpha=0.8))
                labels_right.append(label.split(' ')[0])  # Remove any unit info in parentheses

            # Combine handles and labels
            all_handles = handles_left + handles_right
            all_labels = labels_left + labels_right

            axs[0].legend(all_handles, all_labels, frameon=False, fontsize=9, 
                         ncol=2, bbox_to_anchor=(0.5, 1.08), loc='lower center')
    
        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v_multipanel.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        

    def multiple_analysis_metric_array_wrt_v_twinaxis_multipanel(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], unit_change=[1e-3, 1, 1], global_ylim=False, metric_masks=None, fig_size=None, save_as=None, save=True):
        """
        Plot array metrics (like semi_major_axes) as violin plots with twin y-axes in multi-panel layout.
        Each panel represents one analysis, with two sets of violins (one per y-axis).
        
        Parameters:
        -----------
        metric_lists : list of lists
            [[left_metrics], [right_metrics]] - metrics for left and right y-axes
        metric_ylabels : list of str
            [left_ylabel, right_ylabel]
        metric_labels : list of lists, optional
            [[left_labels], [right_labels]]
        metric_masks : list of lists, optional
            [[left_masks], [right_masks]] - masks for filtering data
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
            zkey_values.append(zkey)
        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 1.5*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
            # Create twin axis for this panel
        cmap = plt.get_cmap('plasma', 15)
        n, m = 3, 8
        ax2_list = []
        # Process each analysis
        for panel_idx, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[panel_idx]
            
            # Get analysis object for mask resolution
            analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
            analysis = self.get_analysis(analysis_name)
            

            ax1 = axs[panel_idx]
            ax2 = ax1.twinx()
            ax2_list.append(ax2)
            # Process both left (ax1) and right (ax2) metrics
            for axis_idx, (metric_list, ax_obj, side) in enumerate([
                (metric_lists[0], ax1, 'low'),   # Left y-axis
                (metric_lists[1], ax2, 'high')   # Right y-axis
            ]):
                
                for metric_idx, metric_name in enumerate(metric_list):
                    datasets = []
                    v_array = []
                    
                    # Collect data for each v_inf
                    for v in analysis_dict.keys():
                        if 'V' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                            continue
                        v_inf = mc['v_inf']
                        
                        metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and metric_masks[axis_idx]:
                            metric_mask = metric_masks[axis_idx][metric_idx] if metric_idx < len(metric_masks[axis_idx]) else None
                            if metric_mask is not None:
                                mask = self.resolve_mask(analysis, metric_mask, v)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    print(f"Panel {panel_idx}, axis {axis_idx}: Applying mask for '{metric_name}' at v={v}, sum={np.sum(mask)}")
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                        
                        # Process array metrics
                        if isinstance(metric, (list, tuple, np.ndarray)):
                            # Flatten and extract valid values
                            # Trim m from two ends to avoid initial or final outliers

                            if len(metric) > 2:
                                metric = metric[1:-1]
                            if len(metric) == 0:
                                continue
                            data_v = [float(np.mean(m))*unit_change[axis_idx] for m in metric if np.size(m) > 0]
                            datasets.append(data_v)
                            v_array.append(v_inf / 1e3)
                        else:
                            continue
                    
                    if len(datasets) == 0:
                        print(f"Panel {panel_idx}, axis {axis_idx}: No array data for '{metric_name}'")
                        continue
                    
                    # Sort by v_inf
                    v_array = np.asarray(v_array)
                    order = np.argsort(v_array)
                    pos = v_array[order]
                    data_sorted = [datasets[i] for i in order]
                    
                    # Create violin plot
                    vp = ax_obj.violinplot(
                        dataset=data_sorted,
                        positions=pos,
                        showmeans=True,
                        showmedians=False,
                        showextrema=True,
                        widths=4,
                        side=side
                    )
                    
                    # Style violins
                    color_idx = n + m*axis_idx
                    for body in vp['bodies']:
                        body.set_zorder(0)
                        body.set_alpha(0.6)
                        body.set_facecolor(cmap(color_idx))
                    
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        vp[partname].set_zorder(1)
                        vp[partname].set_alpha(0.8)
                        vp[partname].set_edgecolor(cmap(color_idx))
                        if partname == 'cmeans':
                            # vp[partname].set_edgecolor('crimson')
                            vp[partname].set_linewidth(1.5)
                        else:
                            
                            vp[partname].set_linewidth(1)
                    
                    vp['cmeans'].set_zorder(2)
                    vp['cmeans'].set_alpha(1)
                
                # Set log scale for both axes
                ax1.set_yscale(scale[0])
                ax2.set_yscale(scale[1])
            
            # Set log scale for both axes
            ax1.set_yscale(scale[0])
            ax2.set_yscale(scale[1])
            # ax1.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
            # ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
            if panel_idx < nrows - 1:
                ax1.tick_params(labelbottom=False)
                ax2.tick_params(labelbottom=False)
            
            # Add analysis label
            ax2.text(0.95, 0.95, 
                    fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey/const.M_sun.value)} {analysis_zlabel[1]}',
                    transform=ax2.transAxes, rotation=0, ha='right', va='top', fontsize=8)

            if scale[0] == 'linear':
                ax1.yaxis.set_major_formatter('{x:0.1f}')
            if scale[1] == 'linear':
                ax2.yaxis.set_major_formatter('{x:0.1f}')
        # Set xlabel only on bottom panel
            axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')

            margin_factor = 0.7
            if global_ylim:
                # Left axes (semi-major axis)
                ylims_left = ax1.get_ylim()
                global_ylim_left = (
                    min(ylims_left), 
                    max(ylims_left)
                )
                ax1.set_ylim(global_ylim_left)
                
                # Right axes (eccentricity)
                ylims_right = ax2.get_ylim() 
                global_ylim_right = (
                    min(ylims_right), 
                    max(ylims_right)
                )
                ax2.set_ylim(global_ylim_right)

                # Left axes (semi-major axis)
                xlims_left = ax1.get_xlim()
                global_xlim_left = (
                    min(xlims_left)+margin_factor, 
                    max(xlims_left)-margin_factor
                )
                ax1.set_xlim(global_xlim_left)
                
                # Right axes (eccentricity)
                xlims_right = ax2.get_xlim() 
                global_xlim_right = (
                    min(xlims_right)+margin_factor, 
                    max(xlims_right)-margin_factor
                )
                ax2.set_xlim(global_xlim_right)
                    
            adjusted_n = plu.adjust_color(cmap(n), 1.3)  # Darker shade for ticks
            adjusted_nm = plu.adjust_color(cmap(n+m), 1.5)  # Darker shade for ticks
            # ✅ SET Y-AXIS LABELS
            # ax1.set_ylabel(metric_ylabels[0], fontsize=11, color=adjusted_n)
            # ax2.set_ylabel(metric_ylabels[1], fontsize=11, color=adjusted_nm)
            
            # ✅ STYLE SPINES AND TICKS
            ax1.spines['left'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)
        

            # Color spines
            ax1.spines['left'].set_color(cmap(n))
            ax2.spines['right'].set_color(cmap(n+m))

            ax1.minorticks_off()
            ax2.minorticks_off()

            # Color y-axis labels
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)

            ax1.tick_params(axis='y', which='both', colors=adjusted_n, labelcolor=adjusted_n)
            ax2.tick_params(axis='y', which='both', colors=adjusted_nm, labelcolor=adjusted_nm)
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)
        
        # ✅ GLOBAL Y-LIMITS IF REQUESTED
        if global_ylim:
            ylim_left = ax1.get_ylim()
            ylim_right = ax2.get_ylim()
            ax1.set_ylim(ylim_left)
            ax2.set_ylim(ylim_right)
        
        # # ✅ LEGEND
        # if metric_labels:
        #     from matplotlib.patches import Rectangle
        #     handles = []
        #     labels = []
        #     for label in metric_labels[0]:
        #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n), alpha=0.6))
        #         labels.append(label)
        #     for label in metric_labels[1]:
        #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n + m), alpha=0.6))
        #         labels.append(label)
        #     ax1.legend(handles, labels, frameon=False, fontsize=9, loc='upper left')
        
        fig.subplots_adjust(left=0.15, right=0.85, hspace=0,top=0.95, bottom=0.15)
    
        fig.text(0.007, 0.5, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11, color=adjusted_n)
        fig.text(0.99, 0.5, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11, color=adjusted_nm)
        
        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')

    def multiple_analysis_metric_array_wrt_v_twinaxis_violin(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], unit_change=[1e-3, 1, 1], global_ylim=False, metric_masks=None, fig_size=None, save_as=None, save=True):
            """
            Plot array metrics (like semi_major_axes) as violin plots with twin y-axes in multi-panel layout.
            Each panel represents one analysis, with two sets of violins (one per y-axis).
            
            Parameters:
            -----------
            metric_lists : list of lists
                [[left_metrics], [right_metrics]] - metrics for left and right y-axes
            metric_ylabels : list of str
                [left_ylabel, right_ylabel]
            metric_labels : list of lists, optional
                [[left_labels], [right_labels]]
            metric_masks : list of lists, optional
                [[left_masks], [right_masks]] - masks for filtering data
            """
            if analysis_key == 'All':
                analysis_list = list(self.analysis_dicts.values())
            elif isinstance(analysis_key, list):
                analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            
            # Get z-axis values (e.g., mC) for each analysis
            zkey_values = []
            for a in analysis_list:
                vkey = list(a.keys())[0]
                zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
                zkey_values.append(zkey)
            
            # Create multi-panel figure
            nrows = 1
            ncols = 1
            fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 1.5*nrows), nrows=nrows, ncols=ncols, sharex=True)
            
            # Make axs iterable even if single subplot
            if nrows == 1:
                axs = [axs]
            
            cmap = plt.get_cmap('plasma', 15)
            n, m = 3, 8
            ax1 = axs[0]
            ax2 = ax1.twinx()

                # Process both left (ax1) and right (ax2) metrics
            for axis_idx, (metric_list, ax_obj, side) in enumerate([
                (metric_lists[0], ax1, 'low'),   # Left y-axis
                (metric_lists[1], ax2, 'high')   # Right y-axis
            ]):
                for metric_idx, metric_name in enumerate(metric_list):
                    datadict = {}
                    # Process each analysis
                    for panel_idx, analysis_dict in enumerate(analysis_list):
                        zkey = zkey_values[panel_idx]
                        
                        # Get analysis object for mask resolution
                        analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                        analysis = self.get_analysis(analysis_name)
                        
                        # Collect data for each v_inf
                        for v in analysis_dict.keys():
                            if 'V' not in v:
                                continue

                            entry = analysis_dict[v]
                            mc = entry.get('mc')
                            if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                                continue
                            v_inf = mc['v_inf']

                            if v not in datadict.keys():
                                datadict[v] = {'v_inf': v_inf*unit_change[0], 'data': []}
                            
                            metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                            
                            # Apply mask if provided
                            if metric_masks is not None and metric_masks[axis_idx]:
                                metric_mask = metric_masks[axis_idx][metric_idx] if metric_idx < len(metric_masks[axis_idx]) else None
                                if metric_mask is not None:
                                    mask = self.resolve_mask(analysis, metric_mask, v)
                                    if mask is not None:
                                        mask = np.asarray(mask)
                                        print(f"Panel {panel_idx}, axis {axis_idx}: Applying mask for '{metric_name}' at v={v}, sum={np.sum(mask)}")
                                        metric = [m for j, m in enumerate(metric) if mask[j]]
                            
                            # Process array metrics
                            if isinstance(metric, (list, tuple, np.ndarray)):
                                # Flatten and extract valid values
                                # Trim m from two ends to avoid initial or final outliers
                                if len(metric) > 2:
                                    metric = metric[1:-1]
                                if len(metric) == 0:
                                    continue
                                metric = [m*unit_change[1+axis_idx] for m in metric]
                                datadict[v]['data'].extend(metric)
                            else:
                                continue
                        
                            if len(datadict[v]['data']) == 0:
                                print(f"v {v}, panel {panel_idx}, axis {axis_idx}: No array data for '{metric_name}'")
                                continue
                        
                    # Sort by v_inf
                    v_array = np.asarray([datadict[v]['v_inf'] for v in datadict])
                    datasets = [datadict[v]['data'] for v in datadict]
                    order = np.argsort(v_array)
                    pos = v_array[order]
                    data_sorted = [datasets[i] for i in order]
                    
                    # Create violin plot
                    vp = ax_obj.violinplot(
                        dataset=data_sorted,
                        positions=pos,
                        showmeans=True,
                        showmedians=False,
                        showextrema=True,
                        widths=4,
                        side=side
                    )
                    
                    # Style violins
                    color_idx = n + m*axis_idx
                    for body in vp['bodies']:
                        body.set_zorder(0)
                        body.set_alpha(0.6)
                        body.set_facecolor(cmap(color_idx))
                    
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        vp[partname].set_zorder(1)
                        vp[partname].set_alpha(0.8)
                        vp[partname].set_edgecolor(cmap(color_idx))
                        if partname == 'cmeans':
                            # vp[partname].set_edgecolor('crimson')
                            vp[partname].set_linewidth(1.5)
                        else:
                            
                            vp[partname].set_linewidth(1)
                    
                    vp['cmeans'].set_zorder(2)
                    vp['cmeans'].set_alpha(1)
                
                # Set log scale for both axes
                ax1.set_yscale(scale[0])
                ax2.set_yscale(scale[1])


                # ax1.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
                # ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
                if panel_idx < nrows - 1:
                    ax1.tick_params(labelbottom=False)
                    ax2.tick_params(labelbottom=False)
                
                # Add analysis label
                # ax2.text(0.1, 0.1, 
                #         fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                #         transform=ax2.transAxes, rotation=0, va='center', fontsize=11)

                if scale[0] == 'linear':
                    ax1.yaxis.set_major_formatter('{x:0.1f}')
                if scale[1] == 'linear':
                    ax2.yaxis.set_major_formatter('{x:0.1f}')
            # Set xlabel only on bottom panel
            axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')

            margin_factor = 0.7
            if global_ylim:
                # Left axes (semi-major axis)
                ylims_left = ax1.get_ylim()
                global_ylim_left = (
                    min(ylims_left), 
                    max(ylims_left)
                )
                ax1.set_ylim(global_ylim_left)
                
                # Right axes (eccentricity)
                ylims_right = ax2.get_ylim() 
                global_ylim_right = (
                    min(ylims_right), 
                    max(ylims_right)
                )
                ax2.set_ylim(global_ylim_right)

                # Left axes (semi-major axis)
                xlims_left = ax1.get_xlim()
                global_xlim_left = (
                    min(xlims_left)+margin_factor, 
                    max(xlims_left)-margin_factor
                )
                ax1.set_xlim(global_xlim_left)
                
                # Right axes (eccentricity)
                xlims_right = ax2.get_xlim() 
                global_xlim_right = (
                    min(xlims_right)+margin_factor, 
                    max(xlims_right)-margin_factor
                )
                ax2.set_xlim(global_xlim_right)
                    
            adjusted_n = plu.adjust_color(cmap(n), 1.3)  # Darker shade for ticks
            adjusted_nm = plu.adjust_color(cmap(n+m), 1.5)  # Darker shade for ticks
            # ✅ SET Y-AXIS LABELS
            ax1.set_ylabel(metric_ylabels[0], fontsize=11, color=adjusted_n)
            ax2.set_ylabel(metric_ylabels[1], fontsize=11, color=adjusted_nm)
            
            # ✅ STYLE SPINES AND TICKS
            ax1.spines['left'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)
        

            # Color spines
            ax1.spines['left'].set_color(cmap(n))
            ax2.spines['right'].set_color(cmap(n+m))

            ax1.minorticks_off()
            ax2.minorticks_off()

            # Color y-axis labels
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)

            ax1.tick_params(axis='y', which='both', colors=adjusted_n, labelcolor=adjusted_n)
            ax2.tick_params(axis='y', which='both', colors=adjusted_nm, labelcolor=adjusted_nm)
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)
            
            # ✅ GLOBAL Y-LIMITS IF REQUESTED
            if global_ylim:
                ylim_left = ax1.get_ylim()
                ylim_right = ax2.get_ylim()
                ax1.set_ylim(ylim_left)
                ax2.set_ylim(ylim_right)
            
            # # ✅ LEGEND
            # if metric_labels:
            #     from matplotlib.patches import Rectangle
            #     handles = []
            #     labels = []
            #     for label in metric_labels[0]:
            #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n), alpha=0.6))
            #         labels.append(label)
            #     for label in metric_labels[1]:
            #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n + m), alpha=0.6))
            #         labels.append(label)
            #     ax1.legend(handles, labels, frameon=False, fontsize=9, loc='upper left')
            
            fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)
    
            if save:
                metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
                figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v.png'
                out = os.path.join(self.plots_dir, figname)
                fig.savefig(out, dpi=300, bbox_inches='tight')


    def multiple_analysis_metric_gaussian_kde_multipanel(self, metric_xnames, metric_ynames, metric_xlabel, metric_ylabel, cmaps, reverse_axes=None, ticks=None, percentage_annotations=None, add_negative_top_axis=False, cut=0.0, time_stamp=None, analysis_key='All', analysis_zkey='mC', mean_annotations=False, analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['linear', 'linear', 'linear'], unit_change=[1, 1, 1], ranges=None, global_ylim=False, metric_mask=None, bw_adjust=0.7, thresh=0.1, plot_outliers=True, fig_size=None, save_as=None, save=True):
        """
        Plot array metrics as Gaussian KDE per v per analysis in multi-panel layout.
        Optionally overlay scatter points for outliers (points outside KDE contours).
        
        Parameters:
        -----------
        metric_xname, metric_yname : str
            Names of x and y metrics
        N : complex
            Grid resolution (e.g., 100j for 100x100 grid)
        bw_adjust : float
            Bandwidth adjustment factor (default 0.7)
        thresh : float
            Threshold for KDE contours (0.1 = 10\% of max density)
        plot_outliers : bool
            Whether to plot scatter points outside KDE contours
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 10  # One column per v_inf value
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 7), nrows=nrows, ncols=ncols, sharex=True)
        
        if nrows == 1:
            axs = [axs]
        cc1, cc2 = None, None 
        cs = [cc1, cc2]   
        for n,cmap in enumerate(cmaps):
            cs[n] = plt.get_cmap(cmap)(0.9)
        # Process each analysis
        for row_idx, analysis_dict in enumerate(analysis_list):
            analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
            analysis = self.get_analysis(analysis_name)
             
            for i in range(len(metric_xnames)):
                metric_xname = metric_xnames[i]
                metric_yname = metric_ynames[i]
                cmap = cmaps[i] if i < len(cmaps) else 'viridis'
                cc = plt.get_cmap(cmap)(0.9)
                v_keys = [vv for vv in analysis_dict.keys() if 'V' in vv]
                zkey = self._get_metric_from_sources(analysis_dict, None, analysis_zkey)
                for col_idx, v in enumerate(v_keys):
                    
                    if col_idx >= ncols:
                        break
                    
                    v_inf = self._get_metric_from_sources(analysis_dict, v, 'v_inf') 
                    metric_x = self._get_metric_from_sources(analysis_dict, v, metric_xname)
                    metric_y = self._get_metric_from_sources(analysis_dict, v, metric_yname)

                    if time_stamp:
                        times = self._get_metric_from_sources(analysis_dict, v, 'times')
                    
                    # Apply mask if provided
                    if metric_mask is not None:
                        if metric_mask[i] is not None:
                            mask = self.resolve_mask(analysis, metric_mask[i], v)
                            mask = np.asarray(mask)
                            print(f"Row {row_idx}, col {col_idx}: Applying mask (sum={np.sum(mask)}), len={len(mask)} to x_array {len(metric_x)}, y_array {len(metric_y)}")
                            metric_x = [m for j, m in enumerate(metric_x) if mask[j]]
                            metric_y = [m for j, m in enumerate(metric_y) if mask[j]]
                            times = [t for j, t in enumerate(times) if mask[j]]
                    # Process metrics
                    def process_metric(m, unit_factor, n=None, unit_factor_n=None):
                        if time_stamp:
                            skipped = 0
                            filtered_m = []
                            filtered_n = []
                            # If time_stamp is provided, filter metric values to only include the closest time point to the specified time_stamp
                            for mm, time, nn in zip(m, times, n):
                                time = np.asarray(time)
                                time_stamp_array = np.full_like(time, time_stamp)
                                time_diffs = np.abs(time - time_stamp_array)
                                # print(f"  Time filtering: {len(time_diffs)} time differences calculated")
                                if len(time_diffs) == 0:
                                    skipped += 1
                                    continue
                                min_time_diff = np.min(time_diffs)
                                if min_time_diff <= 6:  # Only consider values within 6 years of the time_stamp
                                    closest_idx = np.argmin(time_diffs)
                                    mm_closest = mm[closest_idx]
                                    nn_closest = nn[closest_idx] if n is not None else None
                                    # print(f"  Time filtering: using closest with time_diff={min_time_diff:.2e}")
                                    filtered_m.append(float(mm_closest * unit_factor))
                                    filtered_n.append(float(nn_closest * unit_factor_n))
                                # time_diffs = [abs(t - time_stamp) for t in time if np.isclose(t, time_stamp, rtol=0, atol=6)]
                                # if len(time_diffs) > 0:
                                #     closest_idx = np.argmin(time_diffs)
                                #     mm = mm[closest_idx]
                                #     nn = nn[closest_idx] if n is not None else None
                                #     print(f"  Time filtering: Found {len(time_diffs)} values close to time_stamp={time_stamp}, using closest with time_diff={time_diffs[closest_idx]:.2e}")
                                #     filtered_m.append(float(mm * unit_factor))
                                #     filtered_n.append(float(nn * unit_factor_n)) 
                            print(f"  Time filtering: {skipped} points skipped")
                            return filtered_m, filtered_n
                        if isinstance(m, (list, tuple)):
                            if len(m) > 2:
                                m = m[1:-1]
                            if len(m) == 0:
                                return []
                            return [float(mm * unit_factor) for mm in m if np.size(mm) > 0]
                        elif isinstance(m, np.ndarray) and m.ndim > 0:
                            return (m.flatten() * unit_factor).tolist()
                        else:
                            return [float(m * unit_factor)]
                    
                    if time_stamp:
                        data_x, data_y = process_metric(metric_x, unit_change[0], n=metric_y, unit_factor_n=unit_change[1])
                        
                    else:
                        data_x = process_metric(metric_x, unit_change[0])
                        data_y = process_metric(metric_y, unit_change[1])
                        
                    x_array = np.array(data_x)
                    y_array = np.array(data_y)
        
                    # Filter invalid
                    valid_mask = np.isfinite(x_array) & np.isfinite(y_array) & (y_array > 0)
                    x_array = x_array[valid_mask]
                    y_array = y_array[valid_mask]
                    
                    print(f"  {len(x_array)} valid points")
                    
                    if len(x_array) == 0:
                        print(f"  No valid data for v={v}, skipping subplot")
                        continue

                    if add_negative_top_axis:
                        from matplotlib.ticker import FuncFormatter

                        ax_top = axs[row_idx, col_idx].secondary_xaxis('top', functions=(lambda x: -x, lambda x: -x))

                        ax_top.set_xlabel(f'', fontsize=9)

                        # if ticks and ticks[0] is not None:
                        #     ax_top.set_xticks([-t for t in ticks[0]])

                        ax_top.tick_params(labeltop=False)
                        # if row_idx > 0:
                        #     ax_top.tick_params(labeltop=False)
                        # else:
                        #     ax_top.tick_params(axis='x', colors=cc1)
                        #     ax_top.spines['top'].set_color(cc1)

                    if reverse_axes and add_negative_top_axis:  
                        # Reverse x-axis if it is plotted on top x axis
                        x_array = -x_array if reverse_axes[i][0] else x_array
                        y_array = -y_array if reverse_axes[i][1] else y_array
                        print(f"  Reversed axes: x reversed={reverse_axes[i][0]}, y reversed={reverse_axes[i][1]}")

                    if ranges is not None:
                        x_mask = np.ones_like(x_array, dtype=bool)
                        y_mask = np.ones_like(y_array, dtype=bool)  
                        if ranges[0] is not None:
                            x_rng = (ranges[0][0]*unit_change[0], ranges[0][1]*unit_change[0])
                            axs[row_idx, col_idx].set_xlim(x_rng)
                            x_mask = (x_array >= x_rng[0]) & (x_array <= x_rng[1])
                        if ranges[1] is not None:
                            y_rng = (ranges[1][0]*unit_change[1], ranges[1][1]*unit_change[1])
                            axs[row_idx, col_idx].set_ylim(y_rng)
                            y_mask = (y_array >= y_rng[0]) & (y_array <= y_rng[1])

                        x_array = x_array[x_mask & y_mask]
                        y_array = y_array[y_mask & x_mask]

                    df = pd.DataFrame({metric_xname: x_array, metric_yname: y_array})
                    # ============ GAUSSIAN KDE ============
                    # KDE parameters
                    kde_kw = dict(
                        fill=True,
                        levels=15,
                        thresh=thresh,
                        bw_adjust=bw_adjust,
                        common_norm=False,
                        cut=cut,
                        label=None,
                    )
                    sns.kdeplot(x=metric_xname, y=metric_yname, data=df, ax=axs[row_idx, col_idx], cmap=cmap, **kde_kw)
                    axs[row_idx, col_idx].set_ylabel(f'', fontsize=9)
                    axs[row_idx, col_idx].set_xlabel(f'', fontsize=9)
                    kde = gaussian_kde(np.vstack([x_array, y_array]), bw_method='scott')
                    kde.set_bandwidth(kde.factor * kde_kw['bw_adjust'])  # ← Match your bw_adjust=0.7
                    density = kde(np.vstack([x_array, y_array]))
                    threshold = kde_kw['thresh'] * density.max()
                    print(f"  {np.sum(density < threshold)} outliers detected")
                    mask_outliers = density < threshold

                    
                    axs[row_idx, col_idx].scatter(x_array[mask_outliers], y_array[mask_outliers], color=cc, s=2, edgecolor=cc, linewidth=0.5, zorder=10, alpha=0.8)

                    # ============ FORMATTING ============

                    if scale[1] == 'log':
                        axs[row_idx, col_idx].set_yscale('log')
                    if scale[0] == 'log':
                        axs[row_idx, col_idx].set_xscale('log')

                    if ticks:
                        axs[row_idx, col_idx].set_xticks(ticks[0] if ticks and ticks[0] else None)
                        axs[row_idx, col_idx].set_yticks(ticks[1] if ticks and ticks[1] else None)
                    if row_idx < nrows - 1:
                        axs[row_idx, col_idx].tick_params(labelbottom=False)
                    # else:
                    #     axs[row_idx, col_idx].tick_params(axis='x', colors=cc2)

                    # Add colorbar on top row
                    if col_idx > 0:
                        axs[row_idx, col_idx].tick_params(labelleft=False)       

                    if percentage_annotations:
                        within_percentage = np.sum([x_mask & y_mask]) / len(x_mask) * 100 if ranges is not None and ranges[0] is not None else 100
                        axs[row_idx, col_idx].annotate(f'{np.round(within_percentage, 0):.0f}\\%', xy=(0.8, 0.3+0.2*i), xycoords='axes fraction', ha='right', va='bottom', fontsize=9, color=cc)
                    
                    if mean_annotations and i == 1:
                        metric_xmean = np.mean(x_array)
                        axs[row_idx, col_idx].annotate(r'$\bar{x}$=' + f'{np.round(metric_xmean, 0):.0f}', xy=(0.05, 0.05+0.1*i), xycoords='axes fraction', ha='left', va='bottom', fontsize=6, color=cc)
                        metric_ymean = np.mean(y_array)
                        axs[row_idx, col_idx].annotate(r'$\bar{y}$=' + f'{np.round(metric_ymean, 2):.2f}', xy=(0.95, 0.05+0.1*i), xycoords='axes fraction', ha='right', va='bottom', fontsize=6, color=cc)
                    if time_stamp:
                        survivor_percentage_at_time = len(x_array) / len(metric_x) * 100 if len(metric_x) > 0 else 0
                        axs[row_idx, col_idx].annotate(f'{np.round(survivor_percentage_at_time, 0):.0f}\\% ', xy=(0.8, 0.4), xycoords='axes fraction', ha='right', va='bottom', fontsize=9, color='black')

                    if row_idx < nrows - 1:
                        axs[row_idx, col_idx].tick_params(labelbottom=False)
                    print(f'  ✓ Finished subplot')
                    if row_idx == 0:
                        v_inf = np.round(v_inf * 1e-3, 1)  # Convert to km/s and round
                        v_inf_label = f'{v_inf:.0f}'
                        # axs[row_idx, col_idx].xaxis.set_label_position("top")  # ✅ Explicitly set top position
                        axs[row_idx, col_idx].set_title(rf'{v_inf_label}', fontsize=9, rotation=0)
                    if col_idx == ncols - 1:
                        zkey = zkey / const.M_sun.value if zkey is not None else None
                        axs[row_idx, col_idx].yaxis.set_label_position("right")  # ✅ Explicitly set right position
                        axs[row_idx, col_idx].set_ylabel(fr'{plu.sci_notation_latex(zkey)}', fontsize=9, rotation=0, labelpad=13, va='center')

        # ============ LABELS ============
        fig.supylabel(metric_ylabel, fontsize=11, x=0.01)
        fig.supxlabel(metric_xlabel, fontsize=11, y=0.01)
        
        # Add figure-level annotations
        fig.text(0.5, 0.98, r'v$_{\infty}$ [km s$^{-1}$]',  
                fontsize=11, ha='center', va='top')

        fig.text(0.99, 0.5, r'M$_{PBH}$ [M$_{\odot}$]', 
                fontsize=11, ha='center', va='center', rotation=270)


        fig.subplots_adjust(hspace=0.0,wspace=0.0, left=0.08, right=0.93, top=0.90, bottom=0.1)
        if save:
            metricsname = "_".join([plu.latex_label_key(label) for label in [metric_ylabel, metric_xlabel]])
            figname = save_as if save_as else f'multiple_analysis_{metricsname}_gaussian_kde_outliers.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=500, bbox_inches='tight')
            print(f'✓ Saved: {figname}')

    def single_analysis_corner_plot(self, metric_names, metric_labels,analysis_name=None, v_key='All', metric_masks=None, scale='linear', unit_change=None, ranges=None, quantiles=[0.16, 0.5, 0.84], plot_datapoints=True, plot_density=True, plot_contours=True, fig_size=None, save_as=None, save=True):
        """
        Create a corner plot for a single analysis using corner.corner().
        Shows 1D distributions on diagonal and 2D scatter/contours on off-diagonal.
        
        Parameters:
        -----------
        metric_names : list of str
            Names of metrics to include in corner plot (e.g., ['a_au', 'e', 'semi_major_axes'])
        metric_labels : list of str
            Display labels for metrics (e.g., ['a [AU]', 'e', 'a (average) [AU]'])
        analysis_name : str
            Name of analysis to plot. If None, uses first analysis.
        v_key : str or list
            Velocity key(s) to include. 'All' uses all v bins, aggregate across v_inf.
        metric_masks : list of dicts, optional
            Masks to apply to each metric for filtering
        scale : str or list
            'linear' or 'log' for each axis
        unit_change : list, optional
            Unit conversion factors [metric1_factor, metric2_factor, ...]
        ranges : list of tuples, optional
            [(x_min, x_max), (y_min, y_max), ...] for each metric
        quantiles : list
            Quantiles to show on 1D distributions (default: [0.16, 0.5, 0.84])
        plot_datapoints : bool
            Show scatter points in 2D panels
        plot_density : bool
            Show KDE contours in 2D panels
        plot_contours : bool
            Show contour lines
        fig_size : tuple, optional
            Figure size (inches)
        save_as : str, optional
            Custom filename for saving
        save : bool
            Whether to save figure
        """
        try:
            import corner
        except ImportError:
            print("Error: corner package not installed. Install with: pip install corner")
            return
        
        # Get analysis
        if analysis_name is None:
            analysis_name = list(self.analysis_dicts.keys())[0]
        
        analysis_dict = self.analysis_dicts.get(analysis_name)
        if analysis_dict is None:
            print(f"Analysis '{analysis_name}' not found")
            return
        
        analysis = self.get_analysis(analysis_name)
        
        # Default unit changes
        if unit_change is None:
            unit_change = [1] * len(metric_names)
        
        # Get v_keys
        if v_key == 'All':
            v_keys = sorted([k for k in analysis_dict.keys() if isinstance(k, str) and k.startswith('V')],
                        key=lambda x: int(x[1:]))
        else:
            v_keys = [v_key] if isinstance(v_key, str) else v_key
        
        # Collect data for all metrics
        data_dict = {name: [] for name in metric_names}
        
        for v in v_keys:
            for metric_idx, metric_name in enumerate(metric_names):
                metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                
                # Apply mask if provided
                if metric_masks is not None and metric_masks[metric_idx] is not None:
                    mask = self.resolve_mask(analysis, metric_masks[metric_idx], v)
                    mask = np.asarray(mask)
                    metric = [m for j, m in enumerate(metric) if mask[j]]
                
                # Process metric (flatten if needed)
                if isinstance(metric, (list, tuple)):
                    filtered = [float(m * unit_change[metric_idx]) for m in metric if np.size(m) > 0]
                elif isinstance(metric, np.ndarray) and metric.ndim > 0:
                    filtered = (metric.flatten() * unit_change[metric_idx]).tolist()
                else:
                    filtered = [float(metric * unit_change[metric_idx])] if np.isfinite(metric) else []
                
                data_dict[metric_name].extend(filtered)
        
        # Convert to numpy array (N samples × M dimensions)
        samples = np.column_stack([np.array(data_dict[name]) for name in metric_names])
        
        # Filter invalid values
        valid_mask = np.all(np.isfinite(samples), axis=1)
        samples = samples[valid_mask]

        range_limits = []
        for i in range(samples.shape[1]):
            if ranges[i] is not None:
                range_limits.append(ranges[i])
            else:
                col = samples[:, i]
                min_val, max_val = np.min(col), np.max(col)
                range_limits.append((min_val, max_val)) 
        
        print(f"Corner plot for {analysis_name}: {len(samples)} valid samples, {len(metric_names)} dimensions")
        
        fig = corner.corner(
            samples,
            labels=metric_labels,
            labelsize=12,               # ← Axis label font size
            quantiles=quantiles,
            show_titles=True,
            title_kwargs={'fontsize': 8},    # ← Title font size
            label_kwargs={'fontsize': 12},     # ← Label font size
            color=plt.get_cmap('Oranges')(0.8),
            title_fmt='.2f',
            plot_datapoints=plot_datapoints,
            plot_density=plot_density,
            plot_contours=plot_contours,
            range=range_limits,
            verbose=True
        )
        
        # plt.suptitle(f'{analysis_name}', fontsize=14, y=0.995)
        fig.set_size_inches(fig_size) 
        fig.subplots_adjust(hspace=0.0,wspace=0.0)
        
        if save:
            metricsname = "_".join([plu.latex_label_key(label) for label in metric_labels])
            figname = save_as if save_as else f'{analysis_name}_corner_{metricsname}.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'✓ Saved: {figname}')
        
        return fig
    def multiple_analysis_metric_timeseries_gaussian_kde_multipanel(self, metric_xnames, metric_ynames, metric_xlabel, metric_ylabel, levels, cmaps, dt, reverse_axes=None, ticks=None, percentage_annotations=None, add_negative_top_axis=False, cut=0.0, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['linear', 'linear', 'linear'], unit_change=[1, 1, 1], ranges=None, global_ylim=False, metric_mask=None, bw_adjust=0.7, thresh=0.1, plot_outliers=True, fig_size=None, save_as=None, save=True):
        """
        Plot array metrics as Gaussian KDE per v per analysis in multi-panel layout.
        Optionally overlay scatter points for outliers (points outside KDE contours).
        
        Parameters:
        -----------
        metric_xname, metric_yname : str
            Names of x and y metrics
        N : complex
            Grid resolution (e.g., 100j for 100x100 grid)
        bw_adjust : float
            Bandwidth adjustment factor (default 0.7)
        thresh : float
            Threshold for KDE contours (0.1 = 10\\% of max density)
        plot_outliers : bool
            Whether to plot scatter points outside KDE contours
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 1  # One column per v_inf value
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 7), nrows=nrows, ncols=ncols, sharex=True)

        if ncols == 1:
            axs = axs[:, np.newaxis]

        # Process each analysis
        for row_idx, analysis_dict in enumerate(analysis_list):
            analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
            analysis = self.get_analysis(analysis_name)
            col_idx = 0  # Since ncols=1, we only have one column
            for i in range(len(metric_xnames)):
                metric_xname = metric_xnames[i]
                metric_yname = metric_ynames[i]
                
                v_keys = [vv for vv in analysis_dict.keys() if 'V' in vv]
                for c, v in enumerate(v_keys):
                    # if col_idx >= ncols:
                    #     break
                    cmap = cmaps[i] if i < len(cmaps) else 'viridis'
                    cc = plt.get_cmap(cmap)(0.9)

                    metric_x = self._get_metric_from_sources(analysis_dict, v, metric_xname)
                    metric_y = self._get_metric_from_sources(analysis_dict, v, metric_yname)
        
                    # Apply mask if provided
                    if metric_mask is not None:
                        mask = self.resolve_mask(analysis, metric_mask, v)
                        if mask is not None:
                            mask = np.asarray(mask)
                            print(f"Row {row_idx}, col {c}: Applying mask (sum={np.sum(mask)})")
                            metric_x = [m for j, m in enumerate(metric_x) if mask[j]]
                            metric_y = [m for j, m in enumerate(metric_y) if mask[j]]
            
                    # Process metrics
                    # def process_timeseries_metric(time_series, metric_series, unit_change):
                    #     all_times = []
                    #     all_metrics = []
                    #     for times, metrics in zip(time_series, metric_series):
                    #         if times is None or metrics is None:
                    #             continue
                    #         if len(times) == 0 or len(metrics) == 0:
                    #             continue

                    #         # Flatten arrays
                    #         downsample_factor = max(1, len(times) // 200)  # Adjust factor to get ~100 points
                    #         times_flat = np.asarray(times).flatten()[::downsample_factor]
                    #         metrics_flat = np.asarray(metrics).flatten()[::downsample_factor]

                    #         metrics_flat = metrics_flat * unit_change[1]
                    #         times_flat = times_flat * unit_change[0]

                    #         # Ensure same length
                    #         min_len = min(len(times_flat), len(metrics_flat))
                    #         if min_len == 0:
                    #             continue

                    #         all_times.extend(times_flat[:min_len])
                    #         all_metrics.extend(metrics_flat[:min_len])
                    #     return all_times, all_metrics


                    data_x, data_y = self.process_timeseries_metric(metric_x, metric_y, dt_fixed=dt)

                    x_array = np.array(data_x)
                    y_array = np.array(data_y)
        
                    # Filter invalid
                    valid_mask = np.isfinite(x_array) & np.isfinite(y_array) & (y_array > 0)
                    x_array = x_array[valid_mask]
                    y_array = y_array[valid_mask]
                    
                    print(f"  {len(x_array)} valid points")
                    
                    if len(x_array) == 0:
                        continue

                    if scale[0] == 'log':
                        # Take log of x-values for KDE computation
                        x_array_kde = np.log10(x_array)  # or np.log() for natural log
                        print(f"  Using log-transformed x-axis for KDE")
                    else:
                        x_array_kde = x_array

                    # Apply range filtering (still in original space for limits)
                    if ranges is not None:
                        x_mask = np.ones_like(x_array, dtype=bool)
                        y_mask = np.ones_like(y_array, dtype=bool)  
                        if ranges[0] is not None:
                            x_rng = (ranges[0][0]*unit_change[0], ranges[0][1]*unit_change[0])
                            axs[row_idx, col_idx].set_xlim(x_rng)
                            x_mask = (x_array >= x_rng[0]) & (x_array <= x_rng[1])
                        if ranges[1] is not None:
                            y_rng = (ranges[1][0]*unit_change[1], ranges[1][1]*unit_change[1])
                            axs[row_idx, col_idx].set_ylim(y_rng)
                            y_mask = (y_array >= y_rng[0]) & (y_array <= y_rng[1])

                        x_array = x_array[x_mask & y_mask]
                        y_array = y_array[y_mask & x_mask]
                        x_array_kde = x_array_kde[x_mask & y_mask]  # ✅ Apply mask to log-transformed data too

                    # Create DataFrame with log-transformed x for seaborn
                    df = pd.DataFrame({metric_xname: x_array_kde, metric_yname: y_array})

                    # ============ GAUSSIAN KDE ============
                    kde_kw = dict(
                        fill=True,
                        levels=levels,
                        thresh=thresh,
                        bw_adjust=bw_adjust,
                        common_norm=False,
                        cut=cut,
                        label=None,
                    )

                    # Plot KDE (seaborn will use log-transformed x)
                    sns.kdeplot(x=metric_xname, y=metric_yname, data=df, ax=axs[row_idx, col_idx], cmap=cmap, **kde_kw)

                    # Compute density for outlier detection (using log-transformed x)
                    kde = gaussian_kde(np.vstack([x_array_kde, y_array]), bw_method='scott')
                    kde.set_bandwidth(kde.factor * kde_kw['bw_adjust'])
                    density = kde(np.vstack([x_array_kde, y_array]))
                    threshold = kde_kw['thresh'] * density.max()
                    mask_outliers = density < threshold

                    # ✅ Plot outliers using ORIGINAL x-values (not log-transformed)
                    axs[row_idx, col_idx].scatter(x_array[mask_outliers], y_array[mask_outliers], 
                                                color=cc, s=2, edgecolor=cc, linewidth=0.5, 
                                                zorder=10, alpha=0.8)


                    df = pd.DataFrame({metric_xname: x_array, metric_yname: y_array})
                    within_percentage = np.sum([x_mask & y_mask]) / len(x_mask) * 100 if ranges is not None and ranges[0] is not None else 100
                    # ============ GAUSSIAN KDE ============
                    # KDE parameters
                    kde_kw = dict(
                        fill=True,
                        levels=levels,
                        thresh=thresh,
                        bw_adjust=bw_adjust,
                        common_norm=False,
                        cut=cut,
                        label=None,
                    
                    )
                    sns.kdeplot(x=metric_xname, y=metric_yname, data=df, ax=axs[row_idx, col_idx], cmap=cmap, **kde_kw)
                    axs[row_idx, col_idx].set_ylabel(f'', fontsize=9)
                    axs[row_idx, col_idx].set_xlabel(f'', fontsize=9)
                    kde = gaussian_kde(np.vstack([x_array, y_array]), bw_method='scott')
                    kde.set_bandwidth(kde.factor * kde_kw['bw_adjust'])  # ← Match your bw_adjust=0.7
                    density = kde(np.vstack([x_array, y_array]))
                    threshold = kde_kw['thresh'] * density.max()
                    print(f"  {np.sum(density < threshold)} outliers detected")
                    mask_outliers = density < threshold

                    
                    axs[row_idx, col_idx].scatter(x_array[mask_outliers], y_array[mask_outliers], color=cc, s=2, edgecolor=cc, linewidth=0.5, zorder=10, alpha=0.8)

                    # ============ FORMATTING ============

                    if scale[1] == 'log':
                        axs[row_idx, col_idx].set_yscale('log')
                    if scale[0] == 'log':
                        axs[row_idx, col_idx].set_xscale('log')

                    if ticks:
                        axs[row_idx, col_idx].set_xticks(ticks[0] if ticks and ticks[0] else None)
                        axs[row_idx, col_idx].set_yticks(ticks[1] if ticks and ticks[1] else None)
                    if row_idx < nrows - 1:
                        axs[row_idx, col_idx].tick_params(labelbottom=False)

                    # Add colorbar on top row
                    if col_idx > 0:
                        axs[row_idx, col_idx].tick_params(labelleft=False)       

                    if percentage_annotations:
                        axs[row_idx, col_idx].annotate(f'{np.round(within_percentage, 0):.0f}\\%', xy=(0.8, 0.2+0.2*i), xycoords='axes fraction', ha='right', va='bottom', fontsize=8, color=cc)
                    
 

        # ============ LABELS ============
        fig.supylabel(metric_ylabel, fontsize=11)
        fig.supxlabel(metric_xlabel, fontsize=11)
        

        fig.subplots_adjust(hspace=0.0,wspace=0.0, left=0.1, right=0.99, top=0.95, bottom=0.15)
        if save:
            metricsname = "_".join([plu.latex_label_key(label) for label in [metric_ylabel, metric_xlabel]])
            figname = save_as if save_as else f'multiple_analysis_{metricsname}_gaussian_kde_outliers.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'✓ Saved: {figname}')


    def multiple_analysis_time_series_2dhistogram(self, metric_name, metric_ylabel, time_ylabel, metric_mask=None, ticks=None, dt_fixed=1000, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'linear', 'log'], v_key='All', global_ylim=True, normalize=False, metric_range=None, time_range=None, v_lim=[None, None], time_bins=1000, metric_bins=1000, unit_change=[1, 1, 1], use_existing_time_series=True, fig_size=None, save=True, save_as=None):
        """
        Plot 2D histogram of time series metric in multi-panel layout.
        Each panel represents one analysis system.
        
        Parameters:
        -----------
        analysis_key : str or list
            'All' or list of analysis names
        metric_name : str
            Name of the metric array (e.g., 'semi_major_axes', 'eccentricities')
        metric_ylabel : str
            Label for metric axis
        time_ylabel : str
            Label for time axis
        v_key : str
            'All' to include all velocity bins
        time_bins : int
            Number of bins for time axis
        metric_bins : int
            Number of bins for metric axis
        time_range : tuple, optional
            (min, max) for time axis
        metric_range : tuple, optional
            (min, max) for metric axis
        cmap_name : str
            Colormap name
        norm : str
            'log' or 'linear' for colorbar normalization
        vmax : float, optional
            Maximum value for colorbar
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 10
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 2*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
        # Get colormap
        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))

        all_histograms = []  # [(h, xedges, yedges), ...]
        all_data = []  # [(time_array, metric_array), ...]
        # Process each analysis
        # Process each analysis (row)
        for panel_idx, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[panel_idx]
            
            v_keys = [vv for vv in analysis_dict.keys() if 'V' in vv]
            
            # ✅ STEP 1: Collect ALL histograms for this row

            
            for col_idx, v in enumerate(v_keys):
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                    all_histograms.append(None)
                    continue
                
                # Get time series arrays
                time_series = self._get_metric_from_sources(analysis_dict, v, 'times')
                metric_series = self._get_metric_from_sources(analysis_dict, v, metric_name)
                
                if time_series is None or metric_series is None:
                    all_histograms.append(None)
                    continue
                
                # Apply mask if needed
                if metric_mask is not None:
                    analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                    analysis = self.get_analysis(analysis_name)
                    mask = self.resolve_mask(analysis, metric_mask, v)
                    if mask is not None:
                        mask = np.asarray(mask)
                        metric_series = [m for j, m in enumerate(metric_series) if mask[j]]
                        time_series = [m for j, m in enumerate(time_series) if mask[j]]
                
                # Process time series
                all_times, all_metrics = self.process_timeseries_metric(
                    time_series=time_series, 
                    metric_series=metric_series, 
                    dt_fixed=dt_fixed
                )
                
                if len(all_times) == 0:
                    all_histograms.append(None)
                    continue
                
                # Convert to arrays
                time_array = np.concatenate(all_times)
                metric_array = np.concatenate(all_metrics)
                
                # Determine ranges
                time_range_used = time_range if time_range else (time_array.min(), time_array.max())
                metric_range_used = metric_range if metric_range else (metric_array.min(), metric_array.max())
                
                bin_width = dt_fixed
                time_bins_array = np.arange(time_range_used[0], time_range_used[1] + bin_width, bin_width)

                valid_mask = np.isfinite(time_array) & np.isfinite(metric_array) & (metric_array > 0)
                time_array = time_array[valid_mask]
                metric_array = metric_array[valid_mask]
        

                if len(time_array) == 0:
                    all_histograms.append(None)
                    continue
                
                
                # Create histogram
                h, xedges, yedges = np.histogram2d(
                    time_array, metric_array,
                    bins=[time_bins_array, metric_bins],
                    range=[time_range_used, metric_range_used]
                )
                xedges = xedges * unit_change[0]
                yedges = yedges * unit_change[1]
                all_histograms.append((h, xedges, yedges))
                all_data.append((time_array, metric_array))

        # ✅ STEP 2: Find global vmin/vmax for this row
        all_h_values = []
        for hist in all_histograms:
            if hist is not None:
                h, _, _ = hist
                all_h_values.append(h[h > 0])  # Only non-zero values
        
        all_h_flat = np.concatenate(all_h_values)
        
        if v_lim[0] is not None:
            vmin = v_lim[0]
        else:
            vmin = all_h_flat.min() if len(all_h_flat) > 0 else 1
        
        if v_lim[1] is not None:
            vmax = v_lim[1]
        else:
            vmax = all_h_flat.max() if len(all_h_flat) > 0 else None
        
        # Create normalization
        if scale[2] == 'log':
            from matplotlib.colors import LogNorm
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm_obj = Normalize(vmin=vmin, vmax=vmax)
        
        print(f"Row {panel_idx}: vmin={vmin:.2e}, vmax={vmax:.2e}")
        
        # ✅ STEP 3: Plot all columns with SHARED normalization
        first_pcm = None
        for panel_idx in range(nrows):
            row_histograms = all_histograms[panel_idx*ncols:(panel_idx+1)*ncols]
            for col_idx, hist in enumerate(row_histograms):
                ax = axs[panel_idx][col_idx]
                
                if hist is None:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                            ha='center', va='center', fontsize=12)
                    continue
                
                h, xedges, yedges = hist
                
                pcm = ax.pcolormesh(
                    xedges, yedges, h.T,
                    cmap=cmap,
                    norm=norm_obj,  # ✅ Shared norm!
                    rasterized=True,
                    shading='auto'
                )
                
                if first_pcm is None:
                    first_pcm = pcm
                
                # Set scales
                if scale[1] == 'log':
                    ax.set_yscale('log')
                
                # Hide labels
                if col_idx > 0:
                    ax.tick_params(labelleft=False)
                else:
                    if ticks is not None:
                        ax.set_yticks(ticks[1])
                if panel_idx < nrows - 1:
                    ax.tick_params(labelbottom=False)
                
        # ✅ Add colorbars (one for the all rows since they share the same norm)

        if first_pcm is not None:
            cbar = fig.colorbar(
                first_pcm,
                label='Counts',
                location='right',
                pad=0.01,
                aspect=15,
                cax=fig.add_axes([0.92, 0.15, 0.02, 0.89])  # Custom position for each colorbar
            )
        
        
        fig.supxlabel(time_ylabel, fontsize=11)
        fig.supylabel(metric_ylabel, fontsize=11)
        # Adjust layout
        fig.subplots_adjust(hspace=0.0,wspace=0.0, left=0.1, right=0.91, top=0.99, bottom=0.1)
        
        if save:
            figname = f'multiple_analysis_{metric_name}_2dhistogram_multipanel.png' if save_as is None else save_as
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=500, bbox_inches='tight')
        
        return self.time_series_dicts

    def multiple_analysis_time_series_2dhistogram_multipanel(self, metric_name, metric_ylabel, time_ylabel, ticks=[None, None, None], metric_mask=None, dt_fixed=1000, survivor_per_at_t=False, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'linear', 'log'], v_key='All', global_ylim=True, normalize=False, metric_range=None, time_range=None, v_lim=[None, None], time_bins=1000, metric_bins=1000, unit_change=[1, 1, 1], use_existing_time_series=True, fig_size=None, save=True, save_as=None):
        """
        Plot 2D histogram of time series metric in multi-panel layout.
        Each panel represents one analysis system.
        
        Parameters:
        -----------
        analysis_key : str or list
            'All' or list of analysis names
        metric_name : str
            Name of the metric array (e.g., 'semi_major_axes', 'eccentricities')
        metric_ylabel : str
            Label for metric axis
        time_ylabel : str
            Label for time axis
        v_key : str
            'All' to include all velocity bins
        time_bins : int
            Number of bins for time axis
        metric_bins : int
            Number of bins for metric axis
        time_range : tuple, optional
            (min, max) for time axis
        metric_range : tuple, optional
            (min, max) for metric axis
        cmap_name : str
            Colormap name
        norm : str
            'log' or 'linear' for colorbar normalization
        vmax : float, optional
            Maximum value for colorbar
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)

        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 2*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
        # Get colormap
        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))
        
        row_histograms = []  # [(h, xedges, yedges), ...]
        row_data = []  # [(time_array, metric_array), ...]
        all_survivors = []
        # Process each analysis
        for panel_idx, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[panel_idx]
            ax = axs[panel_idx]
            
            # Collect all time series data for this analysis
            all_times = []
            all_metrics = []
            all_lifetimes = []
            survivors = {}
            # if metric_name in self.time_series_dicts.keys() and use_existing_time_series == True:
            #     if panel_idx in self.time_series_dicts[metric_name].keys():
            #         print(f"Panel {panel_idx}: Using existing time series data for '{metric_name}'")
            #         time_array, metric_array = self.time_series_dicts[metric_name][panel_idx]
            #     else:
            #         use_existing_time_series = False

            # if metric_name not in self.time_series_dicts.keys() :
            #     self.time_series_dicts[metric_name] = {}
            v_keys = [vv for vv in analysis_dict.keys() if 'V' in vv]
            for v in v_keys:

                
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                    continue
                
                # Get time series arrays
                time_series = self._get_metric_from_sources(analysis_dict, v, 'times')
                metric_series = self._get_metric_from_sources(analysis_dict, v, metric_name)

                if survivor_per_at_t:
                    lifetime = self._get_metric_from_sources(analysis_dict, v, 'lifetime')
                    if lifetime is not None:
                        lifetime = np.array(lifetime)
                        all_lifetimes.extend(lifetime) 
        

                
                if time_series is None or metric_series is None:
                    continue
                
                if metric_mask is not None:
                    analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                    analysis = self.get_analysis(analysis_name)
                    mask = self.resolve_mask(analysis, metric_mask, v)
                    if mask is not None:
                        mask = np.asarray(mask)
                        metric_series = [m for j, m in enumerate(metric_series) if mask[j]]
                        time_series = [m for j, m in enumerate(time_series) if mask[j]]
                
                # Process time series
                times, metrics = self.process_timeseries_metric(
                    time_series=time_series, 
                    metric_series=metric_series,  
                    dt_fixed=dt_fixed
                )
                if len(times) < 2 or len(metrics) < 2:
                    print(fr"Panel {panel_idx}, v={v}: Not enough data points after processing time series for '{metric_name}'")
                    continue
                # Convert to arrays
                time_array = np.concatenate(times)
                metric_array = np.concatenate(metrics)

                all_times.extend(time_array)
                all_metrics.extend(metric_array)
               
            if survivor_per_at_t and len(all_lifetimes) > 0:
                all_lifetimes = np.array(all_lifetimes)
                for t in survivor_per_at_t:
                    survivors_at_t = (np.sum(all_lifetimes > t) / len(all_lifetimes)) * 100
                    survivors[t] = survivors_at_t
                    print(rf"Panel {panel_idx}, t={t}: {survivors_at_t:.1f}\\% survivors (n={len(all_lifetimes)})")
            
            # ✅ Store dict for this panel
            all_survivors.append(survivors)
            
            if len(all_times) == 0:
                print(f"Panel {panel_idx}: No data for '{metric_name}'")
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
                continue
            
            # Convert to arrays
            time_array = np.array(all_times)
            metric_array = np.array(all_metrics)
            print(f"Panel {panel_idx}: Collected {len(time_array)} data points for '{metric_name}'")

            # Determine ranges
            time_range_used = time_range if time_range else (time_array.min(), time_array.max())
            metric_range_used = metric_range if metric_range else (metric_array.min(), metric_array.max())

            bin_width = dt_fixed
            time_bins_array = np.arange(time_range_used[0], time_range_used[1] + bin_width, bin_width)
            
            valid_mask = np.isfinite(time_array) & np.isfinite(metric_array) & (metric_array > 0)
            time_array = time_array[valid_mask]
            metric_array = metric_array[valid_mask]
        
            
            # Create histogram
            h, xedges, yedges = np.histogram2d(
                time_array, metric_array,
                bins=[time_bins_array, metric_bins],
                range=[time_range_used, metric_range_used]
            )
            xedges = xedges * unit_change[0]
            yedges = yedges * unit_change[1]
            # points_per_bin = len(time_array) / time_bins
            # count__normalization = 1 / points_per_bin
            # if normalize:
            #     h = h * count__normalization
            # Plot with pcolormesh
            # if scale[2] == 'log':
            #     from matplotlib.colors import LogNorm
            #     norm_obj = LogNorm(vmin=v_lim[0] if v_lim[0] is not None else 1, vmax=v_lim[1] if v_lim[1] is not None else h.max())
            # else:
            #     from matplotlib.colors import Normalize
            #     norm_obj = Normalize(vmin=v_lim[0] if v_lim[0] is not None else 0, vmax=v_lim[1] if v_lim[1] is not None else h.max())

            # Store histogram data for later use
            row_histograms.append((h, xedges, yedges))
            row_data.append((time_array, metric_array))


            # Set log scale for metric axis if appropriate
            # if scale[1] == 'log':
            #     ax.set_yscale('log')
                
            zkey = zkey / const.M_sun.value if zkey is not None else None
            # # Add analysis label
            ax.text(0.65, 0.95, 
                fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                transform=ax.transAxes, 
                va='top', ha='left',
                fontsize=9,
                color='white',  # ✅ White text color
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0))  # ✅ Fully transparent box 


        all_h_values = []
        for hist in row_histograms:
            if hist is not None:
                h, _, _ = hist
                all_h_values.append(h[h > 0])  # Only non-zero values
        
        all_h_flat = np.concatenate(all_h_values)
        
        if v_lim[0] is not None:
            vmin = v_lim[0]
        else:
            vmin = all_h_flat.min() if len(all_h_flat) > 0 else 1
        
        if v_lim[1] is not None:
            vmax = v_lim[1]
        else:
            vmax = all_h_flat.max() if len(all_h_flat) > 0 else None
        
        # Create normalization
        if scale[2] == 'log':
            from matplotlib.colors import LogNorm
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm_obj = Normalize(vmin=vmin, vmax=vmax)
        
        print(f"Row {panel_idx}: vmin={vmin:.2e}, vmax={vmax:.2e}")
        
        # ✅ STEP 3: Plot all rows with SHARED normalization
        first_pcm = None
        for panel_idx, hist in enumerate(row_histograms):
            ax = axs[panel_idx]

            if ticks[1] is not None:
                ax.set_yticks(ticks[1])
            # Only show xlabel on bottom panel
            if panel_idx < nrows - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(time_ylabel)
                if ticks[0] is not None:
                    ax.set_xticks(ticks[0])

            if hist is None:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=12)
                continue
            
            h, xedges, yedges = hist
            
            pcm = ax.pcolormesh(
                xedges, yedges, h.T,
                cmap=cmap,
                norm=norm_obj,  # ✅ Shared norm!
                rasterized=True,
                shading='auto'
            )
            

            if survivor_per_at_t:
                survivors_at_t = all_survivors[panel_idx]
                for m, t in enumerate(survivors_at_t.keys()):
                    s = survivors_at_t[t]
                    if s > 1:
                        s = np.round(s, 1)
                        s_label = f'{s:.0f}\\%'
                    else:
                        s = np.round(s, 2)
                        s_label = f'{s:.1f}\\%'
                    t = t * unit_change[0]
                    ax.axvline(t, color='white', linestyle='--', alpha=0.8, linewidth=0.4)   
                    ax.text(t, 0.7, 
                        s_label,
                        transform=ax.get_xaxis_transform(), 
                        va='top', ha='center',
                        fontsize=9,
                        color='white',  # ✅ White text color
                        bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0))  # ✅ Fully transparent box
            
            if first_pcm is None:
                first_pcm = pcm
            
            # Set scales
            if scale[1] == 'log':
                ax.set_yscale('log')
            
                    
        # ✅ Add colorbars (one for all rows)

        if first_pcm is not None:
            cbar = fig.colorbar(
                first_pcm, 
                ax=axs[:].ravel().tolist(),
                label='Counts',
                location='top',          # ← Change from 'right' to 'top'
                orientation='horizontal',
                pad=0.01,
                cax=fig.add_axes([0.16, 0.97, 0.8, 0.02])  # ← Custom position for horizontal colorbar       
            )
        
        # Adjust layout
        fig.supylabel(metric_ylabel, fontsize=11, y=0.54)
        # Adjust layout
        fig.subplots_adjust(hspace=0.0,wspace=0.0, left=0.16, right=0.96, top=0.965, bottom=0.12)
        
        if save:
            figname = f'multiple_analysis_{metric_name}_2dhistogram_multipanel.png' if save_as is None else save_as
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        return self.time_series_dicts

    def multiple_analysis_time_series_2dhistogram_multipanel_stacked(self, metric_names, metric_ylabels, time_ylabel, ticks=[None, None, None, None], metric_masks=None, mask_labels=None, dt_fixed=1000, survivor_per_at_t=False, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'linear', 'linear', 'log'], v_key='All', global_ylim=True, normalize=False, metric_ranges=None, time_range=None, v_lim=[None, None], time_bins=1000, metric_bins=1000, unit_change=[1, 1, 1], use_existing_time_series=True, fig_size=None, save=True, save_as=None):
        """
        Plot 2D histogram of time series metrics in multi-panel layout with stacked subplots.
        Each analysis gets 2 subplots (one per metric) stacked vertically, sharing x-axis (time).
        
        Parameters:
        -----------
        metric_names : list of str
            List of 2 metric names (e.g., ['semi_major_axes', 'eccentricities'])
        metric_ylabels : list of str
            List of 2 y-axis labels for each metric
        metric_ranges : list of tuples
            [(min1, max1), (min2, max2)] for each metric
        ticks : list of 4 elements
            [time_ticks, metric1_ticks, metric2_ticks, colorbar_ticks]
        scale : list of 4 elements
            [time_scale, metric1_scale, metric2_scale, colorbar_scale]
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, None, analysis_zkey)
            zkey_values.append(zkey)
        
        # Create multi-panel figure: nrows = num_analyses * 2 (two metrics per analysis), ncols = 1
        n_analyses = len(analysis_list)
        nrows = n_analyses * 2  # 2 metrics per analysis
        ncols = 1
        
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (5, 2*nrows), 
                                nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single pair
        if nrows == 2:
            axs = [axs[0], axs[1]]
        
        # Get colormap
        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))
        
        # Process each metric separately
        all_histograms = [[], []]  # One list per metric
        all_survivors = [[], []]
        shown_counts_percentage = [[], []]
        mask_sums = [[], []]
        
        for metric_idx, metric_name in enumerate(metric_names):
            print(f"\n📊 Processing metric {metric_idx+1}/{len(metric_names)}: {metric_name}")
            
            metric_range = metric_ranges[metric_idx] if metric_ranges else None
            
            # Process each analysis for this metric
            for analysis_idx, analysis_dict in enumerate(analysis_list):
                zkey = zkey_values[analysis_idx]
                
                # Collect time series data
                all_times = []
                all_metrics = []
                all_lifetimes = []
                survivors = {}
                mask_sum = 0
                
                v_keys = [vv for vv in analysis_dict.keys() if 'V' in vv]
                for v in v_keys:
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    
                    time_series = self._get_metric_from_sources(analysis_dict, v, 'times')
                    metric_series = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    
                    if survivor_per_at_t:
                        lifetime = self._get_metric_from_sources(analysis_dict, v, 'lifetime')
                    
                    if time_series is None or metric_series is None:
                        continue
                    
                    if metric_masks is not None:
                        analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                        analysis = self.get_analysis(analysis_name)
                        mask = self.resolve_mask(analysis, metric_masks[metric_idx] if isinstance(metric_masks, list) else metric_masks, v)
                        mask_label = mask_labels[metric_idx] if isinstance(mask_labels, list) else mask_labels
                        if mask is not None:
                            mask = np.asarray(mask)
                            metric_series = [m for j, m in enumerate(metric_series) if mask[j]]
                            time_series = [m for j, m in enumerate(time_series) if mask[j]]
                            mask_sum += np.sum(mask)
                            if survivor_per_at_t:
                                lifetime = [m for j, m in enumerate(lifetime) if mask[j]] 
                    
                    times, metrics = self.process_timeseries_metric(
                        time_series=time_series, 
                        metric_series=metric_series,  
                        dt_fixed=dt_fixed
                    )
                    
                    if len(times) < 2 or len(metrics) < 2:
                        continue
                    
                    all_times.extend(np.concatenate(times))
                    all_metrics.extend(np.concatenate(metrics))
                    if survivor_per_at_t:
                        all_lifetimes.extend(np.array(lifetime))
                
                # Calculate survivors
                if survivor_per_at_t and len(all_lifetimes) > 0:
                    all_lifetimes = np.array(all_lifetimes)
                    for t in survivor_per_at_t:
                        survivors[t] = (np.sum(all_lifetimes > t) / len(all_lifetimes)) * 100
                
                all_survivors[metric_idx].append(survivors)
                mask_sums[metric_idx].append(mask_sum)
                if len(all_times) == 0:
                    all_histograms[metric_idx].append(None)
                    continue
                
                # Create histogram
                time_array = np.array(all_times)
                metric_array = np.array(all_metrics)
                
                time_range_used = time_range if time_range else (time_array.min(), time_array.max())
                metric_range_used = metric_range if metric_range else (metric_array.min(), metric_array.max())
                
                bin_width = dt_fixed
                time_bins_array = np.arange(time_range_used[0], time_range_used[1] + bin_width, bin_width)
                
                valid_mask = np.isfinite(time_array) & np.isfinite(metric_array) & (metric_array > 0)
                time_array = time_array[valid_mask]
                metric_array = metric_array[valid_mask]
                
                h, xedges, yedges = np.histogram2d(
                    time_array, metric_array,
                    bins=[time_bins_array, metric_bins],
                    range=[time_range_used, metric_range_used]
                )
                
                xedges = xedges * unit_change[0]
                yedges = yedges * unit_change[1]

                counts_shown_percentage = h.sum() / len(time_array) * 100 if len(time_array) > 0 else 0 

                shown_counts_percentage[metric_idx].append(counts_shown_percentage)
                all_histograms[metric_idx].append((h, xedges, yedges))
        
        # Find global vmin/vmax across both metrics
        all_h_values = []
        for metric_histograms in all_histograms:
            for hist in metric_histograms:
                if hist is not None:
                    h, _, _ = hist
                    all_h_values.append(h[h > 0])
        
        all_h_flat = np.concatenate(all_h_values)
        vmin = v_lim[0] if v_lim[0] is not None else (all_h_flat.min() if len(all_h_flat) > 0 else 1)
        vmax = v_lim[1] if v_lim[1] is not None else (all_h_flat.max() if len(all_h_flat) > 0 else None)
        
        if scale[3] == 'log':
            from matplotlib.colors import LogNorm
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm_obj = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot all subplots
        first_pcm = None
        for analysis_idx in range(n_analyses):
            zkey = zkey_values[analysis_idx] / const.M_sun.value if zkey_values[analysis_idx] is not None else None
            
            for metric_idx in range(2):
                # Calculate row index: analysis_idx * 2 + metric_idx
                row_idx = analysis_idx * 2 + metric_idx
                ax = axs[row_idx]
                
                hist = all_histograms[metric_idx][analysis_idx]
                shown_percentage = shown_counts_percentage[metric_idx][analysis_idx]
                if hist is None:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=12)
                    continue
                
                h, xedges, yedges = hist
                
                pcm = ax.pcolormesh(
                    xedges, yedges, h.T,
                    cmap=cmap,
                    norm=norm_obj,
                    rasterized=True,
                    shading='auto'
                )
                
                if first_pcm is None:
                    first_pcm = pcm

                print(f"Total systems in the histogram for metric {metric_idx}: {h.sum()}")
                
                ax.text(0.98, 0.6, f'{np.round(shown_percentage, 0)}\\% repr.', transform=ax.transAxes, va='top', ha='right', fontsize=9, color='white',
                        bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.0))
                
                if metric_masks is not None and metric_masks[metric_idx] is not None:
                    msk_sum = mask_sums[metric_idx][analysis_idx]
                    representation_percentage = (msk_sum / (len(v_keys)*len(mask)))* 100 if isinstance(metric_masks, list) else None
                    mask_str = f'{np.round(representation_percentage, 0):.0f}\\% {mask_label}' if representation_percentage is not None else mask_label
                    if metric_idx == 1:  # Only label on first metric subplot to avoid clutter
                        ax.text(0.98, 0.97, f'{mask_str}', transform=ax.transAxes, va='top', ha='right', fontsize=9, color='white',
                        bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0))
                # Add survivor lines
                if survivor_per_at_t:
                    survivors_at_t = all_survivors[metric_idx][analysis_idx]
                    x_shift = 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])  # ~2% of axis width

                    for t in survivors_at_t.keys():
                        s = survivors_at_t[t]
                        s_label = f'{s:.0f}\\%' if s > 1 else f'{s:.1f}\\%'
                        t_scaled = t * unit_change[0] 
                        if metric_idx == 1:  # Only label on second metric subplot to avoid clutter
                            ax.axvline(t_scaled, ymin=0.24, ymax=1, color='white', linestyle='--', alpha=0.8, linewidth=0.4)
                            ax.text(t_scaled - x_shift, 0.04, s_label, transform=ax.get_xaxis_transform(), 
                                va='bottom', ha='center', fontsize=9, color='white',
                                bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0))
                        else:
                            ax.axvline(t_scaled, color='white', linestyle='--', alpha=0.8, linewidth=0.4)
                            
                
                # Add analysis label on top subplot of each pair
                if metric_idx == 0:
                    ax.text(0.98, 0.97, 
                        fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                        transform=ax.transAxes, va='top', ha='right', fontsize=9,
                        color='white', bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0))
                    
                # Make x axis thicker to denote the group 
                if metric_idx == 1:
                    ax.spines['bottom'].set_linewidth(1.5)

                # Set y-label
                ax.set_ylabel(metric_ylabels[metric_idx], fontsize=10)
                
                # Set y-scale and ticks for each metric
                metric_scale_idx = metric_idx + 1  # scale[1] for metric1, scale[2] for metric2
                if scale[metric_scale_idx] == 'log':
                    ax.set_yscale('log')
                
                tick_idx = metric_idx + 1  # ticks[1] for metric1, ticks[2] for metric2
                if ticks[tick_idx] is not None:
                    ax.set_yticks(ticks[tick_idx])
                
                # Only show x-axis label on bottom subplot
                if row_idx < nrows - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(time_ylabel, fontsize=10)
                    if ticks[0] is not None:
                        ax.set_xticks(ticks[0])
        
        # Add colorbar
        if first_pcm is not None:
            cbar = fig.colorbar(
                first_pcm, 
                ax=axs if isinstance(axs, list) else [axs],
                label='Counts',
                location='top',
                orientation='horizontal',
                pad=0.01,
                cax=fig.add_axes([0.15, 0.98, 0.84, 0.015])
            )
            
            if ticks[3] is not None:
                cbar.set_ticks(ticks[3])
        
        fig.subplots_adjust(hspace=0.0, wspace=0, left=0.15, right=0.99, top=0.98, bottom=0.05)
        
        if save:
            figname = save_as if save_as else f'multiple_analysis_{"_".join(metric_names)}_2dhistogram_stacked.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        return self.time_series_dicts

    def multiple_analysis_voxel_2dhistogram(self, metric_name, metric_ylabel, voxel_name, voxel_xlabel, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=[[['linear', 'log', 'log'], ['linear', 'log', 'log']], [['linear', 'log', 'log'], ['linear', 'log', 'log']]], v_key='All', global_ylim=True, normalize=False, normalize_annotation=False, annotate_bins=False, custom_voxel_bins=[[None, None], [None, None]], metric_range=None, voxel_range=[[None, None], [None, None]], v_lim=[None, None], voxel_bins=1000, metric_bins=1000, unit_change=[[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]], metric_masks=None, use_existing_time_series=False, fig_size=None, save=True, save_as=None):
        """
        Plot 2D histogram of metric vs voxel parameter.
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        norm_const_list = []
        annotate_norm_list = []
        unique_voxel_list = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
            for v in a.keys():
                if 'V' in v:
                    if normalize:
                        normalization_const = self._get_metric_from_sources(a, v, normalize)
                        norm_const_list.append(normalization_const)
                    if annotate_bins and normalize_annotation:
                        annotate_norm_const = self._get_metric_from_sources(a, v, normalize_annotation)
                        annotate_norm_list.append(annotate_norm_const)
        print(f'total samples: {sum(norm_const_list)}, total captures : {sum(annotate_norm_list)}')
        # Handle single voxel_name or list
        if isinstance(voxel_name, str):
            voxel_name_list = [voxel_name]
            voxel_xlabel_list = [voxel_xlabel]
        else:
            voxel_name_list = voxel_name
            voxel_xlabel_list = voxel_xlabel
        if isinstance(metric_name, str):
            metric_name_list = [metric_name] 
        else:
            metric_name_list = metric_name

        ncols = len(voxel_name_list)
        nrows = len(metric_name_list)
        print(f'Creating {nrows}x{ncols} multipanel figure for voxel 2D histograms...')
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 3*nrows), nrows=nrows, ncols=ncols, sharex=False)

        if ncols == 1 and nrows == 1:
            axs = [axs]
        elif nrows == 1:
            axs = axs[np.newaxis, :]
        elif ncols == 1:
            axs = axs[:, np.newaxis]

        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))
        histogram_data_all = []  # [(h, xedges, yedges, counts), ...]

        global_hmin = np.inf
        global_hmax = -np.inf
        for row_idx, metric_name in enumerate(metric_name_list):
            unit_change_row = unit_change[row_idx] 
            voxel_range_row = voxel_range[row_idx] if voxel_range is not None else None
            scale_row = scale[row_idx]
            print(unit_change_row, voxel_range_row)
            histogram_data = []  # Store histogram data for this row
            # ✅ SINGLE PASS: Collect data and create histograms
            for axis_idx in range(ncols):
                voxel_name_axis = voxel_name_list[axis_idx]
                unit_change_axis = unit_change_row[axis_idx]
                voxel_range_axis = voxel_range_row[axis_idx] if voxel_range_row is not None else None
                scale_axis = scale_row[axis_idx]
                print(f'after chosing axis {axis_idx}:', unit_change_axis, voxel_range_axis, voxel_name_axis, scale_axis)
                # Collect data for this axis
                all_voxel = []
                all_metrics = []
                
                for panel_idx, analysis_dict in enumerate(analysis_list):
                    for v in analysis_dict.keys():
                        if 'V' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                            continue
                        
                        metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                        voxel = self._get_metric_from_sources(analysis_dict, v, voxel_name_axis)

                        if metric is None or voxel is None:
                            continue
                    
                        # Apply mask if provided
                        if metric_masks is not None:
                            metric_mask = metric_masks[row_idx] if row_idx < len(metric_masks) else None
                            if metric_mask is not None:
                                analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                                analysis = self.get_analysis(analysis_name)
                                mask = self.resolve_mask(analysis, metric_mask, v)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    print(f"Axis {row_idx}: Applying mask for '{metric_name}' at v={v}, sum={np.sum(mask)}")
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                        
                        # Process array metrics
                        if isinstance(metric, (list, tuple)):
                            if len(metric) > 2:
                                metric = metric[1:-1]
                            if len(metric) == 0:
                                continue             
                            data_m = [float(np.mean(m * unit_change_axis[1])) for m in metric if np.size(m) > 0]
                            data_v = [voxel * unit_change_axis[0]] * len(data_m)
                        
                        elif isinstance(metric, np.ndarray) and metric.ndim > 0:
                            metric = metric.flatten()
                            data_m = (metric * unit_change_axis[1]).tolist()
                            data_v = [voxel * unit_change_axis[0]] * len(data_m)
                        
                        else:
                            capture_count = self._get_metric_from_sources(analysis_dict, v, 'n_captured')
                            data_m = [metric * unit_change_axis[1]] * capture_count
                            data_v = [voxel * unit_change_axis[0]] * capture_count

                        if data_m is not None and data_v is not None:
                            all_metrics.extend(data_m)
                            all_voxel.extend(data_v)

                print(f"Axis {axis_idx}: Collected {len(all_voxel)} data points for '{metric_name}'")
                
                # Convert to arrays
                voxel_array = np.array(all_voxel)
                metric_array = np.array(all_metrics)
                
                # Filter out invalid values
                valid_mask = np.isfinite(voxel_array) & np.isfinite(metric_array) & (metric_array > 0)
                voxel_array = voxel_array[valid_mask]
                metric_array = metric_array[valid_mask]
                
                if len(voxel_array) == 0:
                    print(f"Axis {axis_idx}: No valid data")
                    histogram_data.append(None)
                    continue
                
                # Determine ranges
                if metric_range is None:
                    metric_range_used = (metric_array.min(), metric_array.max())
                else:
                    metric_range_used = metric_range[row_idx]
                if voxel_range_axis is None:
                    voxel_range_used = (voxel_array.min(), voxel_array.max())
                else:
                    voxel_range_used = (voxel_range_axis[0], voxel_range_axis[1])

                print(f'Axis {axis_idx}: Using voxel range {voxel_range_used} and metric range {metric_range_used}')
                if custom_voxel_bins is not False:
                    unique_voxels = np.unique(np.sort(voxel_array))
                    midpoints = (unique_voxels[:-1] + unique_voxels[1:]) / 2
                    
                    # Create bin edges:
                    # - Left edge: extrapolate from first two midpoints
                    # - Middle edges: midpoints between unique values
                    # - Right edge: extrapolate from last two midpoints
                    left_edge = unique_voxels[0] - (midpoints[0] - unique_voxels[0])
                    right_edge = unique_voxels[-1] + (unique_voxels[-1] - midpoints[-1])
                    
                    voxel_bins = np.concatenate([[left_edge], midpoints, [right_edge]])
                    unique_voxel_list.append(unique_voxels)
                    print(f"Axis {axis_idx}: Created {len(voxel_bins)-1} bins for {len(unique_voxels)} unique voxel values")

                if scale_axis[0] == 'log':
                    lowv, highv = voxel_range_used
                    voxel_bins_log = np.logspace(
                        np.log10(lowv),
                        np.log10(highv),
                        voxel_bins[axis_idx]+1
                    )
                    lowm, highm = metric_range_used
                    metric_bins_ = np.linspace(
                        lowm,
                        highm,
                        metric_bins + 1
                    )
                    print(f'Axis {axis_idx}: Using log-spaced voxel bins')
                    # ✅ Create histogram ONCE
                    h, xedges, yedges = np.histogram2d(
                        voxel_array, metric_array,
                        bins=[voxel_bins_log, metric_bins_],
                        
                    )
                else:
                    h, xedges, yedges = np.histogram2d(
                        voxel_array, metric_array,
                        bins=[voxel_bins[axis_idx], metric_bins],
                        range=[voxel_range_used, metric_range_used]
                    )
                
                counts = h.copy()  # Keep raw counts for annotation
                
                if normalize:
                    h = h / sum(norm_const_list)
                
                # Store for plotting
                histogram_data.append((h, xedges, yedges, counts))
                
                # Track global min/max (excluding zeros)
                h_nonzero = h[h > 0]
                if len(h_nonzero) > 0:
                    global_hmin = min(global_hmin, h_nonzero.min())
                    global_hmax = max(global_hmax, h.max())

            histogram_data_all.append(histogram_data)

        # Create normalization based on global min/max
        if v_lim[0] is not None:
            vmin = v_lim[0]
        else:
            vmin = global_hmin if global_hmin != np.inf else 1
        
        if v_lim[1] is not None:
            vmax = v_lim[1]
        else:
            vmax = global_hmax if global_hmax != -np.inf else None

        if scale_row[0][2] == 'log':
            from matplotlib.colors import LogNorm
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm_obj = Normalize(vmin=vmin, vmax=vmax)
        
        # ✅ Plot using stored histogram data
        pcm_list = []
        clabel = 'Occurrence' if normalize else 'Counts'
        for row_idx in range(nrows):
            histogram_data = histogram_data_all[row_idx]
            for axis_idx in range(ncols):
                print([row_idx, axis_idx])
                ax = axs[row_idx, axis_idx]

                if custom_voxel_bins is not False:
                    ax.set_xticks(unique_voxel_list[axis_idx])
                    ax.set_xticklabels([f'{x:.0f}' for x in unique_voxel_list[axis_idx]], rotation=45, ha='right')

                if histogram_data[axis_idx] is None:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
                    continue
                
                h_plot, xedges, yedges, counts = histogram_data[axis_idx]
                # Plot with shared normalization
                pcm = ax.pcolormesh(
                    xedges, yedges, h_plot.T,
                    cmap=cmap,
                    norm=norm_obj,
                    rasterized=True,
                    shading='auto'
                )
                pcm_list.append(pcm)
                
                # Annotations
                if annotate_bins:
                    if scale[row_idx][axis_idx][0] == 'log':
                        x_centers = np.sqrt(xedges[:-1] * xedges[1:])
                    else:
                        x_centers = (xedges[:-1] + xedges[1:]) / 2
                    if scale[row_idx][axis_idx][1] == 'log':
                        y_centers = np.sqrt(yedges[:-1] * yedges[1:])
                    else:
                        y_centers = (yedges[:-1] + yedges[1:]) / 2
                        
                    if normalize_annotation and annotate_norm_list:

                        counts = counts / np.sum(annotate_norm_list) if np.sum(annotate_norm_list) != 0 else counts
                        counts = counts * 1e2 # Scale to percentage
                    for i in range(len(x_centers)):
                        for j in range(len(y_centers)):
                            count = counts[i, j]
                            
                            if count > 0:
                                text = f'{int(np.round(count))}' if count >= 0.9 else f'{np.round(count, 1)}' 
                                color = 'white' 
                                
                                ax.text(x_centers[i], y_centers[j], text,
                                    ha='center', va='center',
                                    fontsize=8, color=color)
                
                # Set scales
                if scale[row_idx][axis_idx][1] == 'log':
                    ax.set_yscale('log')
                if scale[row_idx][axis_idx][0] == 'log':
                    ax.set_xscale('log')
                print(f' finished axis {axis_idx} in row {row_idx}')

        fig.colorbar(pcm_list[0], ax=axs.ravel().tolist(), label=clabel, aspect=20)
                

        for row_idx in range(nrows):
            for axis_idx in range(ncols):
                ax = axs[row_idx, axis_idx]
                ax.set_xlabel(voxel_xlabel_list[axis_idx])
                if axis_idx == 0:
                    ax.set_ylabel(metric_ylabel[row_idx])
        if ncols > 1:
            for row_idx in range(nrows):
                for col_ in range(1, ncols):
                    axs[row_idx, col_].tick_params(labelleft=False)
        if nrows > 1:
            for col_ in range(ncols):
                for row_idx in range(nrows - 1):
                    axs[row_idx, col_].tick_params(labelbottom=False)
        # Add single colorbar
        fig.subplots_adjust(right=0.77, wspace=0.02, hspace=0.02)


        if save:
            if save_as is not None:
                figname = save_as
            else:
                voxel_str = '_'.join(voxel_name_list) if isinstance(voxel_name, list) else voxel_name
                figname = f'multiple_analysis_{metric_name}_2dhistogram_{voxel_str}_{metric_bins*voxel_bins}_{metric_range[1] if metric_range is not None else "nolim"}.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')

 
    def multiple_analysis_gaussian_kde(self, metric_name, metric_label, voxel_name, voxel_label, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=[[['linear', 'log', 'log'], ['linear', 'log', 'log']], [['linear', 'log', 'log'], ['linear', 'log', 'log']]], v_key='All', global_ylim=True, normalize=False, range=None, v_lim=[None, None], voxel_bins=1000, metric_bins=1000, unit_change=[[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]], metric_masks=None, use_existing_time_series=False, fig_size=None, save=True, save_as=None):
        """
        Plot 2D scatter of metric pairs vs voxel parameter (colored by voxel value).
        
        Parameters:
        -----------
        metric_name : list of lists
            [[x_metric, y_metric], ...] - pairs of metrics to plot
        metric_label : list of lists
            [[x_label, y_label], ...] - labels for each metric pair
        voxel_name : str or list
            Name(s) of voxel parameter(s) for coloring points
        scale : list structure
            [[[x_scale, y_scale, z_scale], ...], ...] for each column and row
        unit_change : list structure
            [[[x_unit, y_unit, z_unit], ...], ...] for each column and row
        range : list structure
            [[[x_range, y_range, z_range], ...], ...] for each column and row
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        norm_const_list = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
            for v in a.keys():
                if 'V' in v:
                    if normalize:
                        normalization_const = self._get_metric_from_sources(a, v, normalize)
                        norm_const_list.append(normalization_const)

        print(f'total samples: {sum(norm_const_list)}')
        
        # Handle single voxel_name or list
        if isinstance(voxel_name, str):
            voxel_name_list = [voxel_name]
            voxel_label_list = [voxel_label]
        else:
            voxel_name_list = voxel_name
            voxel_label_list = voxel_label
        
        # metric_name should be list of pairs: [[x1, y1], [x2, y2], ...]
        if not isinstance(metric_name, list):
            metric_name_list = [[metric_name]]
        elif not isinstance(metric_name[0], list):
            # If it's just one pair: [x, y] -> [[x, y]]
            metric_name_list = [metric_name]
        else:
            metric_name_list = metric_name

        ncols = len(voxel_name_list)
        nrows = 5
        
        print(f'Creating {nrows}x{ncols} multipanel figure for 2D scatter plots...')
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 3*nrows), nrows=nrows, ncols=ncols, sharex=False)

        # Ensure axs is always 2D array
        if ncols == 1 and nrows == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = axs[np.newaxis, :]
        elif ncols == 1:
            axs = axs[:, np.newaxis]

        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))
        markers = ['o', 'X', 'D', '^', 'v', '<', '>', 'P', 's']

        # Iterate over columns (different voxel parameters)
        for col_idx, voxel_name_col in enumerate(voxel_name_list):
            # Get column-specific parameters
            unit_change_col = unit_change[col_idx] if col_idx < len(unit_change) else unit_change[0]
            range_col = range[col_idx] if range is not None and col_idx < len(range) else [None] * nrows
            scale_col = scale[col_idx] if col_idx < len(scale) else scale[0]
            row_x_mins = []
            row_x_maxs = []
            row_y_mins = []
            row_y_maxs = []
            # Iterate over rows (different metric pairs)
            for row_idx, metric_pair in enumerate(metric_name_list):
                if len(metric_pair) != 2:
                    print(f"Warning: metric_pair at row {row_idx} should have exactly 2 elements (x, y)")
                    continue
                    
                metric_x_name, metric_y_name = metric_pair
                
                # Get row-specific parameters
                unit_change_row = unit_change_col[row_idx] if row_idx < len(unit_change_col) else [1, 1, 1]
                range_row = range_col[row_idx] if range_col and row_idx < len(range_col) else [None, None, None]
                scale_row = scale_col[row_idx] if row_idx < len(scale_col) else ['linear', 'linear', 'linear']
                
                print(f"\nProcessing col={col_idx}, row={row_idx}")
                print(f"  Metrics: x={metric_x_name}, y={metric_y_name}, voxel={voxel_name_col}")
                print(f"  Unit change: {unit_change_row}")
                print(f"  Scale: {scale_row}")
                
                # Collect data for this subplot
                all_x = []
                all_y = []
                all_z = []
                for panel_idx, analysis_dict in enumerate(analysis_list):
                    for v in analysis_dict.keys():
                        if 'V' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                            continue

                        # Get metrics
                        metric_x = self._get_metric_from_sources(analysis_dict, v, metric_x_name)
                        metric_y = self._get_metric_from_sources(analysis_dict, v, metric_y_name)
                        voxel = self._get_metric_from_sources(analysis_dict, v, voxel_name_col)

                        if metric_x is None or metric_y is None or voxel is None:
                            continue
                    
                        # Apply mask if provided
                        if metric_masks is not None:
                            metric_mask = metric_masks[row_idx] if row_idx < len(metric_masks) else None
                            if metric_mask is not None:
                                analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                                analysis = self.get_analysis(analysis_name)
                                mask = self.resolve_mask(analysis, metric_mask, v)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    print(f"  Applying mask: sum={np.sum(mask)}")
                                    metric_x = [m for j, m in enumerate(metric_x) if mask[j]]
                                    metric_y = [m for j, m in enumerate(metric_y) if mask[j]]
                                    voxel = [vx for j, vx in enumerate(voxel) if mask[j]]
                
                        # Process each metric
                        def process_metric(m, unit_factor):
                            """Convert metric to flat list of floats"""
                            if isinstance(m, (list, tuple)):
                                # List of arrays (time series)
                                if len(m) > 2:
                                    m = m[1:-1]  # Trim edges
                                if len(m) == 0:
                                    return []
                                return [float((mm * unit_factor) for mm in m if np.size(mm) > 0)]
                            
                            elif isinstance(m, np.ndarray) and m.ndim > 0:
                                # Single array
                                m_flat = m.flatten() * unit_factor
                                return [m_flat]
                            
                            else:
                                # Scalar - replicate based on metric_x length
                                return [float(m * unit_factor)]
                        
                        data_x = process_metric(metric_x, unit_change_row[0])
                        data_y = process_metric(metric_y, unit_change_row[1])
                        data_z = process_metric(voxel, unit_change_row[2])
                        
                        
                        # If voxel is scalar, broadcast to match x/y
                        if len(data_z) == 1 and len(data_x) > 1:
                            data_z = data_z * len(data_x)

                        if data_x is not None and data_y is not None and data_z is not None:
                            all_x.extend(data_x)
                            all_y.extend(data_y)
                            all_z.extend(data_z)
                        

                print(f"  Collected {len(all_x)} data points")
                
                if len(all_x) == 0:
                    print(f"  No valid data")
                    axs[0, col_idx].text(0.5, 0.5, 'No Data', 
                                            transform=axs[0, col_idx].transAxes,
                                            ha='center', va='center')
                    continue

                # Convert to arrays
                x_array = np.array(all_x)
                y_array = np.array(all_y)
                z_array = np.array(all_z)

                # Filter invalid values
                valid_mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(z_array) & (y_array > 0)
                x_array = x_array[valid_mask]
                y_array = y_array[valid_mask]
                z_array = z_array[valid_mask]
                
                print(f"  {len(x_array)} valid points after filtering")
                
                if len(x_array) == 0:
                    continue

                row_x_mins.append(x_array.min())
                row_x_maxs.append(x_array.max())
                row_y_mins.append(y_array.min())
                row_y_maxs.append(y_array.max())
                # Determine ranges

                X, Y = np.mgrid[x_array.min():x_array.max():100j, y_array.min():y_array.max():100j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                values = np.vstack([x_array, y_array])
                kernel = gaussian_kde(values, weights=z_array)
                Z = np.reshape(kernel(positions).T, X.shape)
                z_range = range_row[2] if range_row and range_row[2] is not None else (Z.min(), Z.max())    

                # Create scatter plot
                if scale_row[2] == 'log':
                    from matplotlib.colors import LogNorm
                    norm = LogNorm(vmin=z_range[0], vmax=z_range[1])
                else:
                    norm = None  # Use default linear normalization

                # Create scatter plot
                gk = axs[0, col_idx].imshow(
                    Z, extent=(x_array.min(), x_array.max(), y_array.min(), y_array.max()),
                    origin='lower', cmap=cmap, aspect='auto', alpha=0.8,
                    norm=norm  # ✅ CHANGED: Use norm instead of vmin/vmax
                )
                sc = axs[0, col_idx].plot(x_array, y_array, 'k.', markersize=2, alpha=0.5)  # Overlay points for reference
                # Add colorbar (one per column)
                if row_idx == 0:
                    cbar = fig.colorbar(gk, ax=axs[:, col_idx].ravel().tolist(), 
                                        orientation='horizontal', location='top', 
                                        pad=0.1,  # ✅ REDUCED from 0.5 to 0.1
                                        aspect=30, 
                                        shrink=1)  # ✅ ADDED: Shrink colorbar width
                    cbar.set_label(f"{voxel_label_list[col_idx]}", labelpad=10)  # ✅ INCREASED labelpad


                # # Set ranges and scales
                # axs[0, col_idx].set_xlim(x_range)
                # axs[0, col_idx].set_ylim(y_range)

                if scale_row[1] == 'log':
                    axs[0, col_idx].set_yscale('log')
                if scale_row[0] == 'log':
                    axs[0, col_idx].set_xscale('log')

                print(f'  ✓ Finished subplot')


            if len(row_x_mins) > 0:
                # Get global range for this column
                range_row = range_col[0] if range_col and len(range_col) > 0 else [None, None, None]
                
                # Use provided range or calculate from data
                global_x_range = range_row[0] if range_row and range_row[0] is not None else (min(row_x_mins), max(row_x_maxs))
                global_y_range = range_row[1] if range_row and range_row[1] is not None else (min(row_y_mins), max(row_y_maxs))
                
                print(f'\n  ✅ Setting global ranges for column {col_idx}:')
                print(f'     x_range: {global_x_range}')
                print(f'     y_range: {global_y_range}')
                
                # Apply to all rows in this column
                
                axs[0, col_idx].set_xlim(global_x_range)
                axs[0, col_idx].set_ylim(global_y_range)

        # # Set axis labels
        axs[0, 0].set_ylabel(metric_label[1])
        fig.supxlabel(metric_label[0], fontsize=11, y=0.01)

        # # Hide y-tick labels on non-leftmost columns
        if ncols > 1:

            axs[0, -1].tick_params(labelleft=False)

        # # Hide x-tick labels on non-bottom rows
        # if nrows > 1:
        #     for col_idx in range(ncols):
        #         axs[0, col_idx].tick_params(labelbottom=False)

        fig.subplots_adjust(right=0.95, wspace=0.05, hspace=0.05, 
                            top=0.75, bottom=0.15)  # Adjusted to make room for colorbars and labels
        
        if save:
            if save_as is not None:
                figname = save_as
            else:
                voxel_str = '_'.join(voxel_name_list) if isinstance(voxel_name, list) else voxel_name
                metric_str = '_'.join(['_'.join(pair) for pair in metric_name_list])
                figname = f'multiple_analysis_{metric_str}_2D_scatter_{voxel_str}.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'\n✓ Saved: {figname}')

    def multiple_analysis_m_v_contour(self, metric_name, metric_ylabel, voxel_name, voxel_xlabel, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=[[['linear', 'log', 'log'], ['linear', 'log', 'log']], [['linear', 'log', 'log'], ['linear', 'log', 'log']]], v_key='All', global_ylim=True, normalize=False, normalize_annotation=False, annotate_bins=False, custom_voxel_bins=[[None, None], [None, None]], metric_range=None, voxel_range=[[None, None], [None, None]], v_lim=[None, None], voxel_bins=1000, metric_bins=1000, unit_change=[[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]], metric_masks=None, use_existing_time_series=False, fig_size=None, save=True, save_as=None):
        """
        Plot contour of metric on mass and velocity grid parameter.
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        norm_const_list = []
        annotate_norm_list = []
        unique_voxel_list = []
        m_values = []
        v_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            mkey = self._get_metric_from_sources(a[vkey], 'mC')
            m_values.append(mkey)
            if v_values == []:
                vkey = list(a.keys())[0]
                vkey_parsed = float(vkey.split('V')[1])
                v_values.append(vkey_parsed)
            for v in a.keys():
                if 'V' in v and ():
                    if normalize:
                        normalization_const = self._get_metric_from_sources(a[v], normalize)
                        norm_const_list.append(normalization_const)
                    if annotate_bins and normalize_annotation:
                        annotate_norm_const = self._get_metric_from_sources(a[v], normalize_annotation)
                        annotate_norm_list.append(annotate_norm_const)
        
        print(f'total samples: {sum(norm_const_list)}, total captures : {sum(annotate_norm_list)}')
        M, V = np.meshgrid(m_values, v_values, indexing='ij')
        # Fill Z with metric values
        # Handle single voxel_name or list
        if isinstance(voxel_name, str):
            voxel_name_list = [voxel_name]
            voxel_xlabel_list = [voxel_xlabel]
        else:
            voxel_name_list = voxel_name
            voxel_xlabel_list = voxel_xlabel
        if isinstance(metric_name, str):
            metric_name_list = [metric_name] 
        else:
            metric_name_list = metric_name

        ncols = len(voxel_name_list)
        nrows = len(metric_name_list)
        print(f'Creating {nrows}x{ncols} multipanel figure for voxel 2D histograms...')
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (3.5*ncols, 3*nrows), nrows=nrows, ncols=ncols, sharex=False)

        if ncols == 1 and nrows == 1:
            axs = [axs]
        elif nrows == 1:
            axs = axs[np.newaxis, :]
        elif ncols == 1:
            axs = axs[:, np.newaxis]

        cmap = plt.get_cmap('plasma')
        cmap = cmap.with_extremes(bad=cmap(0))
        histogram_data_all = []  # [(h, xedges, yedges, counts), ...]

        global_hmin = np.inf
        global_hmax = -np.inf
        for row_idx, metric_name in enumerate(metric_name_list):
            unit_change_row = unit_change[row_idx] 
            voxel_range_row = voxel_range[row_idx] if voxel_range is not None else None
            scale_row = scale[row_idx]
            print(unit_change_row, voxel_range_row)
            # ✅ SINGLE PASS: Collect data and create histograms
            for axis_idx in range(ncols):
                voxel_name_axis = voxel_name_list[axis_idx]
                unit_change_axis = unit_change_row[axis_idx]
                voxel_range_axis = voxel_range_row[axis_idx] if voxel_range_row is not None else None
                scale_axis = scale_row[axis_idx]
                print(f'after chosing axis {axis_idx}:', unit_change_axis, voxel_range_axis, voxel_name_axis, scale_axis)
                # Collect data for this axis
                all_voxel = []
                all_metrics = []
                Z = np.zeros(M.shape)
                j = 0
                for panel_idx, analysis_dict in enumerate(analysis_list):
                    i = 0
                    for v in analysis_dict.keys():
                        if 'V' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                            continue
                        
                        metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                        voxel = self._get_metric_from_sources(analysis_dict, v, voxel_name_axis)
                        
                        if metric is None or voxel is None:
                            continue

                        # Apply mask if provided
                        if metric_masks is not None:
                            metric_mask = metric_masks[axis_idx] if axis_idx < len(metric_masks) else None
                            if metric_mask is not None:
                                analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                                analysis = self.get_analysis(analysis_name)
                                mask = self.resolve_mask(analysis, metric_mask, v)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    print(f"Axis {axis_idx}: Applying mask for '{metric_name}' at v={v}, sum={np.sum(mask)}")
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                        
                        # Process array metrics
                        if isinstance(metric, (list, tuple)):
                            if len(metric) > 2:
                                metric = metric[1:-1]
                            if len(metric) == 0:
                                continue             
                            data_m = [float(np.mean(m * unit_change_axis[1])) for m in metric if np.size(m) > 0]
                        
                        elif isinstance(metric, np.ndarray) and metric.ndim > 0:
                            metric = metric.flatten()
                            data_m = (metric * unit_change_axis[1]).tolist()
                        
                        else:
                            capture_count = self._get_metric_from_sources(analysis_dict, v, 'n_captured')
                            data_m = [metric * unit_change_axis[1]] * capture_count

                        if data_m is not None:
                            Z[i, j] = np.mean(data_m) if len(data_m) > 1 else 0
                        i += 1
                    j += 1
                print(f"Axis {axis_idx}: Collected {len(all_voxel)} data points for '{metric_name}'")
                


        # Create normalization based on global min/max
        if v_lim[0] is not None:
            vmin = v_lim[0]
        else:
            vmin = global_hmin if global_hmin != np.inf else 1
        
        if v_lim[1] is not None:
            vmax = v_lim[1]
        else:
            vmax = global_hmax if global_hmax != -np.inf else None

        if scale_row[0][2] == 'log' and scale_row[1][2] == 'log':
            from matplotlib.colors import LogNorm
            norm_obj = LogNorm(vmin=vmin, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            norm_obj = Normalize(vmin=vmin, vmax=vmax)
        
        # ✅ Plot using stored histogram data
        pcm_list = []
        clabel = 'Occurrence' if normalize else 'Counts'
        for row_idx in range(nrows):
            histogram_data = histogram_data_all[row_idx]
            for axis_idx in range(ncols):
                print([row_idx, axis_idx])
                ax = axs[row_idx, axis_idx]

                if custom_voxel_bins is not False:
                    ax.set_xticks(unique_voxel_list[axis_idx])
                    ax.set_xticklabels([f'{x:.0f}' for x in unique_voxel_list[axis_idx]], rotation=45, ha='right')

                if histogram_data[axis_idx] is None:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
                    continue
                
                h_plot, xedges, yedges, counts = histogram_data[axis_idx]
                # Plot with shared normalization
                pcm = ax.pcolormesh(
                    xedges, yedges, h_plot.T,
                    cmap=cmap,
                    norm=norm_obj,
                    rasterized=True,
                    shading='auto'
                )
                pcm_list.append(pcm)
                
                # Annotations
                if annotate_bins:
                    if scale[row_idx][axis_idx][0] == 'log':
                        x_centers = np.sqrt(xedges[:-1] * xedges[1:])
                    else:
                        x_centers = (xedges[:-1] + xedges[1:]) / 2
                    if scale[row_idx][axis_idx][1] == 'log':
                        y_centers = np.sqrt(yedges[:-1] * yedges[1:])
                    else:
                        y_centers = (yedges[:-1] + yedges[1:]) / 2
                        
                    if normalize_annotation and annotate_norm_list:

                        counts = counts / np.sum(annotate_norm_list) if np.sum(annotate_norm_list) != 0 else counts
                        counts = counts * 1e2 # Scale to percentage
                    for i in range(len(x_centers)):
                        for j in range(len(y_centers)):
                            count = counts[i, j]
                            
                            if count > 0:
                                text = f'{int(np.round(count))}' if count >= 0.9 else f'{np.round(count, 1)}' 
                                color = 'white' 
                                
                                ax.text(x_centers[i], y_centers[j], text,
                                    ha='center', va='center',
                                    fontsize=8, color=color)
                
                # Set scales
                if scale[row_idx][axis_idx][1] == 'log':
                    ax.set_yscale('log')
                if scale[row_idx][axis_idx][0] == 'log':
                    ax.set_xscale('log')
                print(f' finished axis {axis_idx} in row {row_idx}')

        fig.colorbar(pcm_list[0], ax=axs.ravel().tolist(), label=clabel, aspect=20)
                

        for row_idx in range(nrows):
            for axis_idx in range(ncols):
                ax = axs[row_idx, axis_idx]
                ax.set_xlabel(voxel_xlabel_list[axis_idx])
                if axis_idx == 0:
                    ax.set_ylabel(metric_ylabel[row_idx])
        if ncols > 1:
            for row_idx in range(nrows):
                for col_ in range(1, ncols):
                    axs[row_idx, col_].tick_params(labelleft=False)
        if nrows > 1:
            for col_ in range(ncols):
                for row_idx in range(nrows - 1):
                    axs[row_idx, col_].tick_params(labelbottom=False)
        # Add single colorbar
        fig.subplots_adjust(right=0.77, wspace=0.02, hspace=0.02)


        if save:
            if save_as is not None:
                figname = save_as
            else:
                voxel_str = '_'.join(voxel_name_list) if isinstance(voxel_name, list) else voxel_name
                figname = f'multiple_analysis_{metric_name}_2dhistogram_{voxel_str}_{metric_bins*voxel_bins}_{metric_range[1] if metric_range is not None else "nolim"}.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')

    def multiple_analysis_metric_array_wrt_m_twinaxis_violin(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], unit_change=[1e-3, 1, 1], global_ylim=False, metric_masks=None, fig_size=None, save_as=None, save=True):
            """
            Plot array metrics as violin plots with twin y-axes.
            
            OPTION B: One violin per analysis.
            - X-axis: mC values (one position per analysis)
            - Left y-axis: metric_lists[0] (e.g., semi-major axis)
            - Right y-axis: metric_lists[1] (e.g., eccentricity)
            - Each violin aggregates all v_inf data for that analysis
            
            Parameters:
            -----------
            metric_lists : list of lists
                [[left_metrics], [right_metrics]] - metrics for left and right y-axes
            metric_ylabels : list of str
                [left_ylabel, right_ylabel]
            metric_labels : list of lists, optional
                [[left_labels], [right_labels]]
            scale : list of str
                [left_scale, right_scale] - 'log' or 'linear'
            unit_change : list
                [mC_unit, left_metric_unit, right_metric_unit]
            """
            if analysis_key == 'All':
                analysis_list = list(self.analysis_dicts.values())
            elif isinstance(analysis_key, list):
                analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            
            # Get mC values for each analysis
            zkey_values = []
            for a in analysis_list:
                zkey = self._get_metric_from_sources(a, v_key=None, metric_name=analysis_zkey)
                zkey_values.append(zkey)
            
            # Create figure
            fig, ax1 = self._figure(figsize=fig_size if fig_size is not None else (7, 4))
            ax2 = ax1.twinx()
            
            cmap = plt.get_cmap('plasma', 15)
            n, m = 3, 8
            
            # ✅ COLLECT DATA FOR BOTH AXES FIRST
            all_violin_data_left = []
            all_violin_data_right = []
            mC_positions = []  # Store sorted mC values for x-axis
            
            # Sort analyses by mC value
            sorted_indices = np.argsort(zkey_values)
            sorted_mC_values = [zkey_values[i] for i in sorted_indices]
            sorted_analyses = [analysis_list[i] for i in sorted_indices]
            
            # Collect data for each analysis (aggregating all v_inf)
            for analysis_idx, analysis_dict in enumerate(sorted_analyses):
                zkey = sorted_mC_values[analysis_idx]
                mC_positions.append(zkey * unit_change[0])  # Store for x-axis positioning
                
                data_left = []
                data_right = []
                
                # Aggregate all v_inf data for this analysis
                for v_key in analysis_dict.keys():
                    if 'V' not in v_key:
                        continue
                    
                    # LEFT AXIS metric
                    if metric_lists[0]:
                        for metric_name in metric_lists[0]:
                            metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                            
                            # Apply mask if provided
                            if metric_masks is not None and len(metric_masks) > 0 and metric_masks[0]:
                                mask = self.resolve_mask(self.get_analysis(list(self.analysis_dicts.keys())[sorted_indices[analysis_idx]]), metric_masks[0][0], v_key)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                            
                            if isinstance(metric, (list, tuple, np.ndarray)):
                                if len(metric) > 2:
                                    metric = metric[1:-1]  # Trim outliers
                                metric = [m * unit_change[1] for m in metric if np.isfinite(m)]
                                data_left.extend(metric)
                    
                    # RIGHT AXIS metric
                    if metric_lists[1]:
                        for metric_name in metric_lists[1]:
                            metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                            
                            # Apply mask if provided
                            if metric_masks is not None and len(metric_masks) > 1 and metric_masks[1]:
                                mask = self.resolve_mask(self.get_analysis(list(self.analysis_dicts.keys())[sorted_indices[analysis_idx]]), metric_masks[1][0], v_key)
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                            
                            if isinstance(metric, (list, tuple, np.ndarray)):
                                if len(metric) > 2:
                                    metric = metric[1:-1]  # Trim outliers
                                metric = [m * unit_change[2] for m in metric if np.isfinite(m)]
                                data_right.extend(metric)
                
                all_violin_data_left.append(data_left if len(data_left) > 0 else [np.nan])
                all_violin_data_right.append(data_right if len(data_right) > 0 else [np.nan])
            
            # ✅ CREATE VIOLIN PLOTS ON BOTH AXES
            positions = np.arange(len(mC_positions))
            wd = 1
            # Left axis violins
            if all_violin_data_left and any(len(d) > 1 for d in all_violin_data_left):
                vp_left = ax1.violinplot(
                    dataset=all_violin_data_left,
                    positions=positions,
                    showmeans=True,
                    showmedians=False,
                    showextrema=True,
                    widths=wd,
                    side='low'
                )
                
                for body in vp_left['bodies']:
                    body.set_zorder(0)
                    body.set_alpha(0.6)
                    body.set_facecolor(cmap(n))
                
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    vp_left[partname].set_zorder(1)
                    vp_left[partname].set_edgecolor(cmap(n))
                    if partname == 'cmeans':
                        # vp_right[partname].set_color('crimson')
                        vp_left[partname].set_linewidth(1.5)
                    else:
                        vp_left[partname].set_linewidth(1)
                print(f"Means of left violins: {[vp_left['cmeans'].get_segments()[i][0][1] for i in range(len(vp_left['cmeans'].get_segments()))]}")
            
            # Right axis violins
            if all_violin_data_right and any(len(d) > 1 for d in all_violin_data_right):
                vp_right = ax2.violinplot(
                    dataset=all_violin_data_right,
                    positions=positions,
                    showmeans=True,
                    showmedians=False,
                    showextrema=True,
                    widths=wd,
                    side='high'
                )
                
                for body in vp_right['bodies']:
                    body.set_zorder(0)
                    body.set_alpha(0.6)
                    body.set_facecolor(cmap(n + m))
                
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    vp_right[partname].set_zorder(1)
                    vp_right[partname].set_edgecolor(cmap(n + m))
                    if partname == 'cmeans':
                        # vp_right[partname].set_color('crimson')
                        vp_right[partname].set_linewidth(1.5)
                    else:
                        vp_right[partname].set_linewidth(1)
            print(f"Means of right violins: {[vp_right['cmeans'].get_segments()[i][0][1] for i in range(len(vp_right['cmeans'].get_segments()))]}")
            # ✅ SET X-AXIS TICKS TO SHOW mC VALUES
            ax1.set_xticks(positions)
            mC_labels = [f'{plu.sci_notation_latex(mC)}' for mC in mC_positions]
            ax1.set_xticklabels(mC_labels)
            ax1.set_xlabel(f'{analysis_zlabel[0]} [{analysis_zlabel[1]}]', fontsize=11)
            
            # ✅ SET Y-AXIS SCALES
            ax1.set_yscale(scale[0])
            ax2.set_yscale(scale[1])
            adjusted_n = plu.adjust_color(cmap(n), 1.3)  # Darker shade for ticks
            adjusted_nm = plu.adjust_color(cmap(n+m), 1.5)  # Darker shade for ticks
            # ✅ SET Y-AXIS LABELS
            ax1.set_ylabel(metric_ylabels[0], fontsize=11, color=adjusted_n)
            ax2.set_ylabel(metric_ylabels[1], fontsize=11, color=adjusted_nm, rotation=270, labelpad=15)
            
            # ✅ STYLE SPINES AND TICKS

            ax1.spines['left'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)
            
            # Color spines
            ax1.spines['left'].set_color(cmap(n))
            ax2.spines['right'].set_color(cmap(n+m))

            ax1.minorticks_off()
            ax2.minorticks_off()

            # Color y-axis labels
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)

            ax1.tick_params(axis='y', which='both', colors=adjusted_n, labelcolor=adjusted_n)
            ax2.tick_params(axis='y', which='both', colors=adjusted_nm, labelcolor=adjusted_nm)
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)
            
            # ✅ GLOBAL Y-LIMITS IF REQUESTED
            if global_ylim:
                ylim_left = ax1.get_ylim()
                ylim_right = ax2.get_ylim()
                ax1.set_ylim(ylim_left)
                ax2.set_ylim(ylim_right)
            
            # # ✅ LEGEND
            # if metric_labels:
            #     from matplotlib.patches import Rectangle
            #     handles = []
            #     labels = []
            #     for label in metric_labels[0]:
            #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n), alpha=0.6))
            #         labels.append(label)
            #     for label in metric_labels[1]:
            #         handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(n + m), alpha=0.6))
            #         labels.append(label)
            #     ax1.legend(handles, labels, frameon=False, fontsize=9, loc='upper left')
            
            fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)
            
            if save:
                metricsname = "_".join([metric_ylabels[0].replace(' ', '_'), metric_ylabels[1].replace(' ', '_')])
                figname = save_as if save_as else f'multiple_analysis_{metricsname}_twinaxis_vs_mC.png'
                out = os.path.join(self.plots_dir, figname)
                fig.savefig(out, dpi=300, bbox_inches='tight')

    def multiple_analysis_metric_array_wrt_v_m_twinaxis_violin(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], unit_change=[1, 1, 1], global_ylim=False, metric_masks=None, fig_size=None, save_as=None, save=True):
            """
            Plot array metrics as violin plots with twin y-axes.
            
            OPTION A: One violin per v_inf value, grouped/colored by analysis mC.
            - X-axis: v_inf values (one position per velocity bin)
            - Violins per x-position: One violin per analysis (mC), offset or colored differently
            - Left y-axis: metric_lists[0] (e.g., semi-major axis)
            - Right y-axis: metric_lists[1] (e.g., eccentricity)
            - Each violin shows the distribution of metrics at that v_inf for that mC
            
            Parameters:
            -----------
            metric_lists : list of lists
                [[left_metrics], [right_metrics]] - metrics for left and right y-axes
            metric_ylabels : list of str
                [left_ylabel, right_ylabel]
            metric_labels : list of lists, optional
                [[left_labels], [right_labels]]
            scale : list of str
                [left_scale, right_scale] - 'log' or 'linear'
            unit_change : list
                [v_inf_unit, left_metric_unit, right_metric_unit]
            """
            if analysis_key == 'All':
                analysis_list = list(self.analysis_dicts.values())
            elif isinstance(analysis_key, list):
                analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            
            # Get mC values for each analysis
            zkey_values = []
            for a in analysis_list:
                vkey = list(a.keys())[0]
                zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
                zkey_values.append(zkey)
            
            # Sort by mC for consistent coloring
            sorted_indices = np.argsort(zkey_values)
            sorted_mC_values = [zkey_values[i] for i in sorted_indices]
            sorted_analyses = [analysis_list[i] for i in sorted_indices]
            
            # Get all v_inf values (same for all analyses)
            v_keys_all = sorted(set([v for analysis_dict in sorted_analyses for v in analysis_dict.keys() if 'V' in v]))
            
            # Create figure
            fig, ax1 = self._figure(figsize=fig_size if fig_size is not None else (10, 4))
            ax2 = ax1.twinx()
            
            cmap = plt.get_cmap('plasma', len(sorted_mC_values))
            n, m = 3, 8
            
            # ✅ COLLECT DATA FOR EACH V_INF ACROSS ALL ANALYSES
            v_positions = np.arange(len(v_keys_all))
            n_analyses = len(sorted_mC_values)
            violin_width = 0.8 / n_analyses  # Divide space between analyses
            
            all_violin_data_left_by_v = [[] for _ in v_keys_all]   # [v1_data, v2_data, ...]
            all_violin_data_right_by_v = [[] for _ in v_keys_all]
            
            # For each v_inf value
            for v_idx, v_key in enumerate(v_keys_all):
                # For each analysis (mC value)
                for analysis_idx, analysis_dict in enumerate(sorted_analyses):
                    zkey = sorted_mC_values[analysis_idx]
                    
                    data_left = []
                    data_right = []
                    
                    if v_key not in analysis_dict:
                        # If this analysis doesn't have this v_inf, use NaN placeholder
                        all_violin_data_left_by_v[v_idx].append([np.nan])
                        all_violin_data_right_by_v[v_idx].append([np.nan])
                        continue
                    
                    # LEFT AXIS metric
                    if metric_lists[0]:
                        for metric_name in metric_lists[0]:
                            metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                            
                            # Apply mask if provided
                            if metric_masks is not None and len(metric_masks) > 0 and metric_masks[0]:
                                mask = self.resolve_mask(
                                    self.get_analysis(list(self.analysis_dicts.keys())[sorted_indices[analysis_idx]]),
                                    metric_masks[0][0], 
                                    v_key
                                )
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                            
                            if isinstance(metric, (list, tuple, np.ndarray)):
                                if len(metric) > 2:
                                    metric = metric[1:-1]
                                metric = [m * unit_change[1] for m in metric if np.isfinite(m)]
                                data_left.extend(metric)
                    
                    # RIGHT AXIS metric
                    if metric_lists[1]:
                        for metric_name in metric_lists[1]:
                            metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                            
                            # Apply mask if provided
                            if metric_masks is not None and len(metric_masks) > 1 and metric_masks[1]:
                                mask = self.resolve_mask(
                                    self.get_analysis(list(self.analysis_dicts.keys())[sorted_indices[analysis_idx]]),
                                    metric_masks[1][0],
                                    v_key
                                )
                                if mask is not None:
                                    mask = np.asarray(mask)
                                    metric = [m for j, m in enumerate(metric) if mask[j]]
                            
                            if isinstance(metric, (list, tuple, np.ndarray)):
                                if len(metric) > 2:
                                    metric = metric[1:-1]
                                metric = [m * unit_change[2] for m in metric if np.isfinite(m)]
                                data_right.extend(metric)
                    
                    all_violin_data_left_by_v[v_idx].append(data_left if len(data_left) > 0 else [np.nan])
                    all_violin_data_right_by_v[v_idx].append(data_right if len(data_right) > 0 else [np.nan])
            
            # ✅ CREATE VIOLIN PLOTS FOR EACH V_INF
            for v_idx, v_key in enumerate(v_keys_all):
                v_inf = self._get_metric_from_sources(sorted_analyses[0], v_key, 'v_inf')
                v_pos_base = v_positions[v_idx]
                
                # Left axis violins (side-by-side at this v position)
                if all_violin_data_left_by_v[v_idx] and any(len(d) > 1 for d in all_violin_data_left_by_v[v_idx]):
                    positions_left = [v_pos_base - 0.4 + i * violin_width for i in range(len(sorted_mC_values))]
                    
                    vp_left = ax1.violinplot(
                        dataset=all_violin_data_left_by_v[v_idx],
                        positions=positions_left,
                        showmeans=True,
                        showmedians=False,
                        showextrema=True,
                        widths=violin_width * 0.8,
                        side='low'
                    )
                    
                    for body_idx, body in enumerate(vp_left['bodies']):
                        body.set_zorder(0)
                        body.set_alpha(0.6)
                        body.set_facecolor(cmap(body_idx))
                    
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        for line_idx, line in enumerate(vp_left[partname]):
                            line.set_zorder(1)
                            line.set_edgecolor(cmap(body_idx))
                            if partname == 'cmeans':
                                line.set_color('crimson')
                            line.set_linewidth(1.5)
                
                # Right axis violins (side-by-side at this v position)
                if all_violin_data_right_by_v[v_idx] and any(len(d) > 1 for d in all_violin_data_right_by_v[v_idx]):
                    positions_right = [v_pos_base + 0.05 + i * violin_width for i in range(len(sorted_mC_values))]
                    
                    vp_right = ax2.violinplot(
                        dataset=all_violin_data_right_by_v[v_idx],
                        positions=positions_right,
                        showmeans=True,
                        showmedians=False,
                        showextrema=True,
                        widths=violin_width * 0.8,
                        side='high'
                    )
                    
                    for body_idx, body in enumerate(vp_right['bodies']):
                        body.set_zorder(0)
                        body.set_alpha(0.6)
                        body.set_facecolor(cmap(body_idx))
                    
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        for line_idx, line in enumerate(vp_right[partname]):
                            line.set_zorder(1)
                            line.set_edgecolor(cmap(body_idx))
                            if partname == 'cmeans':
                                line.set_color('crimson')
                            line.set_linewidth(1.5)
            
            # ✅ SET X-AXIS TICKS TO SHOW V_INF VALUES
            ax1.set_xticks(v_positions)
            v_labels = [f'{self._get_metric_from_sources(sorted_analyses[0], v, "v_inf")/1e3:.1f}' for v in v_keys_all]
            ax1.set_xticklabels(v_labels, rotation=45, ha='right')
            ax1.set_xlabel(r'v$_\infty$ [km/s]', fontsize=11)
            
            # ✅ SET Y-AXIS SCALES
            ax1.set_yscale(scale[0])
            ax2.set_yscale(scale[1])
            
            # ✅ SET Y-AXIS LABELS
            adjusted_n = cmap(0)
            adjusted_nm = cmap(n_analyses - 1)
            ax1.set_ylabel(metric_ylabels[0], fontsize=11, color=adjusted_n)
            ax2.set_ylabel(metric_ylabels[1], fontsize=11, color=adjusted_nm)
            
            # ✅ STYLE SPINES AND TICKS
            ax1.spines['left'].set_color(adjusted_n)
            ax2.spines['right'].set_color(adjusted_nm)
            ax1.spines['left'].set_linewidth(2)
            ax2.spines['right'].set_linewidth(2)
            
            ax1.tick_params(axis='y', which='both', colors=adjusted_n, labelcolor=adjusted_n)
            ax2.tick_params(axis='y', which='both', colors=adjusted_nm, labelcolor=adjusted_nm)
            ax1.yaxis.label.set_color(adjusted_n)
            ax2.yaxis.label.set_color(adjusted_nm)
            
            # ✅ GLOBAL Y-LIMITS IF REQUESTED
            if global_ylim:
                ylim_left = ax1.get_ylim()
                ylim_right = ax2.get_ylim()
                ax1.set_ylim(ylim_left)
                ax2.set_ylim(ylim_right)
            
            # ✅ LEGEND (mC values)
            from matplotlib.patches import Rectangle
            handles = [Rectangle((0, 0), 1, 1, facecolor=cmap(i), alpha=0.6) for i in range(len(sorted_mC_values))]
            labels = [f'{analysis_zlabel[0]}={mC/const.M_sun.value:.1e} {analysis_zlabel[1]}' for mC in sorted_mC_values]
            ax1.legend(handles, labels, frameon=False, fontsize=9, loc='upper left', ncol=2)
            
            fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)
            
            if save:
                metricsname = "_".join([metric_ylabels[0].replace(' ', '_'), metric_ylabels[1].replace(' ', '_')])
                figname = save_as if save_as else f'multiple_analysis_{metricsname}_twinaxis_vs_v_colored_mC.png'
                out = os.path.join(self.plots_dir, figname)
                fig.savefig(out, dpi=300, bbox_inches='tight')

    
    def multiple_analysis_metric_array_wrt_twinxaxis_linehist(self, metric_lists, metric_xlabels, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', bins=[50, 50], bin_scale=['log', 'log'], metric_scale=['linear', 'linear'], unit_change=[1, 1, 1], global_ylim=False, metric_range=None, metric_masks=None, normalization=1,fig_size=None, save_as=None, save=True):
        """
        Plot array metrics as line plots with twin x-axes, including histograms of combined v_inf distributions for each analysis represented with a different line.
        
        - Top X-axis: metric_lists[0] (e.g., semi-major axis)
        - Bottom X-axis: metric_lists[1] (e.g., eccentricity
        - Right y-axis: counts of bottom x-axis metric (e.g., eccentricity) for each v_inf, aggregated across all analyses (mC values)
        - Left y-axis: counts of top x-axis metric (e.g., semi-major axis) for each v_inf, aggregated across all analyses (mC values)
        - Each line shows the counts of metrics of all v_inf for that mC, with different lines for different analyses (mC values)
        - Histograms of v_inf distributions can be plotted as insets or as separate subplots
        
        Parameters:
        -----------
        metric_lists : list of lists
            [[left_metrics], [right_metrics]] - metrics for left and right y-axes
        metric_ylabels : list of str
            [left_ylabel, right_ylabel]
        analysis_key : str or list
            'All' or list of analysis names to include
        analysis_zkey : str
            Key to use for z-axis values (e.g., mC) to differentiate analyses
        analysis_zlabel : list
            [zlabel_name, zlabel_unit] for the z-axis metric (e.g., mC)
        v_key : str
            Key to use for v_inf values in the analysis dictionaries
        scale : list of str
            [left_scale, right_scale] - 'log' or 'linear'
        unit_change : list
            [z_unit, left_metric_unit, right_metric_unit]
        global_ylim : bool
            Whether to set global y-limits across all analyses
        metric_range : list of tuples, optional
            [(bottom_min, bottom_max), (top_min, top_max)] - ranges for histogram bins.
            If None, uses auto-range from data
        metric_masks : list of lists, optional
            [[left_metric_masks], [right_metric_masks]] - masks to apply to metrics for left
            and right axes, where each mask can be a function or an array of booleans
        fig_size : tuple, optional
            Figure size (width, height) in inches
        save_as : str, optional
            Filename to save the figure as (without path)
        save : bool
            Whether to save the figure
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
            analysis_names = list(self.analysis_dicts.keys())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            analysis_names = [name for name in analysis_key if name in self.analysis_dicts]
        # Get mC values for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, None, analysis_zkey)
            zkey_values.append(zkey)
        
        # Sort by mC for consistent coloring
        sorted_indices = np.argsort(zkey_values)
        sorted_mC_values = [zkey_values[i] for i in sorted_indices]
        sorted_analyses = [analysis_list[i] for i in sorted_indices]
        
        # Create figure with twin axes
        fig, ax_bottom = self._figure(figsize=fig_size if fig_size is not None else (3.5, 3))
        ax_top = ax_bottom.twiny()  # Top x-axis (second metric) - shares y-axis with bottom
        
        cmap = plt.get_cmap('plasma', len(sorted_mC_values))
        n, m = 3, 8
        
        # ✅ COLLECT ALL METRICS FOR EACH ANALYSIS ACROSS ALL V_INF
        # For each analysis, collect all metric values
        all_metrics_bottom = [[] for _ in sorted_mC_values]  # metric_lists[0] values
        all_metrics_top = [[] for _ in sorted_mC_values]     # metric_lists[1] values
        
        for analysis_idx, analysis_dict in enumerate(sorted_analyses):
            # Collect metrics for this analysis across all v_inf
            analysis_name = analysis_names[sorted_indices[analysis_idx]]
            for v_key in analysis_dict.keys():
                if 'V' not in v_key:
                    continue
                
                # BOTTOM X-AXIS metric (metric_lists[0])
                if metric_lists[0]:
                    for metric_name in metric_lists[0]:
                        metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and len(metric_masks) > 0 and metric_masks[0]:
                            
                            mask = self.resolve_mask(self.get_analysis(analysis_name), metric_masks[0][0], v_key)
                            if mask is not None:
                                mask = np.asarray(mask)
                                metric = [m for j, m in enumerate(metric) if mask[j]]
                                print(f"Applied mask for analysis {analysis_name}, v_key {v_key}, metric {metric_name}: {len(metric)} values after masking")
                        
                        if isinstance(metric, (list, tuple, np.ndarray)):
                            # Handle time-dependent metrics: take average
                            for m in metric:
                                if isinstance(m, (list, tuple, np.ndarray)):
                                    m = np.mean(m)  # Average if array
                                m = m * unit_change[1]
                                if np.isfinite(m) and m > 0:
                                    all_metrics_bottom[analysis_idx].append(m)
                
                # TOP X-AXIS metric (metric_lists[1])
                if metric_lists[1]:
                    for metric_name in metric_lists[1]:
                        metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and len(metric_masks) > 1 and metric_masks[1]:
                            if metric_masks[1][0] == 'Same':
                                print(f"Using same mask for top metric as bottom metric for analysis {analysis_name}, v_key {v_key}, metric {metric_name}")
                            else:
                                mask = self.resolve_mask(self.get_analysis(analysis_name), metric_masks[1][0], v_key)
                            if mask is not None:
                                mask = np.asarray(mask)
                                metric = [m for j, m in enumerate(metric) if mask[j]]
                                print(f"Applied mask for analysis {analysis_name}, v_key {v_key}, metric {metric_name}: {len(metric)} values after masking")
                        if isinstance(metric, (list, tuple, np.ndarray)):
                            # Handle time-dependent metrics: take average
                            for m in metric:
                                if isinstance(m, (list, tuple, np.ndarray)):
                                    m = np.mean(m)  # Average if array
                                m = m * unit_change[2]
                                if np.isfinite(m) and m > 0:
                                    all_metrics_top[analysis_idx].append(m)
        
        # ✅ CREATE HISTOGRAMS AND PLOT LINES FOR EACH ANALYSIS
        n_bins_bottom = bins[0] if isinstance(bins, (list, tuple)) else bins
        n_bins_top = bins[1] if isinstance(bins, (list, tuple)) and len(bins) > 1 else bins[0] if isinstance(bins, (list, tuple)) else bins
        
        # Determine global bin ranges
        all_bottom_flat = np.concatenate(all_metrics_bottom) if all_metrics_bottom and len(all_metrics_bottom[0]) > 0 else np.array([])
        all_top_flat = np.concatenate(all_metrics_top) if all_metrics_top and len(all_metrics_top[0]) > 0 else np.array([])
        
        # Use provided ranges or auto-detect from data
        if metric_range is not None and len(metric_range) >= 1 and metric_range[0] is not None:
            bottom_min, bottom_max = metric_range[0]
        else:
            bottom_min = all_bottom_flat.min() if len(all_bottom_flat) > 0 else 0.1
            bottom_max = all_bottom_flat.max() if len(all_bottom_flat) > 0 else 1
        
        if metric_range is not None and len(metric_range) >= 2 and metric_range[1] is not None:
            top_min, top_max = metric_range[1]
        else:
            top_min = all_top_flat.min() if len(all_top_flat) > 0 else 0.1
            top_max = all_top_flat.max() if len(all_top_flat) > 0 else 1
        
        # Create bins - use log scale if specified, else linear
        if len(all_bottom_flat) > 0:
            if metric_scale[0] == 'log' and bottom_min > 0:
                bottom_bins = np.logspace(np.log10(bottom_min), np.log10(bottom_max), n_bins_bottom)
            else:
                bottom_bins = np.linspace(bottom_min, bottom_max, n_bins_bottom)
        else:
            bottom_bins = np.linspace(0, 1, n_bins_bottom)
        
        if len(all_top_flat) > 0:
            if metric_scale[1] == 'log' and top_min > 0:
                top_bins = np.logspace(np.log10(top_min), np.log10(top_max), n_bins_top)
            else:
                top_bins = np.linspace(top_min, top_max, n_bins_top)
        else:
            top_bins = np.linspace(0, 1, n_bins_top)
        
        bottom_bin_centers = (bottom_bins[:-1] + bottom_bins[1:]) / 2
        top_bin_centers = (top_bins[:-1] + top_bins[1:]) / 2
        
        # Plot histograms for each analysis
        for analysis_idx, analysis_dict in enumerate(sorted_analyses):
            # Bottom x-axis / right y-axis
            if len(all_metrics_bottom[analysis_idx]) > 0:
                counts_bottom, _ = np.histogram(all_metrics_bottom[analysis_idx], bins=bottom_bins)
                zkey = sorted_mC_values[analysis_idx] / const.M_sun.value
                label = fr'{plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}'
                ax_bottom.plot(bottom_bin_centers, counts_bottom/normalization, marker='o', color=cmap(analysis_idx), 
                             label=label, linewidth=2, markersize=4, alpha=0.7)
            
            # Top x-axis / left y-axis (inverted)
            if len(all_metrics_top[analysis_idx]) > 0:
                counts_top, _ = np.histogram(all_metrics_top[analysis_idx], bins=top_bins)
                ax_top.plot(top_bin_centers, counts_top/normalization, marker='s', color=cmap(analysis_idx), 
                           linewidth=2, markersize=4, alpha=0.7, linestyle='--')
        
        # ✅ SET SCALES AND LABELS
        if metric_scale[0] == 'log':
            ax_bottom.set_xscale('log')
        if metric_scale[1] == 'log':
            ax_top.set_xscale('log')
        if bin_scale[0] == 'log':
            ax_bottom.set_yscale('log')
        
        ax_bottom.set_ylabel('Counts', fontsize=11)
        
        ax_bottom.set_xlabel(f'{metric_xlabels[0]}', fontsize=11)
        ax_top.set_xlabel(f'{metric_xlabels[1]}', fontsize=11)
        
        # Legend
        ax_bottom.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
        


        if save:
            metric_bottom_name = metric_lists[0][0] if isinstance(metric_lists[0], list) else metric_lists[0]
            metric_top_name = metric_lists[1][0] if isinstance(metric_lists[1], list) else metric_lists[1]
            figname = save_as if save_as else f'multiple_analysis_{metric_bottom_name}_vs_{metric_top_name}_linehist.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'✓ Saved: {figname}')
        

    def multiple_analysis_metric_array_wrt_twinxaxis_stephist(self, metric_lists, metric_xlabels, labels=[None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', bins=[50, 50], bin_scale=['log', 'log'], metric_scale=['linear', 'linear'], unit_change=[1, 1, 1], global_ylim=False, metric_range=None, metric_masks=None, ticks=None, normalization=1, fig_size=None, save_as=None, save=True):
        """
        Plot array metrics as step histograms with twin x-axes, showing distributions for each analysis with different lines.
        
        - Top X-axis: metric_lists[0] (e.g., semi-major axis)
        - Bottom X-axis: metric_lists[1] (e.g., eccentricity)
        - Left y-axis: counts of bottom x-axis metric
        - Right y-axis: counts of top x-axis metric (inverted)
        - Each step histogram shows the counts of metrics aggregated across all v_inf for that mC value
        - Different analyses (mC values) are represented with different colored step lines
        
        Parameters:
        -----------
        metric_lists : list of lists
            [[bottom_metrics], [top_metrics]] - metrics for bottom and top x-axes
        metric_xlabels : list of str
            [bottom_xlabel, top_xlabel]
        analysis_key : str or list
            'All' or list of analysis names to include
        analysis_zkey : str
            Key to use for z-axis values (e.g., mC) to differentiate analyses
        analysis_zlabel : list
            [zlabel_name, zlabel_unit] for the z-axis metric (e.g., mC)
        v_key : str
            Key to use for v_inf values in the analysis dictionaries
        bins : list of int
            [bottom_bins, top_bins] - number of bins for each axis
        bin_scale : list of str
            [bottom_scale, top_scale] - 'log' or 'linear' scaling for bins
        metric_scale : list of str
            [bottom_scale, top_scale] - 'log' or 'linear' scaling for metric axes
        unit_change : list
            [z_unit, bottom_metric_unit, top_metric_unit]
        global_ylim : bool
            Whether to set global y-limits across all analyses
        metric_range : list of tuples, optional
            [(bottom_min, bottom_max), (top_min, top_max)] - ranges for histogram bins
        metric_masks : list of lists, optional
            [[bottom_masks], [top_masks]] - masks to apply to metrics
        normalization : float
            Factor to normalize counts by
        fig_size : tuple, optional
            Figure size (width, height) in inches
        save_as : str, optional
            Filename to save the figure as (without path)
        save : bool
            Whether to save the figure
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
            analysis_names = list(self.analysis_dicts.keys())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            analysis_names = [name for name in analysis_key if name in self.analysis_dicts]
        
        # Get mC values for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, None, analysis_zkey)
            zkey_values.append(zkey)
        
        # Sort by mC for consistent coloring
        sorted_indices = np.argsort(zkey_values)
        sorted_mC_values = [zkey_values[i] for i in sorted_indices]
        sorted_analyses = [analysis_list[i] for i in sorted_indices]
        
        # Create figure with twin axes
        fig, ax_bottom = self._figure(figsize=fig_size if fig_size is not None else (3.5, 3))
        if metric_lists[1]:  # Only create top axis if second metric is provided
            ax_top = ax_bottom.twiny()  # Top x-axis (second metric) - shares y-axis with bottom
        
        
        n, m = 3, 8
        pre_mask = [0, 0]
        post_mask = [0, 0]
        post_mask_per_analysis = []
       
        above_q_lim = 0

        # ✅ COLLECT ALL METRICS FOR EACH ANALYSIS ACROSS ALL V_INF
        all_metrics_bottom = [[] for _ in sorted_mC_values]
        all_metrics_top = [[] for _ in sorted_mC_values]
        
        for analysis_idx, analysis_dict in enumerate(sorted_analyses):
            # Collect metrics for this analysis across all v_inf
            analysis_name = analysis_names[sorted_indices[analysis_idx]]
            zkey = sorted_mC_values[analysis_idx] / const.M_sun.value
            post_mask_per_a = [0, 0]
            for v_key in analysis_dict.keys():
                if 'V' not in v_key:
                    continue
                
                # BOTTOM X-AXIS metric (metric_lists[0])
                if metric_lists[0]:
                    for metric_name in metric_lists[0]:
                        metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and len(metric_masks) > 0 and metric_masks[0]:
                            mask = np.ones_like(metric, dtype=bool)  # Start with all True
                            for m in range(len(metric_masks[0])):
                            # mask = self.resolve_mask(self.get_analysis(analysis_name), metric_masks[0][0], v_key)
                                mask_m = self.quick_mask(analysis_dict, v_key, metric_masks[0][m])
                                pre_mask[m] += np.sum(mask)
                                mask = mask & mask_m  # Combine masks with AND
                                post_mask[m] += np.sum(mask)
                                post_mask_per_a[m] += np.sum(mask)
                                print(f'{m+1} mask found, combined mask sum: {np.sum(mask)}')
                            print(f'applying mask(s) for analysis {analysis_name}, v_key {v_key}, metric {metric_name} sum: {np.sum(mask) if mask is not None else "No mask"}')
                            if mask is not None:
                                mask_final = np.asarray(mask)
                                metric = [m for j, m in enumerate(metric) if mask_final[j]]
                                print(f"Applied mask, {len(metric)} values after masking")
                                if zkey >= 1e-3:
                                    above_q_lim += len(metric)

                        
                        if isinstance(metric, (list, tuple, np.ndarray)):
                            # Handle time-dependent metrics: take average
                            for m in metric:
                                if isinstance(m, (list, tuple, np.ndarray)):
                                    m = np.mean(m)
                                    print(f"Time-dependent metric, averaged to {m}")
                                m = m * unit_change[1]
                                if np.isfinite(m) and m > 0:
                                    all_metrics_bottom[analysis_idx].append(m)
                
                # TOP X-AXIS metric (metric_lists[1])
                if metric_lists[1]:
                    for metric_name in metric_lists[1]:
                        metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and len(metric_masks) > 1 and metric_masks[1]:
                            mask = np.ones_like(metric, dtype=bool)  # Start with all True
                            for m in range(len(metric_masks[1])):
                            # mask = self.resolve_mask(self.get_analysis(analysis_name), metric_masks[1][0], v_key)
                                mask_m = self.quick_mask(analysis_dict, v_key, metric_masks[1][m])
                                mask = mask & mask_m  # Combine masks with AND
                                print(f'{m+1} mask found, combined mask sum: {np.sum(mask)}')
                            print(f'applying mask(s) for analysis {analysis_name}, v_key {v_key}, metric {metric_name} sum: {np.sum(mask) if mask is not None else "No mask"}')
                            if mask is not None:
                                mask_final = np.asarray(mask)
                                metric = [m for j, m in enumerate(metric) if mask_final[j]]
                                print(f"Applied mask, {len(metric)} values after masking")
                        
                        if isinstance(metric, (list, tuple, np.ndarray)):
                            # Handle time-dependent metrics: take average
                            for m in metric:
                                if isinstance(m, (list, tuple, np.ndarray)):
                                    m = np.mean(m)
                                m = m * unit_change[2]
                                if np.isfinite(m) and m > 0:
                                    all_metrics_top[analysis_idx].append(m)
            post_mask_per_analysis.append(post_mask_per_a)
        # ✅ CREATE HISTOGRAMS AND PLOT STEP LINES FOR EACH ANALYSIS
        n_bins_bottom = bins[0] if isinstance(bins, (list, tuple)) else bins
        n_bins_top = bins[1] if isinstance(bins, (list, tuple)) and len(bins) > 1 else bins[0] if isinstance(bins, (list, tuple)) else bins
        
        # Determine global bin ranges
        all_bottom_flat = np.concatenate(all_metrics_bottom) if all_metrics_bottom and len(all_metrics_bottom[0]) > 0 else np.array([])
        all_top_flat = np.concatenate(all_metrics_top) if all_metrics_top and len(all_metrics_top[0]) > 0 else np.array([])
        
        print(f"\nTotal bottom data points: {len(all_bottom_flat)}")
        print(f"Total top data points: {len(all_top_flat)}")
        
        # Use provided ranges or auto-detect from data
        if metric_range is not None and len(metric_range) >= 1 and metric_range[0] is not None:
            bottom_min = metric_range[0][0] if metric_range[0][0] is not None else (all_bottom_flat.min() if len(all_bottom_flat) > 0 else 0.1)
            bottom_max = metric_range[0][1] if metric_range[0][1] is not None else (all_bottom_flat.max() if len(all_bottom_flat) > 0 else 1)
        else:
            bottom_min = all_bottom_flat.min() if len(all_bottom_flat) > 0 else 0.1
            bottom_max = all_bottom_flat.max() if len(all_bottom_flat) > 0 else 1
        
        if metric_range is not None and len(metric_range) >= 2 and metric_range[1] is not None:
            top_min = metric_range[1][0] if metric_range[1][0] is not None else (all_top_flat.min() if len(all_top_flat) > 0 else 0.1)
            top_max = metric_range[1][1] if metric_range[1][1] is not None else (all_top_flat.max() if len(all_top_flat) > 0 else 1)
        else:
            top_min = all_top_flat.min() if len(all_top_flat) > 0 else 0.1
            top_max = all_top_flat.max() if len(all_top_flat) > 0 else 1
        
        # Create bins - use log scale if specified, else linear
        if len(all_bottom_flat) > 0:
            if metric_scale[0] == 'log' and bottom_min > 0:
                bottom_bins = np.logspace(np.log10(bottom_min), np.log10(bottom_max), n_bins_bottom)
            else:
                bottom_bins = np.linspace(bottom_min, bottom_max, n_bins_bottom)
        else:
            bottom_bins = np.linspace(0, 1, n_bins_bottom)
        
        if len(all_top_flat) > 0:
            if metric_scale[1] == 'log' and top_min > 0:
                top_bins = np.logspace(np.log10(top_min), np.log10(top_max), n_bins_top)
            else:
                top_bins = np.linspace(top_min, top_max, n_bins_top)
        else:
            top_bins = np.linspace(0, 1, n_bins_top)
        
        bottom_bin_centers = (bottom_bins[:-1] + bottom_bins[1:]) / 2
        top_bin_centers = (top_bins[:-1] + top_bins[1:]) / 2
        
        # Plot step histograms for each analysis
        for analysis_idx, analysis_dict in enumerate(sorted_analyses):
            # Bottom x-axis
            if len(all_metrics_bottom[analysis_idx]) > 0:
                # counts_bottom, _ = np.histogram(all_metrics_bottom[analysis_idx], bins=bottom_bins)
                zkey = sorted_mC_values[analysis_idx] / const.M_sun.value
                if labels[0] is None:
                    cmap = plt.get_cmap('plasma', len(sorted_mC_values)+2)
                    label = fr'{plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}'
                    rwidth= 1
                    histtype = 'bar'
                    align='mid'
                    hatch= [None, '///', '...', '\\\\', 'xx', 'OO', '++', '--'][analysis_idx]
                    facecolor = cmap(analysis_idx) if hatch is None else 'none'
                    edgecolor = cmap(analysis_idx)
                    color = cmap(1+analysis_idx*2)
                else:

                    cmap = plt.get_cmap('plasma', 10)                    
                    label = labels[0]
                    rwidth = 1
                    align = 'mid'
                    histtype = 'bar'
                    print(f'zkey: {zkey}, log10(zkey): {np.log10((zkey))}, cmap index: {8+np.log10((zkey))}')
                    facecolor = cmap(8)                    
                    edgecolor = 'none'
                    hatch = None
                    color =  cmap(8)
                m = all_metrics_bottom[analysis_idx] 
                # m = np.array(m)/ normalization
                ax_bottom.hist(m, bins=bottom_bins, density=False, color=color,
                               label=label, linewidth=2, alpha=1, linestyle='-', rwidth=rwidth, align=align, histtype=histtype, facecolor=facecolor, edgecolor=edgecolor, hatch=hatch)
                # ax_bottom.step(bottom_bin_centers, counts_bottom/normalization, where='mid', color=cmap(analysis_idx), 
                #              label=label, linewidth=2, alpha=0.7)
            
            # Top x-axis
            if len(all_metrics_top[analysis_idx]) > 0:
                # counts_top, _ = np.histogram(all_metrics_top[analysis_idx], bins=top_bins)
                m = all_metrics_top[analysis_idx] 
                # m = np.array(m)/ normalization
                if labels[1] is not None:
                    label = labels[1]
                else:                    
                    label = None
                ax_top.hist(m, bins=top_bins, density=False, hatch='xxx', edgecolor= cmap(6), facecolor='none',
                           label=label, linewidth=2, alpha=1)
                # ax_top.step(top_bin_centers, counts_top/normalization, where='mid', color=cmap(analysis_idx), 
                #            linewidth=2, alpha=0.7, linestyle=':')
        



        # ✅ SET SCALES AND LABELS
        if metric_scale[0] == 'log':
            ax_bottom.set_xscale('log')
        if metric_scale[1] == 'log':
            ax_top.set_xscale('log')
        if bin_scale[0] == 'log':
            ax_bottom.set_yscale('log')
        
        ax_bottom.set_ylabel('Counts', fontsize=11)
        ax_bottom.set_xlabel(f'{metric_xlabels[0]}', fontsize=11)
        ax_bottom.set_xlim(bottom_bins[0], bottom_bins[-1])
        ax_bottom.set_xticks(bottom_bins, [f'{b:.0f}' for b in bottom_bins], ha='center', rotation=30)
        ax_bottom.legend(fontsize=8, ncol=3, loc='upper right', labelspacing=0.05, columnspacing=0.5, handletextpad=0.3, borderpad=0.3, handlelength=0.7, frameon=False, bbox_to_anchor=(0.99, 1))

        if metric_lists[1]:  # Only set top x-axis label if second metric is provided
            ax_top.set_xlabel(f'{metric_xlabels[1]}', fontsize=11)
            ax_top.set_xlim(top_bins[0], top_bins[-1])
            ax_top.legend(fontsize=8, ncol=3, loc='upper right', labelspacing=0.05, columnspacing=0.5, handletextpad=0.3, borderpad=0.3, handlelength=0.7, frameon=False, bbox_to_anchor=(0.99, 0.9))
            ax_top.set_xticks(top_bins, [f'{b:.0f}' for b in top_bins], ha='center', rotation=30)
        if ticks is not None:
            if ticks[1] is not None:
                ax_bottom.set_yticks(ticks[1], [f'{plu.sci_notation_latex(t)}' for t in ticks[1]], ha='right', rotation=30)
                ax_bottom.set_ylim(0, ax_bottom.get_ylim()[1]*1.2)
        # Legend
        print(f"Pre-mask first mask: {pre_mask[0]}, second mask: {pre_mask[1]}")
        print(f"Post-mask first mask: {post_mask[0]}, second mask: {post_mask[1]}")
        print(f"Percentage within the first mask: {post_mask[0]/pre_mask[0]*100:.2f}%, second mask: {post_mask[1]/pre_mask[1]*100}%")
        print(f"Number of data points above q limit: {above_q_lim}")
        print(f"Post-mask data points per analysis: {[f'{pm[1]/pm[0]*100}%' for pm in post_mask_per_analysis]}")
        if save:
            metric_bottom_name = metric_lists[0][0] if isinstance(metric_lists[0], list) else metric_lists[0]
            metric_top_name = metric_lists[1][0] if isinstance(metric_lists[1], list) else metric_lists[1]
            figname = save_as if save_as else f'multiple_analysis_{metric_bottom_name}_vs_{metric_top_name}_stephist.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'✓ Saved: {figname}')

    
    def multiple_analysis_contour_plot(self, analysis_key='All', x_metric='semi_major_axes', y_metric='eccentricities', x_ylabel='Semi-Major Axis [AU]', y_ylabel='Eccentricity', v_key='All', x_bins=50, y_bins=50, x_range=None, y_range=None, fig_size=None, save=True): 
        """
        Plot contour plot of two metrics for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        
        # Get z-axis values (e.g., mC) for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, vkey, analysis_zkey)
            zkey_values.append(zkey)
        
        # Create multi-panel figure
        nrows = 1
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 1.5*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
        cmap = plt.get_cmap('plasma', 15)
        n, m = 3, 8

        for metric_idx, metric_name in enumerate(metric_list):
            datadict = {}
            # Process each analysis
            for panel_idx, analysis_dict in enumerate(analysis_list):
                zkey = zkey_values[panel_idx]
                
                # Get analysis object for mask resolution
                analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
                analysis = self.get_analysis(analysis_name)
                
                # Collect data for each v_inf
                for v in analysis_dict.keys():
                    if 'V' not in v:
                        continue

                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']

                    if v not in datadict.keys():
                        datadict[v] = {'v_inf': v_inf*unit_change[0], 'data': []}
                    
                    metric = self._get_metric_from_sources(analysis_dict, v, metric_name)
                    
                    # Apply mask if provided
                    if metric_masks is not None and metric_masks[axis_idx]:
                        metric_mask = metric_masks[axis_idx][metric_idx] if metric_idx < len(metric_masks[axis_idx]) else None
                        if metric_mask is not None:
                            mask = self.resolve_mask(analysis, metric_mask, v)
                            if mask is not None:
                                mask = np.asarray(mask)
                                print(f"Panel {panel_idx}, axis {axis_idx}: Applying mask for '{metric_name}' at v={v}, sum={np.sum(mask)}")
                                metric = [m for j, m in enumerate(metric) if mask[j]]
                    
                    # Process array metrics
                    if isinstance(metric, (list, tuple, np.ndarray)):
                        # Flatten and extract valid values
                        # Trim m from two ends to avoid initial or final outliers
                        if len(metric) > 2:
                            metric = metric[1:-1]
                        if len(metric) == 0:
                            continue
                        metric = [m*unit_change[1+axis_idx] for m in metric]
                        datadict[v]['data'].extend(metric)
                    else:
                        continue
                
                    if len(datadict[v]['data']) == 0:
                        print(f"v {v}, panel {panel_idx}, axis {axis_idx}: No array data for '{metric_name}'")
                        continue
                
            # Sort by v_inf
            v_array = np.asarray([datadict[v]['v_inf'] for v in datadict])
            datasets = [datadict[v]['data'] for v in datadict]
            order = np.argsort(v_array)
            pos = v_array[order]
            data_sorted = [datasets[i] for i in order]
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            x_array, y_array,
            bins=[x_bins, y_bins],
            range=[x_range, y_range]
        )
        pcm = ax.contourf(
            xedges[:-1], yedges[:-1], h.T,
            levels=20,
            cmap='viridis',
            norm="log"
        )
        fig.colorbar(pcm, ax=ax, label='Counts')

        ax.set_xlabel(x_ylabel)
        ax.set_ylabel(y_ylabel)

        if save:
            figname = f'multiple_analysis_{x_metric}_vs_{y_metric}_contour.png'
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)

    def plot_kde(self, metric_lists, metric_labels, labels=[None, None], analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', bins=[50, 50], bin_scale=['log', 'log'], metric_scale=['linear', 'linear'], global_ylim=False, metric_range=None, metric_masks=None,  normalization=1, thresh=0.1, levels=10, bw_adjust=0.6, zorder=[1,1,1], alpha=[1,1,1], linewidth=[1,1,1], linestyle=['-', '--', ':'], cmaps=None, fig_size=(3.5, 3), fill=[True,True,True], unit_change=(1, 1), reverse_axes=None, scale=('linear', 'linear'), ticks=None, ranges=None,  cut=3.0, extra_scatters=None,extra_fills=None, bh3_mass_adjustment=None, save=False, save_as=None):
        """
        Plot KDEs for multiple analyses with twin axes, showing distributions for each analysis with different lines.
        - Top X-axis: metric_lists[0] (e.g., semi-major axis)
        - Bottom X-axis: metric_lists[1] (e.g., eccentricity)
        - Each KDE shows the distribution of metrics aggregated across all v_inf for that mC value
        - Different analyses (mC values) are represented with different colored lines
        Parameters:
        -----------
        metric_lists : list of lists
            [[bottom_metrics], [top_metrics]] - metrics for bottom and top x-axes
        metric_labels : list of str
            [bottom_xlabel, top_xlabel]
        analysis_key : str or list
            'All' or list of analysis names to include
        analysis_zkey : str
            Key to use for z-axis values (e.g., mC) to differentiate analyses
        analysis_zlabel : list  
            [z_xlabel, z_ylabel]
        v_key : str
            Key to use for v_inf values in the analysis dictionaries
        bins : list of int
            [bottom_bins, top_bins] - number of bins for each axis
        bin_scale : list of str
            [bottom_scale, top_scale] - 'log' or 'linear' scaling for bins
        metric_scale : list of str
            [bottom_scale, top_scale] - 'log' or 'linear' scaling for metric axes
        global_ylim : bool          
        metric_range : list of tuples
            [(bottom_min, bottom_max), (top_min, top_max)] - range for each metric axis
        metric_masks : list of lists
            [[bottom_masks], [top_masks]] - masks to apply to metrics
        normalization : float
            Factor to normalize counts by
        thresh : float
            Threshold for KDE evaluation
        levels : int
            Number of contour levels for KDE
        bw_adjust : float   
            Bandwidth adjustment for KDE
        zorder : list
            List of z-orders for plotting each analysis 
        alpha : list
            List of alpha values for plotting each analysis
        linewidth : list    
        linestyle : list
        cmaps : list
            List of colormaps for each analysis
        fig_size : tuple        
        save_as : str
            Filename to save the figure as (without path)
        save : bool
            Whether to save the figure  
        """

        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
            analysis_names = list(self.analysis_dicts.keys())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
            analysis_names = [name for name in analysis_key if name in self.analysis_dicts]
        
        # Get mC values for each analysis
        zkey_values = []
        for a in analysis_list:
            vkey = list(a.keys())[0]
            zkey = self._get_metric_from_sources(a, None, analysis_zkey)
            zkey_values.append(zkey)
        
        # Sort by mC for consistent coloring
        sorted_indices = np.argsort(zkey_values)
        sorted_mC_values = [zkey_values[i] for i in sorted_indices]
        sorted_analyses = [analysis_list[i] for i in sorted_indices]

        pre_mask = np.zeros(len(metric_masks[0]))
        post_mask = np.zeros(len(pre_mask))
        metric_x, metric_y = [[] for _ in (analysis_list)], [[] for _ in (analysis_list)]
        metrics = [metric_x, metric_y]
        for analysis_idx, analysis_dict in enumerate(sorted_analyses):
            # Collect metrics for this analysis across all v_inf
            analysis_name = analysis_names[sorted_indices[analysis_idx]]
            zkey = sorted_mC_values[analysis_idx] / const.M_sun.value
            post_mask_per_a = [0, 0]
            
            print(f"\nProcessing analysis {analysis_name} with zkey {zkey:.3e} {analysis_zlabel[1]}")
            for v_key in analysis_dict.keys():
                if 'V' not in v_key:
                    continue
                
                if metric_lists[0]:
                    
                    for metric_name in metric_lists[0]:
                        metric = self._get_metric_from_sources(analysis_dict, v_key, metric_name)
                        
                        # Apply mask if provided
                        if metric_masks is not None and len(metric_masks) > 0 and metric_masks[0]:
                            mask = np.ones_like(metric, dtype=bool)  # Start with all True
                            for m in range(len(metric_masks[0])):
                            # mask = self.resolve_mask(self.get_analysis(analysis_name), metric_masks[0][0], v_key)
                                mask_m = self.quick_mask(analysis_dict, v_key, metric_masks[0][m])
                                pre_mask[m] += np.sum(mask)
                                mask = mask & mask_m  # Combine masks with AND
                                post_mask[m] += np.sum(mask)
                                post_mask_per_a[m] += np.sum(mask)
                                print(f'{m+1} mask found, combined mask sum: {np.sum(mask)}')
                            # print(f'applying mask(s) for analysis {analysis_name}, v_key {v_key}, metric {metric_name} sum: {np.sum(mask) if mask is not None else "No mask"}')
                            if mask is not None:
                                mask_final = np.asarray(mask)
                                metric = [m for j, m in enumerate(metric) if mask_final[j]]
                                print(f"Applied mask, {len(metric)} values after masking")
                                # if zkey >= 1e-3:
                                #     above_q_lim += len(metric)

                    

                        metrics[metric_lists[0].index(metric_name)][analysis_idx].extend(metric)

            
        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        if cmaps is None:
            ccs = ['Purples', 'Oranges', 'Blues', 'Reds']
        else:
            ccs = cmaps


        i=0

        # metrics = [[m] for m in metrics]
        for metric_x, metric_y, cc in zip(metrics[0], metrics[1], ccs):
            def process_metric(m, unit_factor):
                if isinstance(m, (list, tuple)):
                    if len(m) > 2:
                        m = m[1:-1]
                    if len(m) == 0:
                        return []
                    return [float(mm * unit_factor) for mm in m if np.size(mm) > 0]
                elif isinstance(m, np.ndarray) and m.ndim > 0:
                    return (m.flatten() * unit_factor).tolist()
                else:
                    return [float(m * unit_factor)]
            
            data_x = process_metric(metric_x, unit_change[0])
            data_y = process_metric(metric_y, unit_change[1])
            print(f"Original data points in x: {len(data_x)}")
            print(f"Original data points in y: {len(data_y)}")
            x_array = np.array(data_x)
            y_array = np.array(data_y)

            # Filter invalid
            valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
            x_array = x_array[valid_mask]
            y_array = y_array[valid_mask]
            
            print(f"{len(x_array)} valid points in x")
            print(f"{len(y_array)} valid points in y")
            
            if reverse_axes: 
                # Reverse x-axis if it is plotted on top x axis
                x_array = -x_array if reverse_axes[i][0] else x_array
                y_array = -y_array if reverse_axes[i][1] else y_array
                print(f"  Reversed axes: x reversed={reverse_axes[i][0]}, y reversed={reverse_axes[i][1]}")

            # Check variances, if zero, add a small jitter to avoid KDE failure
            if np.var(x_array) == 0:
                print("  Warning: Zero variance in x data. Adding jitter to avoid KDE failure.")
                x_array += np.random.normal(0,  np.ptp(x_array), size=x_array.shape)
                x_array = np.array(x_array)
            if np.var(y_array) == 0:
                print("  Warning: Zero variance in y data. Adding jitter to avoid KDE failure.")
                y_array = [y + np.random.normal(0,  1e-3) for y in y_array]
                print(f"  Added jitter to y data: new variance = {np.var(y_array):.2e}")
                y_array = np.array(y_array)

            if ranges is not None:
                x_mask = np.ones_like(x_array, dtype=bool)
                y_mask = np.ones_like(y_array, dtype=bool)  
                if ranges[0] is not None:
                    x_rng = (ranges[0][0]*unit_change[0], ranges[0][1]*unit_change[0])
                    axs.set_xlim(x_rng)
                    x_mask = (x_array >= x_rng[0]) & (x_array <= x_rng[1])
                if ranges[1] is not None:
                    y_rng = (ranges[1][0]*unit_change[1], ranges[1][1]*unit_change[1])
                    axs.set_ylim(y_rng)
                    y_mask = (y_array >= y_rng[0]) & (y_array <= y_rng[1])

                x_array = x_array[x_mask & y_mask]
                y_array = y_array[y_mask & x_mask]

            df = pd.DataFrame({metric_labels[0]: x_array, metric_labels[1]: y_array})
            within_percentage = np.sum([x_mask & y_mask]) / len(x_mask) * 100 if ranges is not None and ranges[0] is not None else 100
            # ============ GAUSSIAN KDE ============
            # KDE parameters
            kde_kw = dict(
                fill=fill[i],
                levels=levels,
                thresh=thresh,
                bw_adjust=bw_adjust,
                common_norm=False,
                cut=cut,
                label=None,
                alpha=alpha[i],
                linewidths=linewidth[i],
                linestyles=linestyle[i],
                zorder=zorder[i],
            )
            cmap_obj = plt.colormaps[cc].resampled(256)
            cmap_cols = cmap_obj(np.linspace(0.2, 1.0,256 ))  # Dark purple under color
            my_cmap = LinearSegmentedColormap.from_list("mycmap", cmap_cols)
            sns.kdeplot(x=metric_labels[0], y=metric_labels[1], data=df, ax=axs, cmap=my_cmap, **kde_kw)
                
            axs.set_ylabel(f'', fontsize=9)
            axs.set_xlabel(f'', fontsize=9)
            kde = gaussian_kde(np.vstack([x_array, y_array]), bw_method='scott')
            kde.set_bandwidth(kde.factor * kde_kw['bw_adjust'])  # ← Match your bw_adjust=0.7
            density = kde(np.vstack([x_array, y_array]))
            threshold = kde_kw['thresh'] * density.max()
            print(f"  {np.sum(density < threshold)} outliers detected")
            mask_outliers = density < threshold

            print(f"Average x value: {np.mean(x_array):.2f}, Average y value: {np.mean(y_array):.2f}")
            axs.scatter(x_array[mask_outliers], y_array[mask_outliers], color=plt.get_cmap(cc)(0.9), s=2, edgecolor=plt.get_cmap(cc)(0.9), linewidth=0.5, zorder=10, alpha=0.8)
            
                    # Optional overlay points
            if extra_scatters is not None and i < len(extra_scatters) and extra_scatters[i] is not None:
                s = extra_scatters[i]
                sx = np.asarray(s.get("x", []))
                sy = np.asarray(s.get("y", []))
                def _expand(v, n, default):
                    if isinstance(v, (list, tuple, np.ndarray)):
                        if len(v) != n:
                            raise ValueError(f"Length mismatch: expected {n}, got {len(v)}")
                        return list(v)
                    return [default if v is None else v] * n

                npts = len(sx)
                markers = _expand(s.get("marker", "o"), npts, "o")
                sizes = _expand(s.get("s", 30), npts, 30)
                colors = _expand(s.get("c", "k"), npts, "k")
                markerfacecolors = _expand(s.get("mfc", colors), npts, colors[0])
                edgecolors = _expand(s.get("edgecolors", "white"), npts, "white")
                linewidths = _expand(s.get("linewidths", 0.8), npts, 0.8)
                alphas = _expand(s.get("alpha", 1.0), npts, 1.0)
                text_colors = _expand(s.get("text_color", "black"), npts, "black")  # ← NEW

                for j, (xj, yj) in enumerate(zip(sx, sy)):

                    xerr_j = None
                    yerr_j = None
                    
                    if s.get("x_err") is not None:
                        x_err_tuple = s.get("x_err")
                        xerr_j = ([x_err_tuple[0][j]], [x_err_tuple[1][j]])
                    
                    if s.get("y_err") is not None:
                        y_err_tuple = s.get("y_err")
                        yerr_j = ([y_err_tuple[0][j]], [y_err_tuple[1][j]])
                    
                    axs.errorbar(
                        [xj], [yj],
                        xerr=xerr_j,
                        yerr=yerr_j,
                        capsize=s.get("capsize", 3),
                        capthick=linewidths[j],
                        marker=markers[j],
                        ms=sizes[j],
                        c=colors[j],
                        mec=edgecolors[j],
                        mfc=markerfacecolors[j],
                        linewidth=linewidths[j],
                        mew=s.get("mew", linewidths[j]),
                        alpha=alphas[j],
                        zorder=s.get("zorder", 20),
                        label=s.get("label", None) if j == 0 else None,
                    )
                
                texts = s.get("text", None)
                if texts is not None:
                    if isinstance(texts, str):
                        texts = [texts] * len(sx)
                    xytexts = s.get("xytext", (5, 5))
                    if isinstance(xytexts, tuple) and len(xytexts) == 2 and not isinstance(xytexts[0], (list, tuple, np.ndarray)):
                        xytexts = [xytexts] * len(sx)
                    elif isinstance(xytexts, (list, tuple)):
                        if len(xytexts) != len(sx):
                            raise ValueError("xytext list length must match number of points in x/y.")
                    else:
                        xytexts = [(5, 5)] * len(sx)

                    for x0, y0, t, xyoff, txt_color in zip(sx, sy, texts, xytexts, text_colors):
                        axs.annotate(
                            t, (x0, y0),
                            xytext=tuple(xyoff),
                            textcoords="offset points",
                            fontsize=s.get("fontsize", 8),
                            color=txt_color,  # ← Use custom text color
                            ha=s.get("ha", "left"),
                            va=s.get("va", "bottom"),
                            zorder=s.get("text_zorder", 21),
                            fontweight='bold'
                        )
                        
            # Optional overlay fill_between regions
            if extra_fills is not None and i < len(extra_fills) and extra_fills[i] is not None:
                # print(f"\n✓ DEBUG: Processing extra_fills for dataset {i}")
                # print(f"  extra_fills[{i}] = {extra_fills[i]}")
                f = extra_fills[i]
                
                for fill_idx, fill_region in enumerate(
                    f if isinstance(f, list) else [f]):
                    
                    x_fill = np.asarray(fill_region.get("x", []))
                    y1_fill = np.asarray(fill_region.get("y1", []))
                    y2_fill_raw = fill_region.get("y2", None)
                    y2_fill = np.asarray(y2_fill_raw) if y2_fill_raw is not None else None

                    
                    print(f"  Fill {fill_idx}: x_fill={x_fill}, y2_fill={y2_fill}")
                    
                    if len(x_fill) == 0:
                        print(f"    → Skipped (empty data)")
                        continue
                    
                    # If no y2, assume vertical fill at this x position
                    if y2_fill is None:
                        print(f"    → Drawing axvspan({x_fill[0]:.2f}, {x_fill[-1]:.2f})")
                        print(f"    → color={fill_region.get('color')}, alpha={fill_region.get('alpha')}, zorder={fill_region.get('zorder')}")
                        axs.axvspan(x_fill[0], x_fill[-1] if len(x_fill) > 1 else x_fill[0],
                                color=fill_region.get("color", "gray"),
                                alpha=fill_region.get("alpha", 0.3),
                                zorder=fill_region.get("zorder", 0),
                                label=fill_region.get("label", None) if fill_idx == 0 else None)
                    else:
                        # Between two curves
                        print(f"    → Drawing fill_between")
                        axs.fill_between(x_fill, y1_fill, y2_fill,
                                        color=fill_region.get("color", "gray"),
                                        alpha=fill_region.get("alpha", 0.3),
                                        label=fill_region.get("label", None) if fill_idx == 0 else None,
                                        hatch=fill_region.get("hatch", None),
                                        edgecolor=fill_region.get("edgecolor", "none"),
                                        linewidth=fill_region.get("linewidth", 1.0),
                                        zorder=fill_region.get("zorder", 0))

            i += 1
        # ============ FORMATTING ============

        if scale[1] == 'log':
            axs.set_yscale('log')
        if scale[0] == 'log':
            axs.set_xscale('log')

        if ticks:
            axs.set_xticks(ticks[0] if ticks and ticks[0] else None)
            axs.set_yticks(ticks[1] if ticks and ticks[1] else None)

        axs.set_xlabel(f'{metric_labels[0]}')
        axs.set_ylabel(f'{metric_labels[1]}')
        fig.tight_layout()

        if save:
            if save_as is not None:
                figname = save_as
            else:
                figname = f'{metric_labels[0].strip()}_vs_{metric_labels[1].strip()}_kde.png'
            plt.savefig(figname, dpi=300)
            print(f"Saved KDE plot as {figname}")
        plt.show()
        


    def resolve_mask(self, analysis, spec, v_key):
        if spec is None:
            return None
        if callable(spec):
            return np.asarray(spec(analysis, v_key))
        if isinstance(spec, (list, tuple, np.ndarray)) and (len(spec) == len(catalog[v_key]['rebound'])):
            return np.asarray(spec)
        if isinstance(spec, dict):
            t = spec.get("type")
            if t == "flag_contains":
                return analysis.mask_flag_contains(v_key, spec.get("substr",""), as_int=False)
            if t == "error_contains":
                return analysis.mask_error_contains(v_key, spec.get("substr",""), as_int=False)
            if t == "collision":
                return analysis.mask_collision(v_key, as_int=False)
            if t == "where":
                key = spec.get("key")
                pred = spec.get("predicate", lambda _: True)
                entry_type = spec.get("entry_type", "Rebound")
                return analysis.mask_where(v_key, key, pred, entry_type=entry_type, as_int=False)
            return None

    def _get_metric_from_sources(self, analysis_dict, v_key, metric_name):
        """Search for metric in multiple sources."""
        sp = analysis_dict

        if v_key is None:
            if sp is not None and isinstance(metric_name, list) and metric_name[0] in sp:
                    print(f'✓ Found list metric "{metric_name[0]}" in analysis_dict')
                    value = sp[metric_name[0]]
                    for mn in metric_name[1:]:
                        value = value[mn] if mn in value else None
                    return value
            if sp is not None and metric_name in sp:
                value = sp[metric_name]
                print(f'✓ Found in analysis_dict')
                return value
        an_entry = sp[v_key]
        mc = an_entry.get('mc')
        rb = an_entry.get('rebound', [])
        oc = an_entry.get('occurrences')
        tc = an_entry.get('termination_counts')
        smc = an_entry.get('sampled_mc')

        # 0) Direct entry check
        if metric_name in an_entry:
            value = an_entry[metric_name]
            print(f'✓ Found in an_entry')
            return value

        # 1) Rebound entries (returns list of arrays)
        if rb:
            try:
                metric = [rb_entry[metric_name] for rb_entry in rb 
                        if hasattr(rb_entry, 'files') and metric_name in rb_entry.files]
                if metric:
                    print(f'✓ Found in rebound: {len(metric)} entries')
                    return metric
            except Exception as e:
                print(f'✗ Rebound error: {e}')

        # 2) MC npz file (returns single array)
        if mc is not None and hasattr(mc, 'files') and metric_name in mc.files:
            # ✅ FIX: Make a COPY of the array from the npz file
            value = np.array(mc[metric_name])  # Force copy, not a view
            print(f'✓ Found in mc: shape={value.shape}')
            return value

        # 3) Occurrences
        if oc is not None and metric_name in oc:
            value = oc[metric_name]
            print(f'✓ Found in occurrences')
            return value
        
        # 4) Termination counts
        if tc is not None and metric_name in tc:
            value = tc[metric_name]
            print(f'✓ Found in termination_counts')
            return value
        # 5) Sampled MC
        if smc is not None and metric_name in smc:
            value = smc[metric_name]
            print(f'✓ Found in sampled_mc')
            return value



        print(f'✗ Metric "{metric_name}" not found')
        return np.array([])  # Return empty array instead of []

    def quick_mask(self, analysis, v_key, metric_mask):
        """Quickly create a mask based on a metric and condition."""

        key = metric_mask.get("key")
        condition = metric_mask.get("predicate", lambda _: True)
        metric = self._get_metric_from_sources(analysis, v_key, key)
        if metric is None:
            return None
        metric = np.asarray(metric)
        mask = []
        for m in metric:
            mask.append(bool(condition(m)))
        return np.array(mask)

    def cache_time_series(self, metric_name, analysis_key='All', unit_change=[1, 1]):
        """Cache time series data for a metric."""

        time_series_dict = {}
        time_series_dict[metric_name] = {}
        if analysis_key == 'All':
            analysis_list = list(self.analysis_dicts.values())
        elif isinstance(analysis_key, list):
            analysis_list = [self.analysis_dicts[name] for name in analysis_key if name in self.analysis_dicts]
        

        # Process each analysis
        for panel_idx, analysis_dict in enumerate(analysis_list):
            
            # Collect all time series data for this analysis
            all_times = []
            all_metrics = []    
    
            for v in analysis_dict.keys():
                if 'V' not in v:
                    continue
                
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['n_captured'] == 0 and self.zero_capture_excluded):
                    continue
                
                # Get time series arrays
                time_series = self._get_metric_from_sources(analysis_dict, v, 'times')
                metric_series = self._get_metric_from_sources(analysis_dict, v, metric_name)
                
                if time_series is None or metric_series is None:
                    continue
                
                # Process each system's time series
                for times, metrics in zip(time_series, metric_series):
                    if times is None or metrics is None:
                        continue
                    if len(times) == 0 or len(metrics) == 0:
                        continue
                    
                    # Flatten arrays
                    times_flat = np.asarray(times).flatten()
                    metrics_flat = np.asarray(metrics).flatten()

                    metrics_flat = metrics_flat * unit_change[1]
                    times_flat = times_flat * unit_change[0]
                    
                    # Ensure same length
                    min_len = min(len(times_flat), len(metrics_flat))
                    if min_len == 0:
                        continue
                    
                    all_times.extend(times_flat[:min_len])
                    all_metrics.extend(metrics_flat[:min_len])
            
            if len(all_times) == 0:
                print(f"Panel {panel_idx}: No data for '{metric_name}'")
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
                continue
            
            # Convert to arrays
            time_array = np.array(all_times)
            metric_array = np.array(all_metrics)
            print(f"Panel {panel_idx}: Collected {len(time_array)} data points for '{metric_name}'")

            time_series_dict[metric_name][panel_idx] = (time_array, metric_array)

        self.time_series_dicts = time_series_dict

    
    def get_cached_metrics(self):
        """Return list of cached metric names."""
        return list(self.time_series_dicts.keys())

    def has_cached_data(self, metric_name):
        """Check if metric data is cached."""
        return metric_name in self.time_series_dicts

    def clear_cache(self, metric_name=None):
        """Clear cache for specific metric or all."""
        if metric_name:
            self.time_series_dicts.pop(metric_name, None)
            print(f"Cleared cache for '{metric_name}'")
        else:
            self.time_series_dicts = {}
            print("Cleared all cached data")

    def process_timeseries_metric(self, time_series, metric_series, dt_fixed=100):
        """
        Resample time series to a fixed time step using interpolation.
        
        Parameters:
        -----------
        time_series : list of arrays
            List of time arrays (one per system)
        metric_series : list of arrays
            List of metric arrays (one per system)
        unit_change : list
            [time_unit_factor, metric_unit_factor]
        dt_fixed : float
            Fixed time step for resampling (in years)
        
        Returns:
        --------
        df : dataframe 
        """
        from scipy.interpolate import interp1d
        
        all_times = []
        all_metrics = []
        # dic = {}
        for times, metrics in zip(time_series, metric_series):
            if times is None or metrics is None:
                continue
            if len(times) == 0 or len(metrics) == 0:
                continue

            # Flatten arrays and apply unit conversion
            times_flat = np.asarray(times).flatten() 
            metrics_flat = np.asarray(metrics).flatten() 

            if len(times_flat) < 2:  # Need at least 2 points to interpolate
                continue

            # Ensure same length
            min_len = min(len(times_flat), len(metrics_flat))
            if min_len == 0:
                continue

            t_min = times_flat[0]
            t_max = times_flat[-1]
            
            # Generate new time grid
            t_uniform = np.arange(t_min, t_max, dt_fixed)
            if len(t_uniform) < 2:
                # Time series too short for resampling
                all_times.append(times_flat)
                all_metrics.append(metrics_flat)
                continue
            # Interpolate metrics onto uniform grid
            try:
                f_interp = interp1d(times_flat, metrics_flat, 
                                kind='linear',  # or 'cubic' for smoother
                                bounds_error=False, 
                                fill_value='extrapolate')
                
                metric_uniform = f_interp(t_uniform)

                all_times.append(t_uniform)
                all_metrics.append(metric_uniform)     
                # dic[i] = {'time': t_uniform, 'metric': metric_uniform}

            except Exception as e:
                print(f"  Warning: Interpolation failed for system: {e}")
                # Fall back to original data
                all_times.append(times_flat)
                all_metrics.append(metrics_flat)
            
            # df = pd.DataFrame.from_dict(dic)
        return all_times, all_metrics