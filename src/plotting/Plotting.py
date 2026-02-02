import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from itertools import groupby
from operator import itemgetter
import os
from collections.abc import Iterable
import src.utils.plotting_utils as plu
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.ticker as mt
from astropy import units as u
from astropy import constants as const

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
class Plots:
    def __init__(self, name, analysis, **kwargs):

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
                if 'v' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                    continue
                v_inf = mc['v_inf']

                metric = self._get_metric_from_sources(entry, metric_name)

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
                    if 'v' not in v:
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
                if 'v' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                    continue
                v_inf = mc['v_inf']

                metric = self._get_metric_from_sources(entry, metric_name)

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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(entry, metric_name)
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
    
    def plot_metric_wrt_time(self, metric, metric_ylabel, analysis_name, metric_masks=None, v_key = 'All', system_number='All', v_key_label=None, fig_size=None, save=True):
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

        if v_key == 'All':
            v_keys = an.keys()
            v_key_label = True
            figname = f"{analysis_name}_all_v_{metric}_vs_time.png"
        else:
            v_keys = [v_key]
            figname = f"{analysis_name}_v{v_key}_{metric}_vs_time.png"

        for v in v_keys:
            if 'v' not in v:
                continue
            rb = an[v]['rebound']
            mc = an[v]['mc']
            oc = an[v]['occurrences']
            v_inf = mc['v_inf']
            
            try:
                metric = rb[metric_name] if metric_name in rb else (mc[metric_name] if metric_name in mc else oc[metric_name])
                times = rb['times']
            except Exception as e:
                print(f"Error retrieving metric '{metric_name}' for v={v}: {e}")
                continue
            if not isinstance(metric, (list, np.ndarray)):
                print(f"Metric '{metric_name}' for v={v} is not time series data.")
                continue
                
            y = np.asarray(metric)
            t = np.asarray(times)

            if v_key_label:
                vkey_label = f'v={v_inf/1e3:.0f} km/s'

            ax.plot(v[order], y[order], marker='o', linestyle='--', alpha=0.9, color=cmap(v_keys.index(v)), label=vkey_label, markersize=2, linewidth=1)

        ax.set_xlabel(r'Time [yr]')
        ax.set_ylabel(metric_ylabel)
        ax.legend(frameon=False)

        if save:
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig
    # Example: compare two metrics with shared x-axis across multiple subplots
    def compare_metrics_shared_x(self, metrics=('capture_cross_section', 'termination_rate'), fig_size=None, save=True):
        """
        Build stacked subplots sharing x for easier comparison.
        Each analysis contributes lines to each subplot.
        """
        fig, axes = self._figure(nrows=len(metrics), ncols=1, sharex=True, figsize=(7, 5))
        axes = np.atleast_1d(axes)
        cmap = plt.get_cmap('plasma', len(self.analysis_list) + 1)
        markers = ['o', 's', '^', '*', 'v']
        lines = [':', '--', '-', '-.', ':']

        for row, metric in enumerate(metrics):
            ax = axes[row]
            for i, analysis_dict in enumerate(self.analysis_list):
                v = np.asarray(getattr(an, 'v_vals_kms', []))
                y = np.asarray(getattr(an, metric, None) or (an.metrics.get(metric) if hasattr(an, 'metrics') else None))
                if v.size == 0 or y is None:
                    continue
                order = np.argsort(v)
                ax.plot(v[order], y[order], marker='o', linestyle='--', alpha=0.9, label=getattr(an, 'label', f'analysis_{i}'), color=cmap(i))
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel(r'v$_\infty$ [km/s]')
        # Optional: hide x tick labels on upper plots
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # Single legend combining entries from the last axis
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', frameon=False)

        if save:
            out = os.path.join(self.plots_dir, f'compare_{"_".join(metrics)}_sharedx.png')
            fig.tight_layout(h_pad=0.3)
            fig.savefig(out, dpi=300)
        return fig

    # Example: helper to plot histograms for each analysis on the same axes
    def plot_histogram_compare(self, array_attr='a_au', bins=30, density=True, x_label='Semi-Major Axis [AU]', y_label='Probability density', fig_size=None, save=True):
        """
        Each analysis should have analysis_dict attribute/field providing analysis_dict array, e.g., a_au.
        """
        fig, ax = self._figure(figsize=(7, 4))
        analysis_dict_list = self.analysis_dicts.get(analysis_name, None)
        cmap = plt.get_cmap('plasma', len(self.analysis_list) + 1)

        for i, analysis_dict in enumerate(self.analysis_dict_list):
            arr = getattr(analysis_dict, array_attr, None)
            if arr is None and hasattr(analysis_dict, 'data'):
                arr = analysis_dict.data.get(array_attr)
            if arr is None:
                continue
            ax.hist(np.asarray(arr), bins=bins, alpha=0.6, density=density, label=getattr(analysis_dict, 'label', f'analysis_{i}'), color=cmap(i))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yscale('log')
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        if save:
            out = os.path.join(self.plots_dir, f'hist_{array_attr}.png')
            fig.tight_layout()
            fig.savefig(out, dpi=300)
        return fig

    def multiple_analysis_metric_wrt_v(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = r'M$_{PBH}$', v_key = 'All', fig_size=None, save=True):
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
            mC = self._get_metric_from_sources(a[vkey], analysis_zkey)
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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

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

    def multiple_analysis_metric_wrt_v_multipanel(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = [r'M$_{PBH}$', r'M$_{\odot}$'], scale = 'log', v_key = 'All', normalize=False, fig_size=None, save=True):
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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
            zkey_values.append(zkey)
            normalization_const = self._get_metric_from_sources(a[vkey], 'importance_sampling_size')


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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

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
                
                if normalize:
                    y = y / normalization_const
                
                if scale == 'log':
                    axs[i].set_yscale('log')

                axs[i].plot(v, y, marker=markers[j], linestyle=lines[j], alpha=0.8, 
                        label=metric_label, color=cmap(j), markersize=7, linewidth=1.5,markeredgecolor='black', markeredgewidth=0.6)
                

                
                # Add mC label on right side of each panel
                axs[i].text(0.2, 0.6, fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}', transform=axs[i].transAxes,
                        rotation=0, va='center', fontsize=9)

        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Set ylabel on middle panel (or use fig.supylabel)
        # Option 1: Middle axis
        # axs[len(axs)//2].set_ylabel(metric_ylabel or 'Occurrences')
        
        # Option 2: Figure-level ylabel (better for multi-panel)
        fig.supylabel(metric_ylabel or 'Occurrences', x=0.02)
        fig.subplots_adjust(hspace=0.0, left=0.2, right=0.90, top=0.95, bottom=0.08)

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
                      bbox_to_anchor=(0.5, 1.05), loc='lower center')

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
            mC = self._get_metric_from_sources(a[vkey], analysis_zkey)
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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

                    if metric is None:
                        metric = 0
                        print(f"Metric '{metric_name}' not found for v={v}.")

                    metric = metric * unit_change[1]
                    if normalize:
                        normalization_const = self._get_metric_from_sources(entry, normalize)
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
                axs[i].text(0.1, 0.1, fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}', transform=axs[i].transAxes,
                        rotation=0, va='center', fontsize=9)

        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')
        
        # Set ylabel on middle panel (or use fig.supylabel)
        # Option 1: Middle axis
        # axs[len(axs)//2].set_ylabel(metric_ylabel or 'Occurrences')
        
        # Option 2: Figure-level ylabel (better for multi-panel)
        fig.supylabel(metric_ylabel or 'Occurrences', x=0.02)
        fig.subplots_adjust(hspace=0.0, left=0.2, right=0.90, top=0.95, bottom=0.08)

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
                      bbox_to_anchor=(0.5, 1.05), loc='lower center')

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
            mC = self._get_metric_from_sources(a[vkey], analysis_zkey)
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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf']
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
            zkey_values.append(zkey)
            if normalize:
                normalization_const = self._get_metric_from_sources(a[vkey], normalize)
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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(entry, metric_name)

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
                    if 'v' not in v:
                        continue
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    v_inf = mc['v_inf'] * unit_change[0]
                    
                    metric = self._get_metric_from_sources(entry, metric_name)
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
        fig.text(0.01, 0.5, metric_ylabels[0], rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.99, 0.5, metric_ylabels[1], rotation=270, va='center', ha='center', fontsize=11)
        
        fig.subplots_adjust(hspace=0.0, left=0.15, right=0.85, top=0.95, bottom=0.08)
    
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
            for j, label in enumerate(max(metric_labels[0], metric_labels[1], key=len)):
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
                         ncol=2, bbox_to_anchor=(0.5, 1.05), loc='lower center')
    
        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v_multipanel.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        return fig

    def multiple_analysis_metric_array_wrt_v_twinaxis_multipanel(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], unit_change=[1e-3, 1, 1], global_ylim=False, metric_masks=None, fig_size=None, save=True):
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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
            zkey_values.append(zkey)
        
        # Create multi-panel figure
        nrows = len(analysis_list)
        ncols = 1
        fig, axs = self._figure(figsize=fig_size if fig_size is not None else (7, 1.5*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
        # Make axs iterable even if single subplot
        if nrows == 1:
            axs = [axs]
        
        cmap = plt.get_cmap('plasma', 15)
        ax2_list = []
        # Process each analysis
        for panel_idx, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[panel_idx]
            
            # Get analysis object for mask resolution
            analysis_name = list(self.analysis_dicts.keys())[list(self.analysis_dicts.values()).index(analysis_dict)]
            analysis = self.get_analysis(analysis_name)
            
            # Create twin axis for this panel
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
                        if 'v' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                            continue
                        v_inf = mc['v_inf']
                        
                        metric = self._get_metric_from_sources(entry, metric_name)
                        
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
                        if isinstance(metric, (list, tuple)):
                            # Flatten and extract valid values
                            # Trim m from two ends to avoid initial or final outliers

                            if len(metric) > 2:
                                metric = metric[1:-1]
                            if len(metric) == 0:
                                continue
                            data_v = [float(np.mean(m)) for m in metric if np.size(m) > 0]
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
                        widths=5,
                        side=side
                    )
                    
                    # Style violins
                    color_idx = 3 + 8*axis_idx
                    for body in vp['bodies']:
                        body.set_zorder(0)
                        body.set_alpha(0.3)
                        body.set_facecolor(cmap(color_idx))
                    
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                        vp[partname].set_zorder(1)
                        vp[partname].set_alpha(0.8)
                        vp[partname].set_edgecolor(cmap(color_idx))
                        vp[partname].set_linewidth(1.5)
                    
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
            ax2.text(0.1, 0.1, 
                    fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                    transform=ax2.transAxes, rotation=0, va='center', fontsize=11)

            if scale[0] == 'linear':
                ax1.yaxis.set_major_formatter('{x:0.1f}')
            if scale[1] == 'linear':
                ax2.yaxis.set_major_formatter('{x:0.1f}')
        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')

        if global_ylim:
            # Left axes (semi-major axis)
            ylims_left = [ax.get_ylim() for ax in axs]
            global_ylim_left = (
                min([y[0] for y in ylims_left]), 
                max([y[1] for y in ylims_left])*1.1
            )
            for ax in axs:
                ax.set_ylim(global_ylim_left)
            
            # Right axes (eccentricity)
            ylims_right = [ax2.get_ylim() for ax2 in ax2_list]
            global_ylim_right = (
                min([y[0] for y in ylims_right]), 
                max([y[1] for y in ylims_right])*1.1
            )
            for ax2 in ax2_list:
                ax2.set_ylim(global_ylim_right)
                
            
        # Add legend if labels provided
        if metric_labels:
            handles = []
            labels = []
            # Create dummy handles for legend
            from matplotlib.patches import Rectangle
            for i, label in enumerate(metric_labels[0]):
                handles.append(Rectangle((0,0),1,1, facecolor=cmap(3), alpha=0.5))
                labels.append(label)
            for i, label in enumerate(metric_labels[1]):
                handles.append(Rectangle((0,0),1,1, facecolor=cmap(11), alpha=0.5))
                labels.append(label)
            
            axs[0].legend(handles, labels, frameon=False, fontsize=9, ncol=2,
                        bbox_to_anchor=(0.5, 1.05), loc='lower center')
        
        # Adjust layout
        fig.subplots_adjust(hspace=0.0, left=0.1, right=0.9, top=0.95, bottom=0.1)
        
        # Left y-axis label
        fig.text(0.02, 0.5, metric_ylabels[0], 
                rotation=90, va='center', ha='center', fontsize=11)
        
        # Right y-axis label
        fig.text(0.97, 0.5, metric_ylabels[1], 
                rotation=270, va='center', ha='center', fontsize=11)
    

 
        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        return fig

    def _get_metric_from_sources(self, an_entry, metric_name):
        """Search for metric in multiple sources."""
        # print(f'Searching for metric "{metric_name}"...')
        
        mc = an_entry.get('mc')
        rb = an_entry.get('rebound', [])
        oc = an_entry.get('occurrences')
        tc = an_entry.get('termination_counts')

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

        print(f'✗ Metric "{metric_name}" not found')
        return np.array([])  # Return empty array instead of []
    
    def multiple_analysis_time_series_2dhistogram(self, analysis_key='All', time_key='All', metric_name='separation', metric_ylabel='Separation [AU]', time_ylabel='Time [yr]', v_key='All', time_bins=50, metric_bins=50, time_range=None, metric_range=None, fig_size=None, save=True):
        """
        Plot 2D histogram of time series metric for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = self.analysis_dicts.values()
        elif isinstance(analysis_key, list):
            for a in self.analysis_list:
                analysis_list = [a for a in self.analysis_dicts if a.get_name() in analysis_key]

        fig, ax = self._figure(figsize=(7, 5))
        
        for i, analysis_dict in enumerate(analysis_list):
            for v in analysis_dict.keys():
                if 'v' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                    continue
                time_series = self._get_metric_from_sources(entry, 'times')
                metric_series = self._get_metric_from_sources(entry, metric_name)

                if time_series is None or metric_series is None:
                    print(f"Time series or metric series not found for v={v}.")
                    continue

                time_array = np.concatenate(time_series)
                metric_array = np.concatenate(metric_series)
            cmap = cmap.with_extremes(bad=cmap(0))
            h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
            pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                                    norm="log", vmax=1.5e2, rasterized=True)
            fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
        ax.set_xlabel(time_ylabel)
        ax.set_ylabel(metric_ylabel)
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label('Counts')

        if save:
            figname = f'multiple_analysis_{metric_name}_2dhistogram.png'
            out = os.path.join(self.plots_dir, figname)
            fig.tight_layout()
            fig.savefig(out, dpi=300)

    def multiple_analysis_time_series_2dhistogram_multipanel(self, metric_name, metric_ylabel, time_ylabel, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], scale=['linear', 'log', 'log'], v_key='All', global_ylim=True, normalize=False, metric_range=None, time_range=None, v_lim=[None, None], time_bins=1000, metric_bins=1000, unit_change=[1, 1, 1], use_existing_time_series=True, fig_size=None, save=True):
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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
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
        

        # Process each analysis
        for panel_idx, analysis_dict in enumerate(analysis_list):
            zkey = zkey_values[panel_idx]
            ax = axs[panel_idx]
            
            # Collect all time series data for this analysis
            all_times = []
            all_metrics = []
            if metric_name in self.time_series_dicts.keys() and use_existing_time_series == True:
                if panel_idx in self.time_series_dicts[metric_name].keys():
                    print(f"Panel {panel_idx}: Using existing time series data for '{metric_name}'")
                    time_array, metric_array = self.time_series_dicts[metric_name][panel_idx]
                else:
                    use_existing_time_series = False

            if metric_name not in self.time_series_dicts.keys() or use_existing_time_series == False:
                self.time_series_dicts[metric_name] = {}
                for v in analysis_dict.keys():
                    if 'v' not in v:
                        continue
                    
                    entry = analysis_dict[v]
                    mc = entry.get('mc')
                    if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                        continue
                    
                    # Get time series arrays
                    time_series = self._get_metric_from_sources(entry, 'times')
                    metric_series = self._get_metric_from_sources(entry, metric_name)
                    
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

                self.time_series_dicts[metric_name][panel_idx] = (time_array, metric_array)

            # Filter out invalid values
            valid_mask = np.isfinite(time_array) & np.isfinite(metric_array) & (metric_array > 0)
            time_array = time_array[valid_mask]
            metric_array = metric_array[valid_mask]
            
            if len(time_array) == 0:
                print(f"Panel {panel_idx}: No valid data after filtering")
                continue
            
            # Determine ranges
            if time_range is None:
                time_range_used = (time_array.min(), time_array.max())
            else:
                time_range_used = time_range
            
            if metric_range is None:
                metric_range_used = (metric_array.min(), metric_array.max())
            else:
                metric_range_used = metric_range
            
            # Create 2D histogram
            h, xedges, yedges = np.histogram2d(
                time_array, metric_array,
                bins=[time_bins, metric_bins],
                range=[time_range_used, metric_range_used]
            )
            # points_per_bin = len(time_array) / time_bins
            # count__normalization = 1 / points_per_bin
            # if normalize:
            #     h = h * count__normalization
            # Plot with pcolormesh
            if scale[2] == 'log':
                from matplotlib.colors import LogNorm
                norm_obj = LogNorm(vmin=v_lim[0] if v_lim[0] is not None else 1, vmax=v_lim[1] if v_lim[1] is not None else h.max())
            else:
                from matplotlib.colors import Normalize
                norm_obj = Normalize(vmin=v_lim[0] if v_lim[0] is not None else 0, vmax=v_lim[1] if v_lim[1] is not None else h.max())

            pcm = ax.pcolormesh(
                xedges, yedges, h.T,
                cmap=cmap,
                norm=norm_obj,
                rasterized=True,
                shading='auto'
            )
            
            # Add colorbar for each panel
            cbar = fig.colorbar(pcm, ax=ax, label='Counts', norm=norm_obj)

            
            # Set log scale for metric axis if appropriate
            if scale[1] == 'log':
                ax.set_yscale('log')
                
            
            # Add analysis label
            ax.text(0.60, 0.95, 
                fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                transform=ax.transAxes, 
                va='top', ha='left',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Only show ylabel on leftmost panels
            if panel_idx == 0:
                ax.set_ylabel(metric_ylabel)
            
            # Only show xlabel on bottom panel
            if panel_idx == nrows - 1:
                ax.set_xlabel(time_ylabel)
            else:
                ax.tick_params(labelbottom=False)

        if global_ylim:
            # Synchronize y-limits across panels
            ylims = [ax.get_ylim() for ax in axs]
            global_ylim = (min([y[0] for y in ylims]), max([y[1] for y in ylims]))
            for ax in axs:
                ax.set_ylim(global_ylim)
        
        # Adjust layout
        fig.subplots_adjust(hspace=0.1, right=0.85)
        
        if save:
            figname = f'multiple_analysis_{metric_name}_2dhistogram_multipanel.png'
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
            zkey = self._get_metric_from_sources(a[vkey], analysis_zkey)
            zkey_values.append(zkey)
            for v in a.keys():
                if 'v' in v:
                    if normalize:
                        normalization_const = self._get_metric_from_sources(a[v], normalize)
                        norm_const_list.append(normalization_const)
                    if annotate_bins and normalize_annotation:
                        annotate_norm_const = self._get_metric_from_sources(a[v], normalize_annotation)
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
                        if 'v' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                            continue
                        
                        metric = self._get_metric_from_sources(entry, metric_name)
                        voxel = self._get_metric_from_sources(entry, voxel_name_axis)

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
                            data_v = [voxel * unit_change_axis[0]] * len(data_m)
                        
                        elif isinstance(metric, np.ndarray) and metric.ndim > 0:
                            metric = metric.flatten()
                            data_m = (metric * unit_change_axis[1]).tolist()
                            data_v = [voxel * unit_change_axis[0]] * len(data_m)
                        
                        else:
                            capture_count = self._get_metric_from_sources(entry, 'capture_count')
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
                vkey_parsed = float(vkey.split('v')[1])
                v_values.append(vkey_parsed)
            for v in a.keys():
                if 'v' in v and ():
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
                        if 'v' not in v:
                            continue
                        
                        entry = analysis_dict[v]
                        mc = entry.get('mc')
                        if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                            continue
                        
                        metric = self._get_metric_from_sources(entry, metric_name)
                        voxel = self._get_metric_from_sources(entry, voxel_name_axis)
                        
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
                            capture_count = self._get_metric_from_sources(entry, 'capture_count')
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

    
    def multiple_analysis_contour_plot(self, analysis_key='All', x_metric='semi_major_axes', y_metric='eccentricities', x_ylabel='Semi-Major Axis [AU]', y_ylabel='Eccentricity', v_key='All', x_bins=50, y_bins=50, x_range=None, y_range=None, fig_size=None, save=True): 
        """
        Plot contour plot of two metrics for multiple analyses.
        """
        if analysis_key == 'All':
            analysis_list = self.analysis_dicts.values()
        elif isinstance(analysis_key, list):
            for a in self.analysis_list:
                analysis_list = [a for a in self.analysis_dicts if a.get_name() in analysis_key]

        fig, ax = self._figure(figsize=(7, 5))
        
        for i, analysis_dict in enumerate(analysis_list):
            for v in analysis_dict.keys():
                if 'v' not in v:
                    continue
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                    continue
                x_data = self._get_metric_from_sources(entry, x_metric)
                y_data = self._get_metric_from_sources(entry, y_metric)

                if x_data is None or y_data is None:
                    print(f"Metrics not found for v={v}.")
                    continue

                x_array = np.concatenate(x_data)
                y_array = np.concatenate(y_data)
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
                return analysis.mask_where(v_key, key, pred, as_int=False)
            return None

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
                if 'v' not in v:
                    continue
                
                entry = analysis_dict[v]
                mc = entry.get('mc')
                if mc is None or (entry['capture_count'] == 0 and self.zero_capture_excluded):
                    continue
                
                # Get time series arrays
                time_series = self._get_metric_from_sources(entry, 'times')
                metric_series = self._get_metric_from_sources(entry, metric_name)
                
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