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
from astropy import units as u
from astropy import constants as const

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
class Plots:
    def __init__(self, name, analysis):

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

    def plot_metric_wrt_v(self, metric_list, metric_ylabel, analysis_name, metric_masks=None, metric_labels=None, save=True):
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
                if mc is None:
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

    def plot_metric_wrt_v_twinaxis(self, metric_lists, metric_ylabels, analysis_name, metric_masks=None, metric_labels=None, save=True):
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

    def plot_metric_array_wrt_v(self, metric_list, metric_ylabel, analysis_name, metric_masks=None, metric_labels=None, save=True):
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
                if mc is None:
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

    def plot_metric_array_wrt_v_twinaxis(self, metric_lists, metric_ylabels, analysis_name, metric_masks=None, metric_labels=None, save=True):
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
                    if mc is None:
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
    
    def plot_metric_wrt_time(self, metric, metric_ylabel, analysis_name, metric_masks=None, v_key = 'All', system_number='All', v_key_label=None, save=True):
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
    def compare_metrics_shared_x(self, metrics=('capture_cross_section', 'termination_rate'), save=True):
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
    def plot_histogram_compare(self, array_attr='a_au', bins=30, density=True, x_label='Semi-Major Axis [AU]', y_label='Probability density', save=True):
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

    def multiple_analysis_metric_wrt_v(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = r'M$_{PBH}$', v_key = 'All', save=True):
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
                    if mc is None:
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

    def multiple_analysis_metric_wrt_v_multipanel(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = [r'M$_{PBH}$', r'M$_{\odot}$'], scale = 'log', v_key = 'All', normalize=False, save=True):
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
        fig, axs = self._figure(figsize=(3.5*ncols, 1*nrows), nrows=nrows, ncols=ncols, sharex=True)
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
                    if mc is None:
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
        for ax in axs:
            ax.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
    

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

    def multiple_analysis_metric_wrt_v(self, metric_list, metric_ylabel, metric_labels=None, analysis_key = 'All', analysis_zkey = 'mC', analysis_zlabel = r'M$_{PBH}$', scale = 'log', v_key = 'All', save=True):
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
                    if mc is None:
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

    def multiple_analysis_metric_array_wrt_v_twinaxis_multipanel(self, metric_lists, metric_ylabels, metric_labels=None, analysis_key='All', analysis_zkey='mC', analysis_zlabel=[r'M$_{PBH}$', r'M$_{\odot}$'], v_key='All', scale=['log', 'log'], metric_masks=None, save=True):
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
        fig, axs = self._figure(figsize=(7, 1.5*nrows), nrows=nrows, ncols=ncols, sharex=True)
        
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
                        if mc is None or entry['capture_count'] < 10:
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
            ax1.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
            ax2.yaxis.set_major_formatter(FuncFormatter(plu.log_tick_formatter))
            if panel_idx < nrows - 1:
                ax1.tick_params(labelbottom=False)
                ax2.tick_params(labelbottom=False)
            
            # Add analysis label
            ax2.text(0.35, 0.5, 
                    fr'{analysis_zlabel[0]}={plu.sci_notation_latex(zkey)} {analysis_zlabel[1]}',
                    transform=ax2.transAxes, rotation=0, va='center', fontsize=11)

        # Set xlabel only on bottom panel
        axs[-1].set_xlabel(r'v$_\infty$ [km s$^{-1}$]')

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
        fig.text(0.96, 0.5, metric_ylabels[1], 
                rotation=270, va='center', ha='center', fontsize=11)
    

 
        if save:
            metricsname = "_".join([plu.latex_label_key(ylabel) for ylabel in metric_ylabels])
            figname = f'multiple_analysis_{metricsname}_twinaxis_vs_v.png'
            out = os.path.join(self.plots_dir, figname)
            fig.savefig(out, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _get_metric_from_sources(self, an_entry, metric_name):
        """
        an_entry: catalog entry dict for a single v-key:
          { 'rebound': [...], 'mc': {...}, 'occurrences': {...} }
        Searches:
          - rebound list of entries (.npz or dict) and returns all matching metric values
          - mc npz/dict
          - occurrences dict
        Returns: metric array/value or None.
        """
        rb = an_entry.get('rebound') or []
        mc = an_entry.get('mc') or []
        oc = an_entry.get('occurrences') or None
        tc = an_entry.get('termination_counts') or None

        # 0) check if metric is directly in an_entry
        if metric_name in an_entry:
            return an_entry[metric_name]

        # 1) search rebound entries
        try:
            metric = [rb_entry[metric_name] for rb_entry in rb if metric_name in rb_entry.files]
            if metric:
                return metric
        except Exception:
            pass

        # 2) mc npz/dict
        try:
            metric = [mc_entry[metric_name] for mc_entry in mc if metric_name in mc_entry.files]
            if metric:
                return metric
        except Exception:
            pass

        # 3) occurrences dict
        if oc is not None and metric_name in oc:
            return oc[metric_name]
        
        if tc is not None and metric_name in tc:
            return tc[metric_name]


        return None
    

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