import numpy as np
import pytest

from captus.utils.plotting_utils import advanced_stats


@pytest.mark.parametrize('dtype', ['=f4', '=f8', '<f4', '<f8', '>f4', '>f8'])
def test_advanced_stats_accepts_byte_orders_without_mutating_input(dtype):
    values = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0], dtype=dtype)
    original = values.tobytes()
    coordinates = np.arange(values.size, dtype=float)

    stable = advanced_stats(3, 0.1, 0.1, values, coordinates)

    np.testing.assert_array_equal(stable, [False, True, False, False, True, True, False])
    assert values.tobytes() == original
    assert values.dtype == np.dtype(dtype)


@pytest.mark.parametrize('kind', [
    'plot_capture_cross_section', 'plot_n_eqs', 'plot_coalescence_time', 'plot_all_grids',
])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_gw_plots_save_requested_theme(tmp_path, monkeypatch, kind, typeplotback, inherit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from captus.gw_capture import plotting
    from tdpy.plotting import plot_background_colors, plot_style

    monkeypatch.setattr(plotting, 'save_directory', str(tmp_path) + '/')
    masses = np.geomspace(0.01, 10.0, 4)  # [M_sun]
    velocities = np.array([100.0, 300.0, 600.0, 1000.0])  # [km/s]
    grid = np.arange(1.0, 17.0).reshape(4, 4)
    original_grid = grid.copy()
    background, foreground = plot_background_colors(typeplotback)
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            assert axis.yaxis.label.get_color() == foreground
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        plt.rcParams.update({'text.usetex': False, 'font.family': 'DejaVu Serif', 'savefig.dpi': 60})
        original_settings = dict(plt.rcParams)
        arguments = [masses, velocities, grid]
        if kind == 'plot_all_grids':
            arguments.extend([grid * 2, grid * 3])
        try:
            getattr(plotting, kind)(*arguments, 'test', 8, **options)
            assert dict(plt.rcParams) == original_settings
        finally:
            plt.close('all')
    np.testing.assert_array_equal(grid, original_grid)
    paths = list(tmp_path.glob('*.png'))
    assert len(paths) == 1
    with Image.open(paths[0]) as image:
        pixels = np.asarray(image.convert('RGB'))
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert tuple(pixels[0, 0]) == expected
        assert np.any(pixels != expected)


@pytest.fixture
def plot_manager(tmp_path, monkeypatch):
    from captus.three_body_capture.plotting import Plots

    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    data = {'test': {
        'V100': {'mc': {'v_inf': 100000.0}, 'n_captured': 1, 'metric': 1.0},
        'V200': {'mc': {'v_inf': 200000.0}, 'n_captured': 1, 'metric': 2.0},
    }}
    return Plots('theme', [], analysis_dictionaries=data)


@pytest.mark.parametrize('selection', ['class', 'call', 'scope'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('save', [False, True])
def test_three_body_plot_preserves_data_and_scopes_theme(plot_manager, selection, typeplotback, save):
    import copy
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    original_data = copy.deepcopy(plot_manager.analysis_dicts)
    plot_manager.typeplotback = typeplotback if selection == 'class' else None
    options = {'typeplotback': typeplotback} if selection == 'call' else {}
    outer_theme = typeplotback if selection == 'scope' else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        plt.rcParams['text.usetex'] = False
        original_settings = dict(plt.rcParams)
        figure = plot_manager.plot_metric_wrt_v(
            ['metric'], 'Metric', 'test', metric_labels=['Data'], save=save, **options,
        )
        try:
            FigureCanvasAgg(figure).draw()
            background, foreground = plot_background_colors(typeplotback)
            axis = figure.axes[0]
            assert figure.get_facecolor() == to_rgba(background)
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [100, 200])
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), [1, 2])
            assert dict(plt.rcParams) == original_settings
            assert plot_manager.analysis_dicts == original_data
            if save:
                path = Path(plot_manager.plots_dir) / 'test_metric_vs_v.png'
                with Image.open(path) as image:
                    expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
                    assert image.convert('RGB').getpixel((0, 0)) == expected
        finally:
            plt.close(figure)


def test_every_three_body_plot_method_has_theme_support():
    import inspect
    from captus.three_body_capture.plotting import Plots

    methods = [method for _, method in inspect.getmembers(Plots, inspect.isfunction)
               if hasattr(method, '__wrapped__')]
    assert len(methods) == 28
    for method in methods:
        parameter = inspect.signature(method).parameters['typeplotback']
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_three_body_plot_theme_restores_settings_after_error(plot_manager):
    import matplotlib.pyplot as plt
    from tdpy.plotting import plot_background_colors, plot_style

    with plot_style('white'):
        original = dict(plt.rcParams)
        with pytest.raises(TypeError):
            plot_manager.plot_metric_wrt_v(None, 'Metric', 'test', typeplotback='dark')
        assert dict(plt.rcParams) == original
        assert plot_background_colors() == ('white', 'black')
    plt.close('all')