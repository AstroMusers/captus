import pytest

from captus.utils.formatting import sci_notation_latex


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (1e4, r"10$^{4}$"),
        (2.3e4, r"$2.3 \times 10^{4}$"),
        (9.9995, r"10$^{1}$"),
    ],
)
def test_sci_notation_latex(value, expected):
    assert sci_notation_latex(value) == expected


@pytest.mark.parametrize('with_intervals', [False, True])
@pytest.mark.parametrize('custom_headers', [False, True])
@pytest.mark.parametrize('convert_units', [False, True])
def test_analysis_latex_table_preserves_values_labels_and_intervals(tmp_path, monkeypatch, capsys, with_intervals, custom_headers, convert_units):
    from copy import deepcopy
    from types import SimpleNamespace

    from astropy import constants
    from captus.utils.misc import create_analysis_latex_table

    analyses = [SimpleNamespace(results_dictionary={
        'mC': index * constants.M_sun.value,
        'plain': 1.25 * index,
        'outer': {'inner': {'rate': 2e-6 * index}},
        'errors': {'plain_ci': (1.0 * index, 1.5 * index),
                   'rate_ci': (1e-6 * index, 3e-6 * index)},
    }) for index in (1, 2)]
    original = deepcopy([analysis.results_dictionary for analysis in analyses])
    rows = [('plain',), ('outer', 'inner', 'rate')]
    labels = ['Plain value', 'Rate']
    intervals = ['plain_ci', 'rate_ci'] if with_intervals else None
    factors = [None, 1000.0] if convert_units else None
    headers = ['First', 'Second'] if custom_headers else None
    output = tmp_path / 'analysis_table.tex'
    monkeypatch.setenv('TDPY_VERBOSITY', '0')

    table = create_analysis_latex_table(
        analyses, rows, labels, ci_lists=intervals, unit_change=factors,
        cols_labels=headers, fmt='.3e', filename=output,
    )

    assert capsys.readouterr().out == ''
    assert output.read_text(encoding='utf-8') == table
    assert all(label in table for label in labels)
    if custom_headers:
        assert all(header in table for header in headers)
    else:
        assert r'M$_{\odot}$' in table
    for analysis in analyses:
        catalog = analysis.results_dictionary
        values = [catalog['plain'], catalog['outer']['inner']['rate']]
        for row_index, value in enumerate(values):
            factor = 1000.0 if convert_units and row_index == 1 else 1.0
            if with_intervals:
                lower, upper = catalog['errors'][intervals[row_index]]
                expected = f'{value * factor:.2e} ({lower * factor:.2e}, {upper * factor:.2e})'
            else:
                expected = f'{value * factor:.3e}'
            assert expected in table
    assert [analysis.results_dictionary for analysis in analyses] == original


@pytest.mark.parametrize('options', [
    {'row_labels': ['one']},
    {'ci_lists': ['interval']},
    {'unit_change': [1.0]},
    {'cols_labels': []},
])
def test_analysis_latex_table_rejects_mismatched_labels_or_options(tmp_path, options):
    from types import SimpleNamespace
    from captus.utils.misc import create_analysis_latex_table

    output = tmp_path / 'invalid_table.tex'
    arguments = dict(
        analysis_list=[SimpleNamespace(results_dictionary={'value': 1.0})],
        rows=[('value',), ('value',)], row_labels=['one', 'two'],
        cols_labels=['sample'], filename=output,
    )
    arguments.update(options)
    with pytest.raises(ValueError, match='length'):
        create_analysis_latex_table(**arguments)
    assert not output.exists()


@pytest.mark.parametrize('verbosity', ['0', '1'])
def test_analysis_latex_table_logs_only_requested_file_output(tmp_path, monkeypatch, capsys, verbosity):
    from types import SimpleNamespace
    from captus.utils.misc import create_analysis_latex_table

    monkeypatch.setenv('TDPY_VERBOSITY', verbosity)
    arguments = dict(
        analysis_list=[SimpleNamespace(results_dictionary={'value': 1.25})],
        rows=['value'], row_labels=['Value'], cols_labels=['Sample'],
    )
    table = create_analysis_latex_table(**arguments)
    assert '1.250000' in table
    assert capsys.readouterr().out == ''
    assert not list(tmp_path.iterdir())
    output = tmp_path / 'table.tex'
    assert create_analysis_latex_table(**arguments, filename=output) == table
    expected = f'Writing to {output}...\n' if verbosity == '1' else ''
    assert capsys.readouterr().out == expected
    assert output.read_text(encoding='utf-8') == table