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