import pytest

from atp_stats.utils import get_player_name


@pytest.mark.parametrize(
    "full_name, expected",
    [
        ("Roger Federer", "R. Federer"),
        ("Novak Djokovic", "N. Djokovic"),
        ("Carl Friedrich Gauss", "C. F. Gauss"),
    ],
)
def test_get_player_name(full_name, expected):
    """Tests that get_player_name returns the correct abbreviated name"""
    assert get_player_name(full_name) == expected
