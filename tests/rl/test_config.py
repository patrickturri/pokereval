from pokereval.rl.config import RLConfig, get_preset, PRESETS


def test_presets_exist():
    assert set(PRESETS) == {"smoke", "real"}


def test_smoke_is_small():
    s = get_preset("smoke")
    assert s.num_steps <= 5
    assert s.group_size >= 2
    assert s.batch_spots >= 1


def test_real_is_larger_than_smoke():
    assert get_preset("real").num_steps > get_preset("smoke").num_steps


def test_unknown_preset_raises():
    import pytest
    with pytest.raises(KeyError):
        get_preset("nope")
