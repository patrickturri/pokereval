from pokereval.rl.config import RLConfig, get_preset, PRESETS


def test_presets_exist():
    # the RLVR presets plus the credential-free tabular self-play presets
    assert {"smoke", "real"} <= set(PRESETS)
    assert {"tabular", "tabular_smoke"} <= set(PRESETS)


def test_tabular_preset_is_larger_than_smoke():
    assert get_preset("tabular").num_steps > get_preset("tabular_smoke").num_steps


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
