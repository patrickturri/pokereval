from pokereval.rl.advantage import group_advantages


def test_mean_centered():
    adv = group_advantages([0.0, 1.0])
    assert abs(sum(adv)) < 1e-9
    assert adv[1] > adv[0]


def test_zero_variance_is_zero():
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_singleton_is_zero():
    assert group_advantages([0.9]) == [0.0]


def test_empty_is_empty():
    assert group_advantages([]) == []
