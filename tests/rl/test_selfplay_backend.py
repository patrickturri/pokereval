from pokereval.rl.backend import FakeBackend
from pokereval.rl.sampler import MockSamplingClient


def test_fakebackend_snapshot_returns_sampler_and_counts():
    sc = MockSamplingClient(fixed_action="check")
    be = FakeBackend(sc)
    assert be.snapshot_calls == 0
    snap = be.snapshot()
    assert snap is sc
    assert be.snapshot_calls == 1
