import pytest

pytest.importorskip("tinker")  # build_datum / datum.py construct tinker.Datum (rl extra)

from pokereval.rl.backend import FakeBackend
from pokereval.rl.sampler import MockSamplingClient
from pokereval.rl.datum import build_datum


def test_fake_backend_records_calls():
    be = FakeBackend(MockSamplingClient(fixed_action="bet"))
    assert be.encode("ab") == [97, 98]
    assert be.sampler().sample("p", 2, 1.0)[0].text == "ACTION: bet"
    d = build_datum([1, 2], [3], [-0.1], 1.0)
    out = be.train_step([d, d], lr=1e-5)
    assert out["loss"] == 0.0
    assert be.calls == [(2, 1e-5)]
