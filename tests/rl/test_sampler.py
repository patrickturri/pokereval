from pokereval.rl.sampler import Completion, MockSamplingClient


def test_returns_n_completions():
    sc = MockSamplingClient(fixed_action="bet")
    comps = sc.sample("prompt", n=3, temperature=1.0)
    assert len(comps) == 3
    assert all(isinstance(c, Completion) for c in comps)
    assert all("ACTION: bet" in c.text for c in comps)
    assert all(len(c.tokens) == len(c.logprobs) and c.tokens for c in comps)


def test_action_for_overrides_per_prompt():
    sc = MockSamplingClient(action_for=lambda p: "fold" if "river" in p else "call")
    assert "ACTION: fold" in sc.sample("river spot", 1, 0.0)[0].text
    assert "ACTION: call" in sc.sample("turn spot", 1, 0.0)[0].text
