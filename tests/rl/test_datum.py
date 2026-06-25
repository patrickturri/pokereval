import pytest

tinker = pytest.importorskip("tinker")  # build_datum returns a tinker.Datum (rl extra)

from pokereval.rl.datum import build_datum


def test_shapes_and_alignment():
    prompt = [10, 11, 12]        # 3 prompt tokens
    completion = [20, 21]        # 2 completion tokens
    logps = [-0.5, -0.7]
    d = build_datum(prompt, completion, logps, advantage=2.0)

    L = d.model_input.length
    assert L == len(prompt) + len(completion) - 1  # full[:-1]

    tgt = d.loss_fn_inputs["target_tokens"].tolist()
    adv = d.loss_fn_inputs["advantages"].tolist()
    lp = d.loss_fn_inputs["logprobs"].tolist()
    assert len(tgt) == len(adv) == len(lp) == L
    assert "weights" not in d.loss_fn_inputs  # importance_sampling has no weights arg

    # full = [10,11,12,20,21]; target = full[1:] = [11,12,20,21]
    assert tgt == [11, 12, 20, 21]
    # completion-target positions are the last 2 (prompt positions masked via adv=0)
    assert adv == [0.0, 0.0, 2.0, 2.0]
    assert lp == pytest.approx([0.0, 0.0, -0.5, -0.7], abs=1e-6)  # float32 storage


def test_model_input_tokens_are_full_minus_last():
    d = build_datum([1, 2], [3], [-0.1], advantage=1.0)
    assert d.model_input.to_ints() == [1, 2]  # full=[1,2,3], full[:-1]=[1,2]
