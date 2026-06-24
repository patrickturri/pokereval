"""Diagnose whether the model emits parseable ACTION lines (nonzero reward).

Tests token budget and Qwen 'thinking' on/off. Run with billing active.
"""
import time
import tinker
from pokereval.interface.types import GameVariant
from pokereval.synth.generator import build_labeled_spots
from pokereval.interface.prompts import render_state
from pokereval.rl.reward import spot_reward

MODEL = "Qwen/Qwen3.5-4B"


def encode(tok, text, enable_thinking):
    kw = dict(add_generation_prompt=True, tokenize=True)
    try:
        out = tok.apply_chat_template([{"role": "user", "content": text}],
                                      enable_thinking=enable_thinking, **kw)
    except TypeError:
        out = tok.apply_chat_template([{"role": "user", "content": text}], **kw)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def run(sc, tok, spots, max_tokens, enable_thinking):
    rewards, parsed = [], 0
    for spot in spots:
        ids = encode(tok, render_state(spot.state), enable_thinking)
        mi = tinker.ModelInput.from_ints(ids)
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.7)
        resp = sc.sample(prompt=mi, num_samples=2, sampling_params=params).result()
        for seq in resp.sequences:
            toks = seq.tokens_np.tolist() if seq.tokens_np is not None else list(seq._tokens_list)
            txt = tok.decode(toks)
            r, dec = spot_reward(spot, txt)
            rewards.append(r)
            if dec is not None:
                parsed += 1
    n = len(rewards)
    print(f"  max_tokens={max_tokens} thinking={enable_thinking}: "
          f"parse_rate={parsed}/{n} mean_reward={sum(rewards)/n:.3f}")


def main():
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(base_model=MODEL, rank=32)
    tok = tc.get_tokenizer()
    sc = tc.save_weights_and_get_sampling_client(name="diag")
    spots = build_labeled_spots(GameVariant.LEDUC, iterations=200)[:4]
    for mt in (256, 512):
        for think in (False, True):
            t = time.time()
            run(sc, tok, spots, mt, think)
            print(f"    ({time.time()-t:.1f}s for 4 spots x2 samples)")
    # show one sample tail at 512/thinking=False
    ids = encode(tok, render_state(spots[0].state), False)
    resp = sc.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=tinker.SamplingParams(max_tokens=512, temperature=0.7)).result()
    txt = tok.decode(resp.sequences[0].tokens_np.tolist())
    print("SAMPLE_TAIL:", repr(txt[-160:]))
    print("DIAG_OK")


if __name__ == "__main__":
    main()
