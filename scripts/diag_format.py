"""Find a sampling setup that yields parseable ACTION lines (nonzero reward).

A: terse system prompt, model emits the whole line.
B: assistant-prefix forcing — prefill reply with 'ACTION:' so model completes only
   the action word.
"""
import time
import tinker
from pokereval.interface.types import GameVariant
from pokereval.synth.generator import build_labeled_spots
from pokereval.interface.prompts import render_state
from pokereval.rl.reward import spot_reward

MODEL = "Qwen/Qwen3.5-4B"
SYS = ("You are a poker solver. Reply with ONLY one line, exactly: "
       "ACTION: <action>  (e.g. 'ACTION: raise'). No explanation, no reasoning.")


def enc(tok, msgs, add_gen=True, thinking=False):
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=add_gen,
                                      tokenize=True, enable_thinking=thinking)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=add_gen, tokenize=True)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def approach_A(sc, tok, spots, max_tokens):
    rewards, parsed = [], 0
    for spot in spots:
        ids = enc(tok, [{"role": "system", "content": SYS},
                        {"role": "user", "content": render_state(spot.state)}])
        resp = sc.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=4,
                         sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=0.7)).result()
        for seq in resp.sequences:
            txt = tok.decode(seq.tokens_np.tolist())
            r, dec = spot_reward(spot, txt)
            rewards.append(r); parsed += dec is not None
    n = len(rewards)
    print(f"  A(sys,terse) max_tokens={max_tokens}: parse={parsed}/{n} mean_reward={sum(rewards)/n:.3f}")


def approach_B(sc, tok, spots, max_tokens):
    rewards, parsed = [], 0
    prefix = tok.encode("ACTION:", add_special_tokens=False)
    for spot in spots:
        base = enc(tok, [{"role": "user", "content": render_state(spot.state)}])
        ids = base + prefix
        resp = sc.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=4,
                         sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=0.7)).result()
        for seq in resp.sequences:
            txt = "ACTION:" + tok.decode(seq.tokens_np.tolist())
            r, dec = spot_reward(spot, txt)
            rewards.append(r); parsed += dec is not None
    n = len(rewards)
    print(f"  B(prefix-force) max_tokens={max_tokens}: parse={parsed}/{n} mean_reward={sum(rewards)/n:.3f}")


def main():
    svc = tinker.ServiceClient()
    tc = svc.create_lora_training_client(base_model=MODEL, rank=32)
    tok = tc.get_tokenizer()
    sc = tc.save_weights_and_get_sampling_client()
    spots = build_labeled_spots(GameVariant.LEDUC, iterations=200)[:6]
    t = time.time(); approach_A(sc, tok, spots, 24); print(f"    ({time.time()-t:.1f}s)")
    t = time.time(); approach_B(sc, tok, spots, 8); print(f"    ({time.time()-t:.1f}s)")
    # show a B sample
    base = enc(tok, [{"role": "user", "content": render_state(spots[0].state)}])
    ids = base + tok.encode("ACTION:", add_special_tokens=False)
    resp = sc.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=tinker.SamplingParams(max_tokens=8, temperature=0.7)).result()
    print("B_SAMPLE:", repr("ACTION:" + tok.decode(resp.sequences[0].tokens_np.tolist())))
    print("DIAG_OK")


if __name__ == "__main__":
    main()
