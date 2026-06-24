from pokereval.interface.types import State, GameVariant, Action
from pokereval.synth.types import LabeledSpot
from pokereval.rl.dataset import split_spots


def _spots(n):
    out = []
    for i in range(n):
        s = State(variant=GameVariant.LEDUC, hero_cards=["Js"], pot=1.0,
                  legal_actions=[Action.CHECK, Action.BET], info_state_key=f"k{i}",
                  meta={"os_legal": {"check": 0, "bet": 1}})
        out.append(LabeledSpot(state=s, nash_action_probs={"check": 0.5, "bet": 0.5}, tags=[]))
    return out


def test_split_ratio_and_disjoint():
    train, ev = split_spots(_spots(100), eval_frac=0.2, seed=0)
    assert len(ev) == 20 and len(train) == 80
    tk = {s.state.info_state_key for s in train}
    ek = {s.state.info_state_key for s in ev}
    assert tk.isdisjoint(ek)


def test_split_is_deterministic():
    a = split_spots(_spots(50), 0.2, seed=0)
    b = split_spots(_spots(50), 0.2, seed=0)
    assert [s.state.info_state_key for s in a[1]] == [s.state.info_state_key for s in b[1]]
