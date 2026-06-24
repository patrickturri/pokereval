from pokereval.models.client import FakeClient, LLMClient


def test_fakeclient_returns_static_string():
    c = FakeClient("fixed", "ACTION: call")
    assert c.complete("anything") == "ACTION: call"
    assert c.name == "fixed"


def test_fakeclient_supports_callable():
    c = FakeClient(
        "dyn", lambda prompt: "ACTION: raise" if "Js" in prompt else "ACTION: fold"
    )
    assert c.complete("Your cards: Js") == "ACTION: raise"
    assert c.complete("Your cards: 2c") == "ACTION: fold"


def test_fakeclient_satisfies_protocol():
    assert isinstance(FakeClient("x"), LLMClient)
