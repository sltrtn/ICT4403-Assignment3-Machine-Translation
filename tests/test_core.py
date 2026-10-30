from src.data import clean
from src.train_pretrained import causal_prompt


def test_clean_normalizes_whitespace():
    assert clean("  one\u00a0  two\nthree ") == "one two three"


def test_causal_prompt_is_french_and_can_include_target():
    prompt = causal_prompt("Good morning")
    assert "English: Good morning" in prompt
    assert "French:" in prompt
    assert "German" not in prompt
    assert causal_prompt("Good morning", "Bonjour").endswith("Bonjour")
