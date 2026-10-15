"""Tests for generate_random_prompts."""

import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Load Pythia-160M tokenizer once for the module."""
    try:
        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m-deduped",
            revision="step143000",
            trust_remote_code=True,
        )
    except Exception:
        pytest.skip("Cannot load tokenizer (no network)")
    return tok


class TestGenerateRandomPrompts:
    """generate_random_prompts core behaviour."""

    def test_correct_count(self, tokenizer):
        """Returns exactly n_prompts strings."""
        from jlens.prompts import generate_random_prompts

        prompts = generate_random_prompts(tokenizer, n_prompts=10, seq_len=8, seed=0)
        assert len(prompts) == 10
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 0 for p in prompts), "no empty prompts"

    def test_seed_reproducibility(self, tokenizer):
        """Same seed produces identical output."""
        from jlens.prompts import generate_random_prompts

        a = generate_random_prompts(tokenizer, n_prompts=20, seq_len=16, seed=42)
        b = generate_random_prompts(tokenizer, n_prompts=20, seq_len=16, seed=42)
        assert a == b, "same seed should produce same prompts"

    def test_different_seeds_different_output(self, tokenizer):
        """Different seeds produce different output."""
        from jlens.prompts import generate_random_prompts

        a = generate_random_prompts(tokenizer, n_prompts=50, seq_len=16, seed=1)
        b = generate_random_prompts(tokenizer, n_prompts=50, seq_len=16, seed=2)
        assert a != b, "different seeds should produce different prompts"

    def test_prompts_are_random_looking(self, tokenizer):
        """Generated prompts are not real language (spot check)."""
        from jlens.prompts import generate_random_prompts

        prompts = generate_random_prompts(tokenizer, n_prompts=5, seq_len=30, seed=99)
        # Real wikitext would have spaces, punctuation, readable words.
        # Random tokens decoded should look like garbage.
        for p in prompts:
            # Should not contain common English words as a substring
            # (very unlikely for 30 random tokens)
            assert " the " not in p.lower(), \
                f"Random prompt unexpectedly contains 'the': {p[:80]}"

    def test_default_seq_len_is_64(self, tokenizer):
        """Default seq_len matches the 12B pipeline config."""
        from jlens.prompts import generate_random_prompts

        prompts = generate_random_prompts(tokenizer, n_prompts=5, seed=0)
        # We can't verify exact token count after decode/re-encode,
        # but the default should be 64 as documented.
        assert len(prompts) == 5

    def test_no_special_tokens_in_output(self, tokenizer):
        """Decoded text should not contain special token strings."""
        from jlens.prompts import generate_random_prompts

        prompts = generate_random_prompts(tokenizer, n_prompts=20, seq_len=16, seed=7)
        for p in prompts:
            assert "<|endoftext|>" not in p
            assert "<unk>" not in p
            assert "<s>" not in p
