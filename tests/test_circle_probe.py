"""Tests for jlens.circle_probe — hook extraction on a tiny random Qwen3.5,
word-position lookup with the real Qwen3.6 tokenizer, and the circularity
scores on synthetic point clouds.  CPU only.
"""

import numpy as np
import pytest
import torch

from jlens.circle_probe import (
    CONCEPTS,
    _Capture,
    build_items,
    closure_metrics,
    cyclic_alignment,
    extract,
    get_text_model,
    h1_ratio,
    load_probe,
    loo_coupling,
    loo_score_from_coupling,
    pca_scores,
    register_hooks,
    save_probe,
    score_all,
    score_unit,
    template_demean,
    unit_array,
    word_token_index,
)

QWEN = "Qwen/Qwen3.6-27B"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        pad_token_id=0,
    )
    m = Qwen3_5ForCausalLM(cfg).eval()
    # random (non-zero) norm weights so norm hooks are actually tested
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 1:
                p.normal_(0, 0.3)
    return m


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    except Exception:
        pytest.skip("Qwen3.6 tokenizer not cached")


def _run(model, ids, pos, mask=None):
    tm = get_text_model(model)
    cap = _Capture()
    handles, full, lin = register_hooks(tm, cap)
    cap.pos = torch.tensor(pos)
    try:
        with torch.no_grad():
            out = tm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return cap.store, full, lin, out


# ---------------------------------------------------------------------------
# Extraction hooks
# ---------------------------------------------------------------------------

class TestHooks:
    def test_layer_bookkeeping(self, tiny_model):
        ids = torch.randint(1, 300, (2, 7))
        store, full, lin, _ = _run(tiny_model, ids, [3, 6])
        assert full == [3, 7]
        assert lin == [0, 1, 2, 4, 5, 6]
        assert store[("head_out", 0)][0].shape == (2, 4, 16)
        assert store[("value", 1)][0].shape == (2, 2, 16)
        assert store[("dn_head_out", 5)][0].shape == (2, 4, 8)
        assert store[("resid", 8)][0].shape == (2, 64)

    def test_resid_matches_hidden_states(self, tiny_model):
        ids = torch.randint(1, 300, (2, 7))
        pos = [3, 6]
        store, _, _, out = _run(tiny_model, ids, pos)
        hs = out.hidden_states
        for l in range(8):
            ref = hs[l][torch.arange(2), pos]
            assert torch.allclose(store[("resid", l)][0], ref, atol=1e-5), l
        # final residual (pre-norm) -> final norm == last_hidden_state
        tm = get_text_model(tiny_model)
        normed = tm.norm(store[("resid", 8)][0])
        ref = out.last_hidden_state[torch.arange(2), pos]
        assert torch.allclose(normed, ref, atol=1e-5)

    def test_resid_norm_and_value(self, tiny_model):
        tm = get_text_model(tiny_model)
        ids = torch.randint(1, 300, (2, 7))
        store, full, _, _ = _run(tiny_model, ids, [2, 5])
        for l in range(8):
            ref = tm.layers[l].input_layernorm(store[("resid", l)][0])
            assert torch.allclose(store[("resid_norm", l)][0], ref, atol=1e-5)
        for fi, l in enumerate(full):
            v = tm.layers[l].self_attn.v_proj(store[("resid_norm", l)][0])
            assert torch.allclose(store[("value", fi)][0].reshape(2, -1), v, atol=1e-5)

    def test_head_out_reconstructs_attention_output(self, tiny_model):
        """sum_h W_O[:, h] @ head_out[h] == attention block output."""
        tm = get_text_model(tiny_model)
        ids = torch.randint(1, 300, (2, 7))
        pos = [4, 6]
        attn_out = {}
        hs = []
        for l in (3, 7):
            def hook(m, i, o, l=l):
                attn_out[l] = o[0][torch.arange(2), pos]
            hs.append(tm.layers[l].self_attn.register_forward_hook(hook))
        try:
            store, full, _, _ = _run(tiny_model, ids, pos)
        finally:
            for h in hs:
                h.remove()
        for fi, l in enumerate(full):
            ho = store[("head_out", fi)][0]           # (B, H, hd)
            W = tm.layers[l].self_attn.o_proj.weight  # (d, H*hd)
            recon = sum(ho[:, h] @ W[:, h * 16:(h + 1) * 16].T for h in range(4))
            assert torch.allclose(recon, attn_out[l], atol=1e-5)

    def test_dn_head_out_reconstructs_deltanet_output(self, tiny_model):
        tm = get_text_model(tiny_model)
        ids = torch.randint(1, 300, (2, 7))
        pos = [1, 6]
        dn_out = {}

        def hook(m, i, o):
            dn_out["x"] = o[torch.arange(2), pos]
        h = tm.layers[5].linear_attn.register_forward_hook(hook)
        try:
            store, _, lin, _ = _run(tiny_model, ids, pos)
        finally:
            h.remove()
        gi = lin.index(5)
        x = store[("dn_head_out", gi)][0].reshape(2, -1)
        recon = tm.layers[5].linear_attn.out_proj(x)
        assert torch.allclose(recon, dn_out["x"], atol=1e-5)

    def test_right_padding_does_not_change_word_position(self, tiny_model):
        """A short sentence padded in a batch gives the same reps as alone."""
        torch.manual_seed(1)
        short = torch.randint(1, 300, (1, 5))
        long = torch.randint(1, 300, (1, 9))
        batch = torch.zeros(2, 9, dtype=torch.long)
        batch[0, :5] = short[0]
        batch[1] = long[0]
        mask = torch.zeros(2, 9, dtype=torch.long)
        mask[0, :5] = 1
        mask[1] = 1
        s_b, *_ = _run(tiny_model, batch, [4, 8], mask)
        s_a, *_ = _run(tiny_model, short, [4], torch.ones(1, 5, dtype=torch.long))
        for key in [("resid", 8), ("head_out", 1), ("dn_head_out", 5), ("value", 0)]:
            assert torch.allclose(s_b[key][0][0], s_a[key][0][0], atol=1e-4), key


# ---------------------------------------------------------------------------
# Items and token positions
# ---------------------------------------------------------------------------

class TestItems:
    def test_build_items(self):
        items = build_items(CONCEPTS)
        assert len(items) == sum(len(c.words) * len(c.templates) for c in CONCEPTS)
        for it in items:
            assert it.text[it.char_start:it.char_end] == it.word
            assert it.char_start > 0

    def test_concepts_well_formed(self):
        names = [c.name for c in CONCEPTS]
        assert len(set(names)) == len(names)
        for c in CONCEPTS:
            assert len(set(c.words)) == len(c.words)
            assert len(c.templates) >= 8
            assert len(set(c.templates)) == len(c.templates)

    def test_word_token_index_synthetic(self):
        offs = [(0, 3), (3, 7), (7, 12), (12, 13), (0, 0)]
        assert word_token_index(offs, 4, 12) == 2
        assert word_token_index(offs, 3, 7) == 1
        with pytest.raises(ValueError):
            word_token_index(offs, 20, 25)

    def test_word_token_index_real_tokenizer(self, tokenizer):
        """Decoding up to the found token must end exactly with the word."""
        items = build_items(CONCEPTS)
        enc = tokenizer([it.text for it in items], return_offsets_mapping=True,
                        add_special_tokens=False)
        for it, ids, off in zip(items, enc["input_ids"], enc["offset_mapping"]):
            idx = word_token_index(off, it.char_start, it.char_end)
            assert idx > 0
            assert tokenizer.decode(ids[:idx + 1]).endswith(it.word), it.text

    def test_extract_end_to_end(self, tiny_model, tokenizer, tmp_path):
        """Small extraction with real tokenizer (ids clipped into tiny vocab)."""
        class ClipTok:
            def __init__(self, t):
                self.t = t
                self.pad_token = "x"
                self.padding_side = "right"

            def __call__(self, *a, **k):
                enc = self.t(*a, **k)
                enc["input_ids"] = enc["input_ids"] % 299 + 1
                return enc

        items = [it for it in build_items(CONCEPTS) if it.concept == "seasons"][:10]
        data = extract(tiny_model, ClipTok(tokenizer), items, batch_size=4)
        n = len(items)
        assert data["resid"].shape == (n, 9, 64)
        assert data["resid_norm"].shape == (n, 8, 64)
        assert data["head_out"].shape == (n, 2, 4, 16)
        assert data["value"].shape == (n, 2, 2, 16)
        assert data["dn_head_out"].shape == (n, 6, 4, 8)
        assert np.isfinite(data["resid"]).all()
        p = tmp_path / "probe.h5"
        save_probe(p, data, items, {"model": "tiny"})
        d = load_probe(p)
        assert list(d["concept"]) == ["seasons"] * n
        assert d["attrs"]["model"] == "tiny"
        np.testing.assert_allclose(d["resid"], data["resid"], rtol=1e-2, atol=1e-2)
        np.testing.assert_array_equal(unit_array(d, "head_out", 7, 2), d["head_out"][:, 1, 2])
        np.testing.assert_array_equal(unit_array(d, "dn_head_out", 4, 1), d["dn_head_out"][:, 3, 1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _cloud(kind, n_words=12, n_templates=16, dim=128, noise=0.1, seed=0):
    """Synthetic (points, word_idx, template_idx, true_angles)."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(n_words) / n_words
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    if kind == "circle":
        base = np.stack([np.cos(ang), np.sin(ang)], 1) @ Q[:2]
    elif kind == "line":
        base = (np.arange(n_words) / n_words)[:, None] * 2 * Q[:1]
    elif kind == "arc":
        # smooth, curved, NON-closed 1-D path (number-line-like): random
        # low-frequency cosine modes on t in [0, 1]
        tt = np.arange(n_words) / (n_words - 1)
        modes = np.stack([np.cos(np.pi * f * tt) for f in range(1, 5)], 1)
        base = modes @ (rng.normal(size=(4, dim)) / np.sqrt(dim) * 2)
    elif kind == "hidden_circle":
        # circle (radius 1) in a 2-D subspace + idiosyncratic per-word offsets
        # of norm ~1.7 in the other dims
        base = np.stack([np.cos(ang), np.sin(ang)], 1) @ Q[:2]
        base = base + rng.normal(size=(n_words, dim)) @ Q.T * np.r_[0, 0, np.full(dim - 2, 0.15)]
    elif kind == "random":
        base = rng.normal(size=(n_words, dim)) / np.sqrt(dim)
    tmpl = rng.normal(size=(n_templates, dim)) * 3   # large template offsets
    w_idx = np.repeat(np.arange(n_words), n_templates)
    t_idx = np.tile(np.arange(n_templates), n_words)
    X = base[w_idx] + tmpl[t_idx] + rng.normal(size=(len(w_idx), dim)) * noise / np.sqrt(dim) * 4
    return X.astype(np.float32), w_idx, t_idx, ang


class TestScores:
    def test_cyclic_alignment_invariances(self):
        t = 2 * np.pi * np.arange(12) / 12
        assert cyclic_alignment(t + 1.3, t) == pytest.approx(1.0)
        assert cyclic_alignment(-t + 0.4, t) == pytest.approx(1.0)
        rng = np.random.default_rng(0)
        vals = [cyclic_alignment(rng.uniform(0, 2 * np.pi, 12), t) for _ in range(200)]
        assert np.mean(vals) < 0.45

    def test_template_demean(self):
        X, w, t, _ = _cloud("circle")
        Xd = template_demean(X, t)
        for ti in np.unique(t):
            np.testing.assert_allclose(Xd[t == ti].mean(0), 0, atol=1e-5)

    def test_loo_coupling_matches_explicit_ridge(self):
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(60, 5))
        w = np.repeat(np.arange(6), 10)
        ang = 2 * np.pi * np.arange(6) / 6
        T = np.stack([np.cos(ang), np.sin(ang)], 1)
        C = loo_coupling(Z, w, 6, lam=0.1)
        assert np.allclose(np.diag(C), 0)
        for j in range(6):
            tr = w != j
            Xt, Yt = Z[tr], T[w[tr]]
            mu, ymu = Xt.mean(0), Yt.mean(0)
            G = (Xt - mu).T @ (Xt - mu)
            a = 0.1 * np.trace(G) / 5
            B = np.linalg.solve(G + a * np.eye(5), (Xt - mu).T @ (Yt - ymu))
            pred = ymu + (Z[w == j].mean(0) - mu) @ B
            np.testing.assert_allclose((C @ T)[j], pred, atol=1e-8)

    def test_circle_scores_high(self):
        X, w, t, ang = _cloud("circle")
        r = score_unit(X, w, t, ang, n_perm=200)
        assert r["pca_order"] > 0.95
        assert r["loo"] > 0.9
        assert r["pca_order_p"] < 0.01 and r["loo_p"] < 0.01
        assert r["h1"] > 0.3

    def test_line_fails_loo_and_h1(self):
        X, w, t, ang = _cloud("line")
        r = score_unit(X, w, t, ang, n_perm=200)
        assert r["loo"] < 0.5
        assert r["h1"] < 0.1

    def test_random_not_significant(self):
        ps = []
        for seed in range(5):
            X, w, t, ang = _cloud("random", seed=seed)
            r = score_unit(X, w, t, ang, n_perm=200, rng=np.random.default_rng(seed))
            ps.append(r["loo_p"])
        assert np.median(ps) > 0.1

    def test_shuffled_labels_destroy_circle(self):
        X, w, t, ang = _cloud("circle")
        perm = np.random.default_rng(1).permutation(12)
        r = score_unit(X, perm[w], t, ang, n_perm=200)
        assert r["loo_p"] > 0.01 or r["loo"] < 0.5

    def test_h1_ratio(self):
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        circle = np.stack([np.cos(a), np.sin(a)], 1)
        assert h1_ratio(circle) > 0.5
        blob = np.random.default_rng(0).normal(size=(12, 2)) * 0.01
        blob[0] += 10
        assert h1_ratio(blob) < 0.05

    def test_score_all_parallel_matches_serial(self):
        from jlens.circle_probe import Concept
        X1, w, t, _ = _cloud("circle", n_words=6, n_templates=8, dim=32)
        X2, *_ = _cloud("random", n_words=6, n_templates=8, dim=32)
        n = len(w)
        d = {
            "resid": np.stack([X1, X2], 1),
            "resid_norm": np.stack([X2], 1),
            "full_layers": np.array([1]),
            "linear_layers": np.array([0]),
            "head_out": np.stack([X1, X2], 1)[:, None],      # (n, 1 layer, 2 heads, dim)
            "concept": np.array(["c6"] * n),
            "word_idx": w,
            "template_idx": t,
        }
        cs = [Concept("c6", list("abcdef"), ["x {w}"] * 8)]
        a = score_all(d, cs, n_perm=50, n_jobs=1).sort_values(["rep", "layer", "head"])
        b = score_all(d, cs, n_perm=50, n_jobs=2).sort_values(["rep", "layer", "head"])
        assert len(a) == 2 + 1 + 2
        np.testing.assert_allclose(a["loo"].values, b["loo"].values)
        np.testing.assert_allclose(a["loo_p"].values, b["loo_p"].values)
        top = a.sort_values("loo").iloc[-1]
        assert top["head"] in (-1, 0) and top["loo"] > 0.9


class TestPCA:
    def test_pca_scores_match_svd(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 300)) * np.linspace(3, 0.1, 300)
        P = pca_scores(X, 5)
        Xc = X - X.mean(0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        ref = U[:, :5] * S[:5]
        np.testing.assert_allclose(np.abs(P), np.abs(ref), atol=1e-8)

    def test_pca_scores_degenerate(self):
        assert pca_scores(np.zeros((10, 50)), 4).shape == (10, 0)
        X = np.outer(np.arange(10.0), np.ones(50))      # rank 1
        assert pca_scores(X, 4).shape == (10, 1)

    def test_score_unit_degenerate_inputs(self):
        _, w, t, ang = _cloud("circle")
        for X in (np.zeros((len(w), 64)),
                  np.ones((len(w), 64)),
                  np.outer(t, np.ones(64)),             # only template signal
                  np.outer(w == 3, np.ones(64))):        # one word differs
            r = score_unit(X.astype(np.float32), w, t, ang, n_perm=20)
            from jlens.circle_probe import _NAN_SCORES
            assert set(r) == set(_NAN_SCORES)

    def test_score_unit_tiny_values(self):
        """fp16-scale tiny activations must still score (no underflow)."""
        X, w, t, ang = _cloud("circle")
        r = score_unit((X * 1e-4).astype(np.float32), w, t, ang, n_perm=50)
        assert r["loo"] > 0.9


class TestClosure:
    """circ_minus_lin / closure separate closed loops from open curves."""

    def test_circle_is_closed(self):
        X, w, t, ang = _cloud("circle")
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["circ_minus_lin"] > 0.3
        assert r["closure"] < 1.5
        assert r["circ_minus_lin_p"] < 0.02

    @pytest.mark.parametrize("seed", range(4))
    def test_open_arc_is_not_closed(self, seed):
        X, w, t, ang = _cloud("arc", seed=seed)
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["circ_minus_lin"] < 0
        assert r["closure"] > 2

    def test_open_arc_fools_loo(self):
        """Regression: LOO alone calls a curved open path circular (this is
        what happened with the 'numbers' control on Qwen3.6)."""
        X, w, t, ang = _cloud("arc", seed=0)
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["loo"] > 0.7

    def test_line_is_not_closed(self):
        X, w, t, ang = _cloud("line")
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["circ_minus_lin"] < -0.3
        assert r["closure"] > 5

    def test_closure_metrics_exact(self):
        a = 2 * np.pi * np.arange(8) / 8
        M = np.stack([np.cos(a), np.sin(a)], 1)
        cm = closure_metrics(M)
        assert cm["closure"] == pytest.approx(1.0)
        assert cm["rho_circ"] == pytest.approx(1.0)
        L = np.arange(8.0)[:, None]
        cm = closure_metrics(L)
        assert cm["rho_lin"] == pytest.approx(1.0)
        assert cm["closure"] == pytest.approx(7.0)
        assert cm["circ_minus_lin"] < 0

    def test_closure_label_permutation_invariance(self):
        """Metrics only depend on centroids in the given (true) order."""
        X, w, t, ang = _cloud("circle")
        perm = np.random.default_rng(2).permutation(12)
        r = score_unit(X, perm[w], t, ang, n_perm=100)
        assert r["circ_minus_lin"] < 0.3


class TestLooClosure:
    """Supervised closure: LOO R^2 of circular vs linear targets."""

    def test_circle(self):
        X, w, t, ang = _cloud("circle")
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["loo_r2_circ"] > 0.8
        assert r["loo_circ_minus_lin"] > 0.3
        assert r["loo_circ_minus_lin_p"] < 0.05

    @pytest.mark.parametrize("seed", range(4))
    def test_open_arc(self, seed):
        X, w, t, ang = _cloud("arc", seed=seed)
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["loo_circ_minus_lin"] < 0

    def test_line(self):
        X, w, t, ang = _cloud("line")
        r = score_unit(X, w, t, ang, n_perm=100)
        assert r["loo_r2_lin"] > 0.9
        assert r["loo_circ_minus_lin"] < -0.3

    def test_hidden_circle(self):
        """Circle in a 2-D subspace under per-word noise 1.7x its radius.
        Measured power (10 seeds): circ_minus_lin 1.0, loo_circ_minus_lin
        0.2 -- the unsupervised distance test is the more sensitive one;
        the supervised one keeps the right sign."""
        cml, lcl, sig = [], [], []
        for seed in range(5):
            X, w, t, ang = _cloud("hidden_circle", seed=seed)
            r = score_unit(X, w, t, ang, n_perm=100)
            cml.append(r["circ_minus_lin"])
            lcl.append(r["loo_circ_minus_lin"])
            sig.append(r["circ_minus_lin_p"] < 0.05)
        assert np.mean(sig) >= 0.8
        assert min(cml) > 0
        assert np.median(lcl) > 0.2
