"""Circle probe: where do known cyclic concepts form circles inside the model?

For a set of concepts with a known cyclic order (days, months, ...) plus
controls, we run templated sentences through the model and record, at the
position of the concept word, several representations per layer:

  resid        (L+1, d)        residual stream entering layer l (l = L: final output, pre-norm)
  resid_norm   (L, d)          output of input_layernorm = what the token mixer sees
  value        (F, n_kv, hd)   v_proj output per KV head (full-attention layers)
  head_out     (F, n_h, hd)    per-head attention output, gated, before o_proj
  dn_head_out  (G, n_vh, vhd)  per-head GatedDeltaNet output, before out_proj

Each (representation, layer, head) "unit" is then scored per concept:

  circ_minus_lin  Spearman(centroid distances, circular index distance)
                  minus Spearman(..., linear index distance).  > 0: the
                  ordered sequence closes into a loop; < 0: open path.
                  THE primary circle score.
  closure         d(last, first) / mean adjacent step  (~1 loop, >> 1 open).
  loo_circ_minus_lin  supervised closure test: LOO R^2 predicting (cos, sin)
                  of the angle minus LOO R^2 predicting the linear index.
                  Open paths break the circular fit at their ends, loops
                  break the linear fit at the wrap.  Large effect sizes but
                  a wide permutation null: on synthetic hidden circles it
                  has LOWER power than circ_minus_lin (which detects a
                  circle under per-word noise up to ~2x its radius).
  pca_order       cyclic alignment between the angles of word centroids in
                  their top-2 PCs and the true angles (rotation+reflection).
  loo             leave-one-word-out ridge fit of (cos, sin) of the true
                  angle, mean cos(error) on held-out words.
  h1              max H1 persistence / diameter of the word centroids.

Caveat: pca_order and loo measure *ordinal smoothness*, not closure.  Any
curved open 1-D path (e.g. a number line) scores high on both (PCA
horseshoe effect; ridge interpolation from neighbours).  The 'numbers'
control on Qwen3.6 does.  Only circ_minus_lin / closure test the wrap.

Order-based scores get a permutation p-value (shuffle word -> angle).
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    name: str
    words: list[str]          # in cyclic order
    templates: list[str]      # each contains "{w}" exactly once, never at start
    cyclic: bool = True       # False for controls (scored as if cyclic)

    @property
    def angles(self) -> np.ndarray:
        n = len(self.words)
        return 2 * np.pi * np.arange(n) / n


_DAY_T = [
    "The meeting was moved to {w} afternoon.",
    "I will see you on {w}, if that works for you.",
    "Every {w} we go to the market together.",
    "The museum is closed on {w} for cleaning.",
    "She was born on a {w} in the middle of winter.",
    "Our flight leaves early on {w} morning.",
    "He always calls his mother on {w} evening.",
    "The report is due next {w} at noon.",
    "We had pizza last {w} night with friends.",
    "The store opens late on {w}.",
    "My favourite day of the week is {w}.",
    "The game was postponed until {w}.",
    "They got married on a sunny {w} in June.",
    "Trash is collected every {w} on our street.",
    "The package should arrive by {w}.",
    "Classes start again on {w}.",
]
_MONTH_T = [
    "The festival takes place every year in {w}.",
    "She was born on the third of {w}.",
    "The weather in {w} is usually pleasant here.",
    "We are planning a trip for next {w}.",
    "The law came into force in {w} of that year.",
    "Sales usually drop in {w} and recover later.",
    "Their wedding is set for late {w}.",
    "The album was released in {w} 1998.",
    "I started this job last {w}.",
    "The report covers the period up to {w}.",
    "By the end of {w} the project should be finished.",
    "Prices rose sharply in {w}.",
    "He moved to Paris in early {w}.",
    "The conference was held in {w} in Berlin.",
    "My favourite month of the year is {w}.",
    "The school term ends in {w}.",
]
_SEASON_T = [
    "The garden looks beautiful in {w}.",
    "We usually travel during the {w} holidays.",
    "My favourite season of the year is {w}.",
    "The trees change a lot in {w}.",
    "They opened the shop last {w}.",
    "The days feel different in {w}.",
    "Prices for flights go up in {w}.",
    "The book was published in the {w} of 2010.",
    "She loves walking in the park in {w}.",
    "The farm is busiest in {w}.",
    "Next {w} we will paint the house.",
    "The birds return every {w}.",
    "He wrote the song during a cold {w}.",
    "The team trained hard all {w}.",
    "That {w} was the best of my life.",
    "The festival is held each {w}.",
]
_DIR_T = [
    "The wind was blowing from the {w}.",
    "The village lies to the {w} of the city.",
    "We drove {w} for three hours.",
    "The window faces {w}, so it gets the light.",
    "Turn the map so that {w} is at the top.",
    "The river flows {w} towards the sea.",
    "Storms usually come from the {w} here.",
    "The army marched {w} through the valley.",
    "The ship sailed {w} across the bay.",
    "Head {w} at the next junction.",
    "The new houses were built on the {w} side of town.",
    "The compass needle pointed {w}.",
    "The birds were flying {w} for the winter.",
    "The road continues {w} past the farm.",
    "There is a lake just {w} of the forest.",
    "The airport is located {w} of the capital.",
]
_HOUR_T = [
    "The train leaves at {w} o'clock sharp.",
    "Let's meet at {w} o'clock in front of the station.",
    "The shop closes at {w} o'clock every day.",
    "I woke up at {w} o'clock this morning.",
    "The clock struck {w} as we left.",
    "The meeting starts at {w} o'clock.",
    "He came home around {w} o'clock.",
    "We had lunch at {w} o'clock.",
    "The film begins at {w} o'clock tonight.",
    "The church bells rang at {w} o'clock.",
    "Dinner will be served at {w} o'clock.",
    "She finished work at {w} o'clock.",
    "The bus arrived at {w} o'clock exactly.",
    "Wake me up at {w} o'clock, please.",
    "It was already {w} o'clock when we arrived.",
    "The party ended at {w} o'clock.",
]
_COLOR_T = [
    "She painted the door {w} last weekend.",
    "He was wearing a {w} shirt.",
    "The sky turned {w} at sunset.",
    "They bought a {w} car.",
    "My favourite colour is {w}.",
    "The flowers in the garden are {w}.",
    "The walls of the room were {w}.",
    "Mix the paint until it becomes {w}.",
    "The logo is a {w} circle.",
    "Her eyes looked almost {w} in that light.",
    "The light on the panel turned {w}.",
    "The team plays in {w} jerseys.",
    "The leaves were a deep {w}.",
    "He drew the line in {w} ink.",
    "The bird had {w} feathers.",
    "The dress was bright {w}.",
]
_NUMBER_T = [
    "There were {w} apples on the table.",
    "She has {w} brothers and sisters.",
    "We waited for {w} minutes.",
    "The team scored {w} goals.",
    "He bought {w} new books.",
    "I counted {w} birds on the wire.",
    "The recipe needs {w} eggs.",
    "They have lived here for {w} years.",
    "The building has {w} floors.",
    "Only {w} people came to the meeting.",
    "The box contains {w} pencils.",
    "We visited {w} countries last year.",
    "The class has {w} new students.",
    "He ran {w} kilometres yesterday.",
    "There are {w} chairs in the room.",
    "She wrote {w} letters to him.",
]
_NOUN_T = [
    "I saw the {w} yesterday.",
    "She talked about the {w} for an hour.",
    "The {w} was very old.",
    "We need a new {w}.",
    "He looked at the {w} carefully.",
    "There is a {w} near the house.",
    "They sold the {w} last year.",
    "The picture shows a {w}.",
    "Nobody noticed the {w}.",
    "The story is about a {w}.",
    "My friend bought a {w}.",
    "The {w} was covered in dust.",
    "Can you see the {w} over there?",
    "The children drew a {w}.",
    "The {w} is on the left.",
    "Everyone was talking about the {w}.",
]

_TWELVE = ["one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve"]

CONCEPTS: list[Concept] = [
    Concept("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"], _DAY_T),
    Concept("months", ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November",
                       "December"], _MONTH_T),
    Concept("seasons", ["spring", "summer", "autumn", "winter"], _SEASON_T),
    Concept("directions", ["north", "northeast", "east", "southeast", "south",
                           "southwest", "west", "northwest"], _DIR_T),
    Concept("hours", list(_TWELVE), _HOUR_T),
    Concept("colors", ["red", "orange", "yellow", "green", "blue", "purple"],
            _COLOR_T),
    # Controls: same scoring, no true cyclic structure expected.
    Concept("numbers", list(_TWELVE), _NUMBER_T, cyclic=False),
    Concept("nouns", ["table", "river", "doctor", "apple", "window", "guitar",
                      "mountain", "letter", "engine", "garden", "bottle",
                      "horse"], _NOUN_T, cyclic=False),
]


# ---------------------------------------------------------------------------
# Building the probe dataset
# ---------------------------------------------------------------------------

@dataclass
class ProbeItem:
    concept: str
    word: str
    word_idx: int
    template_idx: int
    text: str
    char_start: int
    char_end: int


def build_items(concepts: list[Concept]) -> list[ProbeItem]:
    items = []
    for c in concepts:
        for ti, t in enumerate(c.templates):
            assert t.count("{w}") == 1, t
            start = t.index("{w}")
            assert start > 0, f"word must not be first: {t}"
            for wi, w in enumerate(c.words):
                text = t.replace("{w}", w)
                items.append(ProbeItem(c.name, w, wi, ti, text, start, start + len(w)))
    return items


def word_token_index(offsets: list[tuple[int, int]], char_start: int,
                     char_end: int) -> int:
    """Index of the last token overlapping [char_start, char_end)."""
    idx = -1
    for i, (s, e) in enumerate(offsets):
        if e > s and s < char_end and e > char_start:
            idx = i
    if idx < 0:
        raise ValueError(f"no token covers chars [{char_start}, {char_end})")
    return idx


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_text_model(hf_model):
    m = hf_model.model if hasattr(hf_model, "model") else hf_model
    if hasattr(m, "language_model"):
        m = m.language_model
    return m


@dataclass
class _Capture:
    """Holds per-batch gather positions and collects hook outputs."""
    pos: object = None           # LongTensor (B,) positions to gather
    store: dict = field(default_factory=dict)

    def take(self, key, t):
        b = t.shape[0]
        g = t[np.arange(b), self.pos]          # (B, ...) at word position
        self.store.setdefault(key, []).append(g.detach().float().cpu())


def register_hooks(text_model, cap: _Capture):
    layers = text_model.layers
    handles = []
    full, lin = [], []

    for l, layer in enumerate(layers):
        def pre(mod, args, kwargs, l=l):
            hs = args[0] if args else kwargs["hidden_states"]
            cap.take(("resid", l), hs)
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))

        def ln_out(mod, inp, out, l=l):
            cap.take(("resid_norm", l), out)
        handles.append(layer.input_layernorm.register_forward_hook(ln_out))

        if getattr(layer, "self_attn", None) is not None:
            fi = len(full)
            full.append(l)
            attn = layer.self_attn
            n_kv, hd = attn.config.num_key_value_heads, attn.head_dim

            def v_out(mod, inp, out, fi=fi, n_kv=n_kv, hd=hd):
                cap.take(("value", fi), out.view(*out.shape[:-1], n_kv, hd))
            handles.append(attn.v_proj.register_forward_hook(v_out))

            def o_in(mod, args, fi=fi, hd=hd):
                x = args[0]
                cap.take(("head_out", fi), x.view(*x.shape[:-1], -1, hd))
            handles.append(attn.o_proj.register_forward_pre_hook(o_in))

        if getattr(layer, "linear_attn", None) is not None:
            gi = len(lin)
            lin.append(l)
            la = layer.linear_attn
            vhd = la.head_v_dim

            def dn_in(mod, args, gi=gi, vhd=vhd):
                x = args[0]
                cap.take(("dn_head_out", gi), x.view(*x.shape[:-1], -1, vhd))
            handles.append(la.out_proj.register_forward_pre_hook(dn_in))

    last = len(layers) - 1

    def final(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        cap.take(("resid", last + 1), hs)
    handles.append(layers[last].register_forward_hook(final))
    return handles, full, lin


def extract(hf_model, tokenizer, items: list[ProbeItem],
            batch_size: int = 64) -> dict:
    """Run all probe sentences, return stacked arrays (see module docstring)."""
    import torch

    text_model = get_text_model(hf_model)
    device = next(text_model.parameters()).device
    cap = _Capture()
    handles, full, lin = register_hooks(text_model, cap)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    positions = []
    try:
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            enc = tokenizer([it.text for it in batch], padding=True,
                            return_offsets_mapping=True, return_tensors="pt",
                            add_special_tokens=False)
            pos = [word_token_index(off.tolist(), it.char_start, it.char_end)
                   for off, it in zip(enc["offset_mapping"], batch)]
            positions.extend(pos)
            cap.pos = torch.tensor(pos, device=device)
            with torch.no_grad():
                text_model(input_ids=enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device))
    finally:
        for h in handles:
            h.remove()

    def stack(name, n):
        per = [torch.cat(cap.store[(name, i)]) for i in range(n)]
        return torch.stack(per, dim=1).numpy()

    n_layers = len(text_model.layers)
    out = {
        "resid": stack("resid", n_layers + 1),
        "resid_norm": stack("resid_norm", n_layers),
        "full_layers": np.array(full),
        "linear_layers": np.array(lin),
        "positions": np.array(positions),
    }
    if full:
        out["value"] = stack("value", len(full))
        out["head_out"] = stack("head_out", len(full))
    if lin:
        out["dn_head_out"] = stack("dn_head_out", len(lin))
    return out


def save_probe(path, data: dict, items: list[ProbeItem], meta: dict):
    import h5py
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if v.dtype.kind == "f":
                f.create_dataset(k, data=v.astype(np.float16))
            else:
                f.create_dataset(k, data=v)
        f.create_dataset("concept", data=np.array([it.concept for it in items], dtype="S"))
        f.create_dataset("word", data=np.array([it.word for it in items], dtype="S"))
        f.create_dataset("word_idx", data=np.array([it.word_idx for it in items]))
        f.create_dataset("template_idx", data=np.array([it.template_idx for it in items]))
        f.create_dataset("text", data=np.array([it.text for it in items], dtype="S"))
        for k, v in meta.items():
            f.attrs[k] = v


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cyclic_alignment(angles: np.ndarray, true_angles: np.ndarray) -> float:
    """Mean resultant length of angle differences, best over reflection.

    1 = same cyclic order and spacing up to a global rotation (and optional
    reflection); ~1/sqrt(n) for random angles.
    """
    a = np.asarray(angles)
    t = np.asarray(true_angles)
    r1 = np.abs(np.mean(np.exp(1j * (a - t))))
    r2 = np.abs(np.mean(np.exp(1j * (a + t))))
    return float(max(r1, r2))


def template_demean(X: np.ndarray, template_idx: np.ndarray) -> np.ndarray:
    """Subtract the per-template mean so only word-to-word variation is left."""
    Xd = X.astype(np.float64, copy=True)
    for t in np.unique(template_idx):
        m = template_idx == t
        Xd[m] -= Xd[m].mean(axis=0)
    return Xd


def word_centroids(X: np.ndarray, word_idx: np.ndarray, n_words: int) -> np.ndarray:
    return np.stack([X[word_idx == w].mean(axis=0) for w in range(n_words)])


def pca_scores(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal component scores of X (n, dim), via eigh of the n x n
    Gram matrix.  Unlike LAPACK gesdd this never fails to converge on
    rank-deficient input, and is faster when dim >> n.  Components with
    (relative) zero variance are dropped, so the result may have < k columns.
    """
    Xc = X - X.mean(axis=0)
    G = Xc @ Xc.T
    w, U = np.linalg.eigh(G)
    w, U = w[::-1], U[:, ::-1]
    if w[0] <= 0:
        return np.zeros((len(X), 0))
    keep = int(min(k, (w > w[0] * 1e-10).sum()))
    return U[:, :keep] * np.sqrt(w[:keep])


def pca_angles(M: np.ndarray) -> np.ndarray:
    P = pca_scores(M, 2)
    if P.shape[1] < 2:
        return np.zeros(len(M))
    return np.arctan2(P[:, 1], P[:, 0])


def loo_coupling(Z: np.ndarray, word_idx: np.ndarray, n_words: int,
                 lam: float = 0.1) -> np.ndarray:
    """Leave-one-word-out ridge as a linear operator on targets.

    Returns C (n_words, n_words) such that for any per-word targets T
    (n_words, 2), the LOO prediction for word j is (C @ T)[j]: the ridge model
    is fit on points of all words != j (with intercept) and evaluated at
    word j's mean point.  C[j, j] = 0.  This lets permutation nulls reuse
    the same fits.
    """
    k = Z.shape[1]
    C = np.zeros((n_words, n_words))
    for j in range(n_words):
        tr = word_idx != j
        Xt = Z[tr]
        mu = Xt.mean(axis=0)
        Xc = Xt - mu
        G = Xc.T @ Xc
        alpha = lam * np.trace(G) / k + 1e-12
        x_j = Z[word_idx == j].mean(axis=0) - mu
        # prediction = mean(Y_tr) + x_j^T (G + aI)^-1 Xc^T Y_tr
        coef = np.linalg.solve(G + alpha * np.eye(k), x_j)
        p = Xc @ coef + 1.0 / tr.sum()             # weights over train points
        np.add.at(C[j], word_idx[tr], p)
    return C


def loo_score_from_coupling(C: np.ndarray, true_angles: np.ndarray) -> float:
    T = np.stack([np.cos(true_angles), np.sin(true_angles)], axis=1)
    P = C @ T
    pred = np.arctan2(P[:, 1], P[:, 0])
    return float(np.mean(np.cos(pred - true_angles)))


def _rank_corr_pairs(D: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """Spearman of pairwise distances D[i<j] with linear and circular index
    distance, where word i sits at cyclic position pos[i]."""
    from scipy.stats import rankdata
    n = len(pos)
    iu = np.triu_indices(n, 1)
    dl = np.abs(pos[:, None] - pos[None, :])[iu].astype(float)
    dc = np.minimum(dl, n - dl)
    d = D[iu]
    rd = rankdata(np.round(d / d.max(), 9) if d.max() > 0 else d)  # fp ties

    def corr(a):
        ra = rankdata(a)
        sa, sd = ra.std(), rd.std()
        if sa == 0 or sd == 0:
            return 0.0
        return float(np.mean((ra - ra.mean()) * (rd - rd.mean())) / (sa * sd))
    return corr(dl), corr(dc)


def closure_metrics(M: np.ndarray, pos: np.ndarray | None = None) -> dict:
    """Does the ordered sequence of centroids M close into a loop?

    rho_lin / rho_circ: Spearman of centroid distances with linear |i-j| /
    circular min(|i-j|, n-|i-j|) index distance.  circ_minus_lin > 0 means
    the geometry prefers a loop over an open path.
    closure: d(last, first) / mean adjacent-step distance (~1 for a loop,
    >> 1 for an open path).
    """
    from scipy.spatial.distance import pdist, squareform
    n = len(M)
    pos = np.arange(n) if pos is None else pos
    D = squareform(pdist(M))
    rho_lin, rho_circ = _rank_corr_pairs(D, pos)
    order = np.argsort(pos)
    steps = [D[order[i], order[i + 1]] for i in range(n - 1)]
    mstep = float(np.mean(steps))
    closure = float(D[order[-1], order[0]] / mstep) if mstep > 0 else np.nan
    return dict(rho_lin=rho_lin, rho_circ=rho_circ,
                circ_minus_lin=rho_circ - rho_lin, closure=closure)


def _circ_targets(true_angles):
    return np.stack([np.cos(true_angles), np.sin(true_angles)], axis=1)


def _lin_targets(true_angles):
    i = true_angles / (2 * np.pi) * len(true_angles)
    return ((i - i.mean()) / i.std())[:, None]


def loo_r2(C: np.ndarray, T: np.ndarray) -> float:
    """R^2 of leave-one-word-out predictions C @ T against targets T."""
    P = C @ T
    sst = ((T - T.mean(axis=0)) ** 2).sum()
    return float(1 - ((P - T) ** 2).sum() / sst)


def h1_ratio(M: np.ndarray) -> float:
    import warnings
    from ripser import ripser
    from scipy.spatial.distance import pdist
    diam = pdist(M).max()
    if diam <= 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # "more columns than rows"
        dg = ripser(M / diam, maxdim=1)["dgms"][1]
    if len(dg) == 0:
        return 0.0
    return float((dg[:, 1] - dg[:, 0]).max())


_NAN_SCORES = dict(pca_order=np.nan, pca_order_p=np.nan, loo=np.nan,
                   loo_p=np.nan, h1=np.nan, rho_lin=np.nan, rho_circ=np.nan,
                   circ_minus_lin=np.nan, closure=np.nan,
                   circ_minus_lin_p=np.nan, loo_r2_circ=np.nan,
                   loo_r2_lin=np.nan, loo_circ_minus_lin=np.nan,
                   loo_circ_minus_lin_p=np.nan)


def score_unit(X: np.ndarray, word_idx: np.ndarray, template_idx: np.ndarray,
               true_angles: np.ndarray, n_perm: int = 200, k: int = 32,
               rng: np.random.Generator | None = None) -> dict:
    """All scores for one (concept, unit) point cloud X (n_points, dim)."""
    rng = rng or np.random.default_rng(0)
    n = len(true_angles)
    Xd = template_demean(X, template_idx)
    # PCA reduce (label-free) for speed and conditioning
    Z = pca_scores(Xd, k)
    if Z.shape[1] < 2:
        return dict(_NAN_SCORES)
    M = word_centroids(Z, word_idx, n)

    ang = pca_angles(M)
    pca_order = cyclic_alignment(ang, true_angles)
    C = loo_coupling(Z, word_idx, n)
    loo = loo_score_from_coupling(C, true_angles)

    cm = closure_metrics(M)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(M))

    perm_pca = np.empty(n_perm)
    perm_loo = np.empty(n_perm)
    perm_cml = np.empty(n_perm)
    perm_lcl = np.empty(n_perm)
    r2c = loo_r2(C, _circ_targets(true_angles))
    r2l = loo_r2(C, _lin_targets(true_angles))
    for i in range(n_perm):
        p = rng.permutation(n)
        t = true_angles[p]
        perm_pca[i] = cyclic_alignment(ang, t)
        perm_loo[i] = loo_score_from_coupling(C, t)
        rl, rc = _rank_corr_pairs(D, p)
        perm_cml[i] = rc - rl
        perm_lcl[i] = loo_r2(C, _circ_targets(t)) - loo_r2(C, _lin_targets(t))
    return dict(
        **cm,
        loo_r2_circ=r2c, loo_r2_lin=r2l, loo_circ_minus_lin=r2c - r2l,
        loo_circ_minus_lin_p=float((1 + (perm_lcl >= r2c - r2l - 1e-12).sum()) / (n_perm + 1)),
        circ_minus_lin_p=float((1 + (perm_cml >= cm["circ_minus_lin"] - 1e-12).sum()) / (n_perm + 1)),
        pca_order=pca_order,
        pca_order_p=float((1 + (perm_pca >= pca_order - 1e-12).sum()) / (n_perm + 1)),
        loo=loo,
        loo_p=float((1 + (perm_loo >= loo - 1e-12).sum()) / (n_perm + 1)),
        h1=h1_ratio(M),
    )


REPS = ("resid", "resid_norm", "value", "head_out", "dn_head_out")

_G: dict = {}   # shared with forked workers


def load_probe(path) -> dict:
    import h5py
    d = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            d[k] = f[k][:]
        d["attrs"] = dict(f.attrs)
    for k in ("concept", "word", "text"):
        d[k] = d[k].astype(str)
    return d


def list_units(d: dict) -> list[tuple[str, int, int]]:
    """(rep, layer, head) for every unit; head = -1 for whole-residual reps."""
    units = []
    for rep in ("resid", "resid_norm"):
        units += [(rep, l, -1) for l in range(d[rep].shape[1])]
    for rep, layers in (("value", d["full_layers"]), ("head_out", d["full_layers"]),
                        ("dn_head_out", d["linear_layers"])):
        if rep in d:
            units += [(rep, int(l), h) for l in layers for h in range(d[rep].shape[2])]
    return units


def unit_array(d: dict, rep: str, layer: int, head: int) -> np.ndarray:
    """(n_points, dim) array for one unit."""
    if head < 0:
        return d[rep][:, layer]
    layers = d["full_layers"] if rep in ("value", "head_out") else d["linear_layers"]
    i = int(np.where(layers == layer)[0][0])
    return d[rep][:, i, head]


def _score_job(args):
    rep, layer, head, n_perm = args
    d = _G["d"]
    X = unit_array(d, rep, layer, head)
    rows = []
    for c in _G["concepts"]:
        m = d["concept"] == c.name
        try:
            r = score_unit(X[m].astype(np.float32), d["word_idx"][m],
                           d["template_idx"][m], c.angles, n_perm=n_perm,
                           rng=np.random.default_rng(zlib.crc32(f"{rep}/{layer}/{head}/{c.name}".encode())))
        except np.linalg.LinAlgError as e:
            logger.warning("unit %s L%d h%d %s failed: %s", rep, layer, head, c.name, e)
            r = dict(_NAN_SCORES)
        rows.append(dict(concept=c.name, cyclic=c.cyclic, n_words=len(c.words),
                         rep=rep, layer=layer, head=head, **r))
    return rows


def score_all(d: dict, concepts: list[Concept], n_perm: int = 200,
              n_jobs: int = 1, units=None):
    """Score every unit for every concept. Returns a pandas DataFrame."""
    import multiprocessing as mp
    import pandas as pd

    names = set(d["concept"])
    concepts = [c for c in concepts if c.name in names]
    units = units if units is not None else list_units(d)
    _G["d"], _G["concepts"] = d, concepts
    jobs = [(rep, l, h, n_perm) for rep, l, h in units]
    rows = []
    if n_jobs == 1:
        for j in jobs:
            rows += _score_job(j)
    else:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_score_job, jobs, chunksize=8)):
                rows += r
                if (i + 1) % 500 == 0:
                    logger.info("scored %d/%d units", i + 1, len(jobs))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", default="Qwen/Qwen3.6-27B")
    e.add_argument("--out", required=True)
    e.add_argument("--batch-size", type=int, default=64)
    s = sub.add_parser("score")
    s.add_argument("--probe", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-perm", type=int, default=200)
    s.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.cmd == "extract":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map={"": 0})
        model.eval()
        logger.info("model loaded in %.0fs", time.time() - t0)
        items = build_items(CONCEPTS)
        logger.info("%d probe sentences", len(items))
        data = extract(model, tok, items, batch_size=a.batch_size)
        save_probe(a.out, data, items, {"model": a.model,
                                        "date": time.strftime("%Y-%m-%d %H:%M:%S")})
        logger.info("saved %s (%.0fs)", a.out, time.time() - t0)
    else:
        d = load_probe(a.probe)
        df = score_all(d, CONCEPTS, n_perm=a.n_perm, n_jobs=a.n_jobs)
        df.to_csv(a.out, index=False)
        logger.info("saved %s (%d rows)", a.out, len(df))


if __name__ == "__main__":
    main()
