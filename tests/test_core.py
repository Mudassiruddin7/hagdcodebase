"""Unit tests for the method's building blocks (Eq. 1-11)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from hagd.attribution import attribute_edges, attribute_nodes, build_graph, select_candidates
from hagd.baselines.acdc import _EdgeState, _ReplicaModel
from hagd.data.tasks import SPLITS, encode_examples, get_task, score_batch
from hagd.gnn import CircuitGNN, GATLayer, make_sample, sample_loss, train_gnn
from hagd.hierarchy import build_hierarchy, normalized_laplacian
from hagd.models import transcoder_layers
from hagd.replacement import FeatureModel, nodes_to_dict
from hagd.search import FlatRanking, Traversal, select_size
from hagd.transcoder import CrossLayerTranscoder
from hagd.transfer import transfer_coefficient
from hagd.validation import necessity_prune


def _batches(lm, cfg, name="mod_arith", split="discover", n=24):
    task = get_task(name, cfg.data)
    return task, encode_examples(lm, task.make(split, n, 0), task.kind, 8, 48)


# ------------------------------------------------------------------ transcoder (Eq. 1, 3, 4)


def test_topk_sparsity_and_shapes(lm, tc):
    h = torch.randn(5, 7, lm.d_model)
    f = tc.encode(h, 0)
    assert f.shape == (5, 7, tc.m)
    assert int((f > 0).sum(-1).max()) <= tc.k
    assert (f >= 0).all()
    assert tc.decode(f, 0).shape == h.shape
    pred = tc.predict(f, 0)
    assert pred.shape == f.shape and float(pred.min()) >= 0 and float(pred.max()) <= 1  # sigmoid head (Eq. 3)


def test_decoder_initialised_as_encoder_transpose():
    t = CrossLayerTranscoder([0, 1], 16, expansion=4, k=4)
    assert torch.allclose(t.W_dec[0], t.W_enc[0].t())  # Algorithm 1, step 2
    assert torch.allclose(t.W_dec[0].norm(dim=0), torch.ones(t.m), atol=1e-5)


def test_loss_matches_eq4_and_trains():
    torch.manual_seed(0)
    t = CrossLayerTranscoder([0, 1], 16, expansion=4, k=4, pred_rank=None)
    hs = [torch.randn(64, 16), torch.randn(64, 16)]
    t.fit_normalization(hs)
    total, _ = t.loss(hs, lambda_pred=0.5, lambda_l1=0.01)
    manual = 0.0
    feats = []
    for li, h in enumerate(hs):
        x = t.standardize(h, li)
        f = t.encode(h, li)
        feats.append(f)
        manual += (x - f @ t.W_dec[li].t()).pow(2).sum(-1).mean() + 0.01 * f.abs().sum(-1).mean()
    manual += 0.5 * (feats[1] - t.predict(feats[0], 0)).pow(2).sum(-1).mean()
    assert torch.allclose(total, manual, atol=1e-5)
    opt = torch.optim.Adam(t.parameters(), lr=1e-2)
    first = float(total)
    for _ in range(40):
        loss, _ = t.loss(hs, 0.5, 0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
        t.normalize_decoder()
    assert float(loss) < first


def test_transcoder_roundtrip(tmp_path, tc):
    tc.save(tmp_path)
    again = CrossLayerTranscoder.load(tmp_path)
    h = torch.randn(4, tc.d)
    assert torch.allclose(tc.encode(h, 1), again.encode(h, 1), atol=1e-6)


def test_transcoder_layers_always_include_last():
    assert transcoder_layers(12, 1) == list(range(12))
    assert transcoder_layers(32, 4)[-1] == 31
    assert transcoder_layers(10, 4) == [3, 7, 9]


# ------------------------------------------------------------------ tasks


@pytest.mark.parametrize("name", ["mod_arith", "parity", "sorting", "ioi", "greater_than", "induction"])
def test_task_splits_are_disjoint_and_deterministic(cfg, name):
    task = get_task(name, cfg.data)
    seen = {}
    for split in SPLITS:
        exs = task.make(split, 40, 0)
        assert exs == task.make(split, 40, 0) or [e.choices for e in exs] == [e.choices for e in task.make(split, 40, 0)]
        for e in exs:
            key = tuple(e.choices)
            assert seen.setdefault(key, split) == split, "example leaked across splits"


def test_task_answers_are_correct(cfg):
    for e in get_task("mod_arith", cfg.data).make("test", 30, 0):
        assert (e.meta["a"] + e.meta["b"]) % int(cfg.data.mod_p) == int(e.choices[0][1])
    for e in get_task("parity", cfg.data).make("test", 30, 0):
        assert e.choices[0][1].strip() == ("odd" if e.meta["bits"].count("1") % 2 else "even")
    for e in get_task("sorting", cfg.data).make("test", 30, 0):
        assert e.choices[0][1].split() == [str(x) for x in sorted(e.meta["nums"])]


def test_score_batch_generative_and_multiple_choice(lm, cfg):
    task, batches = _batches(lm, cfg)
    b = batches[0]
    logits = torch.full((*b.input_ids.shape, 128), -20.0)
    logits[:, :-1].scatter_(-1, b.input_ids[:, 1:].unsqueeze(-1), 20.0)  # a model that is always right
    s = score_batch(logits, b)
    assert bool(s["correct"].all()) and float(s["loss"].max()) < 1e-3
    wrong = score_batch(-logits, b)
    assert not bool(wrong["correct"].any())
    task, batches = _batches(lm, cfg, "ioi")
    s = score_batch(lm.logits(batches[0].input_ids, batches[0].attention_mask), batches[0])
    assert s["dist"].shape == (batches[0].n_examples, 2)
    assert torch.allclose(s["dist"].exp().sum(-1), torch.ones(batches[0].n_examples), atol=1e-5)


# ------------------------------------------------------------------ interventions (Eq. 9, 10)


def test_restricting_to_all_features_is_the_identity(lm, tc, cfg):
    _, batches = _batches(lm, cfg)
    fm = FeatureModel(lm, tc)
    stats = attribute_nodes(fm, batches).stats
    everything = {li: torch.arange(tc.m) for li in range(fm.n_layers)}
    clean = fm.forward(batches[0])
    same = fm.forward(batches[0], fm.intervention(stats, everything, mode="restrict", error="clean"))
    assert torch.allclose(clean, same, atol=1e-3)
    nothing = fm.forward(batches[0], fm.intervention(stats, {}, mode="knockout"))
    assert torch.allclose(clean, nothing, atol=1e-3)
    empty = fm.forward(batches[0], fm.intervention(stats, {}, mode="restrict", error="clean"))
    assert not torch.allclose(clean, empty, atol=1e-3)


def test_bos_and_padding_are_never_modified(lm, tc, cfg):
    _, batches = _batches(lm, cfg, "sorting")
    fm = FeatureModel(lm, tc)
    stats = attribute_nodes(fm, batches).stats
    b = batches[0]
    iv = fm.intervention(stats, {}, mode="restrict", error="mean")
    h = torch.randn(*b.input_ids.shape, lm.d_model)
    out = iv.apply(h, 0, b.token_mask())
    assert torch.equal(out[:, 0], h[:, 0])
    assert torch.equal(out[~b.attention_mask.bool()], h[~b.attention_mask.bool()])


# ------------------------------------------------------------------ attribution (Eq. 5)


def test_edge_attribution_matches_finite_differences(lm, tc, cfg):
    """A_{i->j} = (df_j/df_i) f_i : compare the autograd Jacobian with a numerical derivative."""
    _, batches = _batches(lm, cfg, n=8)
    fm = FeatureModel(lm, tc)
    na = attribute_nodes(fm, batches)
    cands = select_candidates(fm, na, 6, 1000)
    w_abs, w_signed, _, _ = attribute_edges(fm, batches[:1], cands, 4, batched=False)
    b = batches[0]
    tm = fm.sites(b).unsqueeze(-1).float()  # attribution is taken at the intervention sites
    li = 0
    up, down = cands[li], cands[li + 1]
    i = int(torch.argmax(w_abs[li].sum(1)))
    eps = 1e-3

    def downstream(scale: float) -> torch.Tensor:
        def fn(h):
            f = tc.encode(h.float(), li)
            delta = torch.zeros_like(f)
            delta[..., up[i]] = f[..., up[i]] * scale * tm[..., 0]  # move feature i along its own activation, at the sites
            return h + tc.decode_delta(delta, li)

        hs = {}
        lm.run(b.input_ids, b.attention_mask, {fm.layers[li]: fn, fm.layers[li + 1]: lambda h: hs.setdefault("h", h.detach())})
        return (tc.encode(hs["h"].float(), li + 1)[..., down] * tm).view(b.n_examples, -1, len(down)).sum(1)

    numeric = (downstream(eps) - downstream(-eps)) / (2 * eps)  # d f_j / d(scale) = sum_p J * f_i
    assert torch.allclose(numeric.mean(0), w_signed[li][i], atol=5e-3, rtol=0.1)


def test_batched_and_sequential_jacobians_agree(lm, tc, cfg):
    _, batches = _batches(lm, cfg, n=8)
    fm = FeatureModel(lm, tc)
    cands = select_candidates(fm, attribute_nodes(fm, batches), 5, 1000)
    a, _, _, _ = attribute_edges(fm, batches, cands, 3, batched=True)
    s, _, _, _ = attribute_edges(fm, batches, cands, 3, batched=False)
    for x, y in zip(a, s):
        assert torch.allclose(x, y, atol=1e-5)


def test_graph_is_layered(lm, tc, cfg):
    _, batches = _batches(lm, cfg)
    g, _ = build_graph(FeatureModel(lm, tc), batches, cfg.attribution, "tiny", "mod_arith")
    assert g.n > 0 and g.n_edges > 0
    assert (g.node_layer[g.edge_dst] - g.node_layer[g.edge_src] == 1).all()  # edges join adjacent layers only
    assert (g.edge_w >= 0).all() and g.anchors.shape[1] == g.n


# ------------------------------------------------------------------ hierarchy (Eq. 6, Definition 1)


def _planted(n_blocks=4, size=12, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    src, dst, w = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            same = i // size == j // size
            if rng.random() < (0.8 if same else 0.03):
                src.append(i)
                dst.append(j)
                w.append(1.0 if same else 0.05)
    return n, np.array(src), np.array(dst), np.array(w)


def test_normalized_laplacian_formula():
    a = sp.csr_matrix(np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=float))
    d = np.diag(np.array(a.sum(1)).ravel() ** -0.5)
    assert np.allclose(normalized_laplacian(a).toarray(), np.eye(3) - d @ a.toarray() @ d)


def test_hierarchy_recovers_planted_modules_and_has_log_depth():
    n, src, dst, w = _planted()
    hier = build_hierarchy(n, src, dst, w, branching=4, embed_dim=8)
    assert hier.levels[0].n == n and hier.levels[-1].n <= 4
    assert hier.depth <= math.ceil(math.log(n, 2)) + 1
    for r, parent in enumerate(hier.parent):  # every level is a partition of the one below
        assert len(parent) == hier.levels[r].n and parent.max() + 1 == hier.levels[r + 1].n
    members = sorted(np.concatenate(hier.levels[-1].members).tolist())
    assert members == list(range(n))
    # level-1 clusters should not mix planted blocks much
    purity = np.mean([np.bincount(m // 12).max() / len(m) for m in hier.levels[1].members])
    assert purity > 0.9


# ------------------------------------------------------------------ GNN (Eq. 7, 8)


def test_attention_weights_are_a_softmax_over_neighbours():
    torch.manual_seed(0)
    layer = GATLayer(8, use_edge_weight=False)
    z = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 1, 1, 0, 2]])
    src, dst = edge_index
    e = (z @ layer.a_src)[src] + (z @ layer.a_dst)[dst]
    agg = torch.zeros_like(z)
    for v in range(5):
        incoming = dst == v
        if incoming.any():
            alpha = torch.softmax(e[incoming], 0)  # Eq. 8 by hand: softmax over the neighbours of v
            assert float(alpha.sum()) == pytest.approx(1.0, abs=1e-5)
            agg[v] = (alpha.unsqueeze(-1) * z[src[incoming]]).sum(0)
    assert torch.allclose(layer(z, edge_index, torch.zeros(6)), layer.mlp(z + agg), atol=1e-5)  # Eq. 7


def test_gnn_learns_planted_circuit(lm, tc, cfg):
    _, batches = _batches(lm, cfg)
    g, _ = build_graph(FeatureModel(lm, tc), batches, cfg.attribution, "tiny", "mod_arith")
    hier = build_hierarchy(g.n, g.edge_src, g.edge_dst, g.edge_w, 3, 8, order_hint=g.node_layer)
    circuit = np.argsort(-g.node_attr_abs)[: max(3, g.n // 6)]
    sample = make_sample(g, hier, circuit)
    assert sample.levels[0].x.shape == (g.n, 27)
    before = float(sample_loss(CircuitGNN(32, 2), sample))
    model, info = train_gnn([sample], cfg.gnn, 0)
    assert info["trained"] and float(sample_loss(model, sample)) < before
    assert info["train_auc"] > 0.9
    scores = model.score(sample)
    assert len(scores) == len(hier.levels) and all(0 <= s.min() and s.max() <= 1 for s in scores)


# ------------------------------------------------------------------ search and validation (Eq. 2, 9, 10)


def test_traversal_visits_every_leaf_once_and_prefers_high_scores():
    n, src, dst, w = _planted()
    hier = build_hierarchy(n, src, dst, w, branching=4, embed_dim=8)
    target = set(range(12))  # the first planted block is "the circuit"
    scores = [np.array([np.mean([v in target for v in m]) * 0.9 + 0.05 for m in lvl.members]) for lvl in hier.levels]
    trav = Traversal(hier, scores, gamma=0.3)
    order = trav.take(n)
    assert sorted(order) == list(range(n))
    assert set(order[:12]) == target
    assert trav.ops["decisions"] <= 3 * n  # O(n) supernode decisions (Theorem 1)


def test_select_size_finds_smallest_sufficient_prefix():
    needed = {3, 7, 11, 20}
    order = [3, 7, 99, 11, 50, 20, 1, 2, 4, 5, 6, 8, 9, 10, 12]
    ranking = FlatRanking(np.zeros(1))
    ranking.order = order

    def evaluate(nodes):
        return {"acc": len(needed & set(nodes.tolist())) / len(needed), "loss": 0.0}

    nodes, trace, reached = select_size(ranking, evaluate, lambda r: r["acc"], 0.99, 2, 15, 1.5, 15)
    assert reached and len(nodes) == 6 and needed <= set(nodes.tolist())


def test_necessity_prune_drops_only_unnecessary_features():
    needed = {0, 1, 2}

    def evaluate(nodes):
        hit = len(needed & set(nodes.tolist()))
        return {"acc": hit / 3, "loss": float(3 - hit)}

    kept, info = necessity_prune(np.arange(8), evaluate, lambda r: r["acc"], epsilon=0.5, theta=0.9, rounds=2)
    assert set(kept.tolist()) == needed and info["removed"] == 5


# ------------------------------------------------------------------ ACDC replica engine


@pytest.mark.parametrize("error", ["mean", "clean", "zero"])
def test_acdc_replicas_agree_with_node_interventions(lm, tc, cfg, error):
    """With no edge removed, the replica forward pass must equal a plain 'restrict to the pool' run."""
    _, batches = _batches(lm, cfg, n=8)
    fm = FeatureModel(lm, tc)
    na = attribute_nodes(fm, batches)
    pools = [na.attr_abs[li].topk(3).indices for li in range(fm.n_layers)]
    model = _ReplicaModel(fm, na.stats, pools, error)
    refs = fm.evaluate(batches, None, keep_dist=True)["dist"]
    state = _EdgeState([3] * fm.n_layers)
    iv = fm.intervention(na.stats, {li: p for li, p in enumerate(pools)}, mode="restrict", error=error)
    expected = fm.evaluate(batches, iv, ref=refs)["kl"]
    assert abs(model.kl(state, batches, refs) - expected) < 1e-4
    # removing every edge into the output equals dropping the last layer's pool from the circuit
    state.out_removed[:] = True
    iv = fm.intervention(na.stats, {li: p for li, p in enumerate(pools[:-1])}, mode="restrict", error=error)
    assert abs(model.kl(state, batches, refs) - fm.evaluate(batches, iv, ref=refs)["kl"]) < 1e-4
    assert not state.alive()[-1].any() and not state.alive()[0].any()


def test_removing_all_edges_of_a_layer_equals_dropping_that_layer(lm, tc, cfg):
    """Edge interventions are exact.

    When the transcoder residual is mean-ablated, a layer's hidden state is a function of its
    circuit features only.  Cutting every edge from layer 0 into layer 1 must then give exactly
    the model in which layer 0's features are not part of the circuit at all.
    """
    _, batches = _batches(lm, cfg, n=8)
    fm = FeatureModel(lm, tc)
    na = attribute_nodes(fm, batches)
    k = 24
    pools = [na.attr_abs[li].topk(k).indices for li in range(fm.n_layers)]
    refs = fm.evaluate(batches, None, keep_dist=True)["dist"]
    model = _ReplicaModel(fm, na.stats, pools, "mean")
    base = model.kl(_EdgeState([k] * fm.n_layers), batches, refs)
    cut = _EdgeState([k] * fm.n_layers)
    cut.removed[0][:, :] = True
    all_cut = model.kl(cut, batches, refs)
    iv = fm.intervention(na.stats, {li: p for li, p in enumerate(pools) if li > 0}, mode="restrict", error="mean")
    assert all_cut == pytest.approx(fm.evaluate(batches, iv, ref=refs)["kl"], abs=1e-5)
    assert abs(all_cut - base) > 1e-6, "cutting a whole layer of edges must change the output"
    # a single edge has a smaller, but real, effect
    one = _EdgeState([k] * fm.n_layers)
    one.removed[0][0, :] = True  # every input of one layer-1 feature
    single = model.kl(one, batches, refs)
    assert single != pytest.approx(all_cut, abs=1e-9)
    assert not cut.alive()[0].any() and cut.alive()[1].all()


# ------------------------------------------------------------------ transfer (Eq. 11)


def test_transfer_coefficient_bounds():
    rng = np.random.default_rng(0)
    acts = rng.random((40, 6))
    circ = {"nodes": [{"vertex": i} for i in range(6)], "edges": [[0, 2, 1.0], [1, 3, 1.0], [2, 4, 1.0], [3, 5, 1.0]]}
    same = transfer_coefficient(circ, acts, 3, circ, acts, 3)
    assert same["tau"] == pytest.approx(1.0) and same["aligned"] == 6
    other = {"nodes": circ["nodes"], "edges": [[0, 3, 1.0], [1, 2, 1.0]]}
    assert transfer_coefficient(circ, acts, 3, other, acts, 3)["tau"] == pytest.approx(0.0)
    unrelated = transfer_coefficient(circ, acts, 3, circ, rng.random((40, 6)), 3, min_corr=0.9)
    assert unrelated["tau"] == pytest.approx(0.0) and unrelated["aligned"] == 0


def test_criterion_rejects_empty_circuit_near_chance():
    """The accuracy ratio alone is ~1 for an empty circuit when the model is near chance; the combined criterion is not."""
    from hagd.validation import Criterion

    full, empty = {"acc": 0.52, "loss": 0.69}, {"acc": 0.50, "loss": 0.70}
    c = Criterion("combined", full, empty)
    assert Criterion("ratio", full, empty)(empty) > 0.9
    assert c(empty) == pytest.approx(0.0)
    assert c(full) == pytest.approx(1.0)


def test_necessity_prune_only_tests_the_tail():
    calls = []

    def evaluate(nodes):
        calls.append(len(nodes))
        return {"acc": 1.0, "loss": 0.0}

    necessity_prune(np.arange(50), evaluate, lambda r: r["acc"], epsilon=0.5, theta=0.9, rounds=1, max_tests=5)
    assert calls.count(49) == 5  # five leave-one-out runs, not fifty


def test_exhaustive_search_shrinks_but_stays_above_theta(lm, tc):
    """Pool growth + bisection + elimination + enumeration on a real (tiny) model."""
    from hagd.baselines.exhaustive import run_exhaustive
    from hagd.config import load_config
    from hagd.extract import TaskData

    cfg = load_config("smoke", {"hub": {"enabled": False}, "search": {"theta": 0.5, "criterion": "ratio"},
                                "validation": {"bootstrap": 10},
                                "baselines": {"exhaustive": {"pool": 4, "max_enum": 4, "drop_frac": 0.25, "max_minutes": 2}}})
    task = get_task("ioi", cfg.data)  # two-way choice: a random model sits near 50%, so phi is well defined
    mk = lambda split, n: encode_examples(lm, task.make(split, n, 0), task.kind, 8, 48)  # noqa: E731
    data = TaskData(task, mk("discover", 16), mk("val", 16), mk("test", 16))
    fm = FeatureModel(lm, tc)
    graph, na = build_graph(fm, data.discover, cfg.attribution, "tiny", "ioi")
    res = run_exhaustive(fm, graph, na.stats, data, cfg)
    assert res["status"] in ("ok", "timeout", "unreachable")
    if res["status"] != "unreachable":
        assert res["val_score"] >= 0.5
        assert res["metrics"]["nodes"] <= res["pool"]
    else:
        assert res["metrics"]["nodes"] == graph.n  # never shrunk to a trivial circuit


def test_exhaustive_core_recovers_planted_circuit():
    """Elimination + enumeration find exactly the needed features, starting from a pool that is too small."""
    from hagd.baselines.exhaustive import search

    needed = {5, 17, 30}
    order = np.array([5, 1, 2, 17, 3, 4, 6, 7, 8, 30, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19])

    def score(nodes):
        return len(needed & set(np.asarray(nodes).tolist())) / len(needed)

    circuit, info = search(order, score, theta=0.99, pool=4, max_enum=5, drop_frac=0.25, budget_seconds=60)
    assert info["status"] == "ok" and info["pool"] >= 10  # pool had to grow past position 9 (feature 30)
    assert info["rounds"] > 0 and info["enumerated_exactly"]
    assert set(circuit.tolist()) == needed

    _, info = search(order, lambda nodes: 0.5, theta=0.99, pool=4, max_enum=5, drop_frac=0.25, budget_seconds=60)
    assert info["status"] == "unreachable" and info["pool"] == len(order)


def test_nodes_to_dict_groups_by_layer():
    d = nodes_to_dict(np.array([0, 0, 1, 2]), np.array([5, 6, 7, 8]), np.array([0, 2, 3]))
    assert {k: v.tolist() for k, v in d.items()} == {0: [5], 1: [7], 2: [8]}
