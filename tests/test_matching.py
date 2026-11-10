"""Task 2: similarity aggregation, segment reductions, Chamfer and unbalanced OT."""

import pytest
import torch

from insid3_ext.matching import (aggregate_similarities, region_scores_chamfer, region_scores_ot,
                                 segment_amax, segment_logsumexp, segment_mean, unbalanced_sinkhorn_plan)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    sims = torch.rand(40, 4, generator=g) * 2 - 1
    labels = torch.randint(0, 5, (40,), generator=g)
    labels[:5] = torch.arange(5)                       # every cluster non-empty
    w = torch.tensor([0.4, 0.3, 0.2, 0.1])
    return sims, labels, w


def test_segment_ops_match_loops(data):
    sims, labels, _ = data
    for k in range(5):
        idx = labels == k
        assert torch.allclose(segment_mean(sims[:, 0], labels, 5)[k], sims[idx, 0].mean(), atol=1e-6)
        assert torch.allclose(segment_amax(sims, labels, 5)[k], sims[idx].max(0).values)
        assert torch.allclose(segment_logsumexp(sims, labels, 5)[k], torch.logsumexp(sims[idx], 0), atol=1e-5)


def test_segment_empty_cluster():
    v = torch.ones(3, 2)
    lab = torch.tensor([0, 0, 2])
    assert segment_mean(v[:, 0], lab, 3)[1] == 0
    assert torch.isinf(segment_amax(v, lab, 3)[1]).all()
    assert torch.isinf(segment_logsumexp(v, lab, 3)[1]).all()


def test_aggregation_ordering(data):
    sims, _, w = data
    wt = aggregate_similarities(sims, w, "weighted")
    ls = aggregate_similarities(sims, w, "lse", 0.1)
    mx = aggregate_similarities(sims, w, "max")
    assert torch.all(wt <= ls + 1e-6) and torch.all(ls <= mx + 1e-6)


def test_lse_limits(data):
    sims, _, w = data
    assert torch.allclose(aggregate_similarities(sims, w, "lse", 1e-4), sims.max(1).values, atol=1e-2)
    assert torch.allclose(aggregate_similarities(sims, w, "lse", 1e3), (sims * w).sum(1), atol=1e-3)


def test_single_prototype_all_rules_agree():
    s = torch.rand(10, 1)
    w = torch.ones(1)
    for mode in ("max", "weighted", "lse"):
        assert torch.allclose(aggregate_similarities(s, w, mode), s[:, 0], atol=1e-6)


def test_unknown_mode():
    with pytest.raises(ValueError):
        aggregate_similarities(torch.zeros(2, 2), torch.ones(2) / 2, "median")


def test_chamfer_by_hand():
    sims = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]])
    labels = torch.tensor([0, 0, 1])
    w = torch.tensor([0.5, 0.5])
    r = region_scores_chamfer(sims, labels, 2, w)
    assert torch.allclose(r[0], torch.tensor(0.5 * ((0.9 + 0.8) / 2 + (0.5 * 0.9 + 0.5 * 0.8))))
    assert torch.allclose(r[1], torch.tensor(0.5 * (0.5 + (0.25 + 0.2))))


def test_ot_row_marginals(data):
    sims, labels, w = data
    plan = unbalanced_sinkhorn_plan(sims, labels, 5, w, 0.1, 1.0, 100)
    cnt = torch.bincount(labels, minlength=5).float()
    assert torch.allclose(plan.sum(1), 1.0 / cnt[labels], atol=1e-5)
    assert torch.all(plan >= 0)


def test_ot_rho_to_zero_is_softmax_assignment(data):
    sims, labels, w = data
    eps = 0.1
    r = region_scores_ot(sims, labels, 5, w, eps, 1e-8, 20)
    p = torch.softmax(torch.log(w)[None] + sims / eps, dim=1)
    per_tok = (p * sims).sum(1)
    expected = segment_mean(per_tok, labels, 5)
    assert torch.allclose(r, expected, atol=1e-4)


def test_ot_large_rho_is_balanced(data):
    sims, labels, w = data
    plan = unbalanced_sinkhorn_plan(sims, labels, 5, w, 0.1, 1e6, 500)
    for k in range(5):
        col = plan[labels == k].sum(0)
        assert torch.allclose(col, w, atol=2e-3)


def test_ot_scores_bounded(data):
    sims, labels, w = data
    r = region_scores_ot(sims, labels, 5, w)
    for k in range(5):
        s = sims[labels == k]
        assert s.min() - 1e-5 <= r[k] <= s.max() + 1e-5
