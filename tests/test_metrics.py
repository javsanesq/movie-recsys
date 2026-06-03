import numpy as np
from eval.metrics import ndcg_at_k, recall_at_k, precision_at_k, average_precision_at_k

def test_ndcg_perfect_is_one():
    ranked = [10, 20, 30]; relevant = {10, 20, 30}
    assert abs(ndcg_at_k(ranked, relevant, 3) - 1.0) < 1e-9

def test_ndcg_reversed_less_than_one():
    ranked = [99, 98, 10]; relevant = {10}
    assert 0.0 < ndcg_at_k(ranked, relevant, 3) < 1.0

def test_recall_half():
    assert recall_at_k([1, 2, 9, 9], {1, 2, 3, 4}, 4) == 0.5

def test_precision_quarter():
    assert precision_at_k([1, 9, 9, 9], {1, 2}, 4) == 0.25

def test_average_precision():
    ap = average_precision_at_k([1, 9, 2, 9], {1, 2}, 4)
    assert abs(ap - (1.0 + 2/3) / 2) < 1e-9
