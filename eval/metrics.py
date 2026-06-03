"""Ranking metrics. `ranked` = list of item ids (best first); `relevant` = set of ids."""
import numpy as np

def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(ranked, relevant, k):
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k

def average_precision_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    score = hits = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)
