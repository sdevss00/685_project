from typing import List, Dict, Any
import math

def recall_at_k(pred_ranked: List[int], gold_set: set, k: int) -> float:
    if not gold_set:
        return None
    pred_k = pred_ranked[:k]
    return len(set(pred_k) & gold_set) / len(gold_set)

def hit_at_k(pred_ranked: List[int], gold_set: set, k: int) -> float:
    if not gold_set:
        return None
    pred_k = pred_ranked[:k]
    return 1.0 if len(set(pred_k) & gold_set) > 0 else 0.0


def dcg(rels):
  return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(rels))

def ndcg_at_k(cand_ids, gold_set, k=10):
  rels = [1 if i in gold_set else 0 for i in cand_ids[:k]]
  ideal_rels = sorted(rels, reverse=True)

  if sum(ideal_rels) == 0:
      return 0.0

  return dcg(rels) / dcg(ideal_rels)