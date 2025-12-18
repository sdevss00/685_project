import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict


from Tagger import A1TaggerLLM
from MemoryStore import MemoryStore
from data_pre import extract_single_session
from ci_metrics import bootstrap_ci, aggregate_with_ci
from retrieval_metrics import ndcg_at_k, hit_at_k, recall_at_k
from forget_gate import M4ForgetGate

NUM_EVAL = 50


@torch.no_grad()
def forgetting_metrics_for_example(memory_items, f_probs, tau=0.7):
    y = np.array([it["label"] for it in memory_items], dtype=int)
    f = np.array(f_probs, dtype=float)

    forgotten = (f > tau)
    low_util  = (y == 0)
    high_util = (y == 1)

    F = forgotten.sum()
    L = low_util.sum()

    fp = ((forgotten & low_util).sum() / F) if F > 0 else 1.0
    fr = ((forgotten & low_util).sum() / L) if L > 0 else 1.0
    hfr = ((forgotten & high_util).sum() / F) if F > 0 else 0.0

    gold_idx = np.where(y == 1)[0]
    ru = float(np.mean(1.0 - f[gold_idx])) if len(gold_idx) else 0.0

    return {"FP": fp, "FR": fr, "HFR": hfr, "RU": ru}

def compute_forgetting_metrics(memory, forget_mask):
    labels = [it["label"] for it in memory.items]

    F = [i for i, f in enumerate(forget_mask) if f]
    R = [i for i, f in enumerate(forget_mask) if not f]

    L0 = {i for i, l in enumerate(labels) if l == 0}
    L1 = {i for i, l in enumerate(labels) if l == 1}

    F_and_L0 = [i for i in F if i in L0]
    F_and_L1 = [i for i in F if i in L1]
    R_and_L1 = [i for i in R if i in L1]

    FP = len(F_and_L0) / max(1, len(F))
    FR = len(F_and_L0) / max(1, len(L0))
    HFR = len(F_and_L1) / max(1, len(F))
    RU = len(R_and_L1) / max(1, len(R))

    return {
        "FP": FP,
        "FR": FR,
        "HFR": HFR,
        "RU": RU
    }


def retrieve_bm25_with_m4(
    memory,
    question,
    gate,
    embedder,
    sample,
    k_cand=10,
    tau=0.5
):
    q_tokens = question.split()
    bm25_scores = torch.tensor(
        memory.bm25.get_scores(q_tokens),
        dtype=torch.float32
    )

    tuple_texts = [
        f"{it['attribute']} {it['value']}"
        for it in memory.items
    ]

    emb = embedder.encode(
        tuple_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    num_turns = len(sample["dialog"])
    age = torch.tensor(
        [it["turn_id"] / max(1, num_turns - 1) for it in memory.items],
        dtype=torch.float32
    ).unsqueeze(1)

    bm25_feat = bm25_scores.unsqueeze(1)

    with torch.no_grad():
        f, _ = gate(emb.cuda(), age.cuda(), bm25_feat.cuda())

    forget_mask = (f >= tau).cpu().numpy()  # True = forgotten

    score = bm25_scores.cuda() * (1.0 - f)
    topk = torch.topk(score, k=k_cand).indices.tolist()

    return topk, forget_mask



def run_A1_M4_pipeline(sample, tagger, gate, embedder, ks=[1,3,5]):
    memory = MemoryStore()
    memory.build(sample["dialog"], tagger)

    cand_ids, forget_mask = retrieve_bm25_with_m4(
        memory,
        sample["question"],
        gate=gate,
        embedder=embedder,
        sample=sample,
        k_cand=10
    )

    gold_tuples = {
        idx for idx, item in enumerate(memory.items)
        if item["label"] == 1
    }

    metrics = {}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(cand_ids, gold_tuples, k)
        metrics[f"recall@{k}"] = recall_at_k(cand_ids, gold_tuples, k)
        metrics[f"ndcg@{k}"]   = ndcg_at_k(cand_ids, gold_tuples, k)

    forget_metrics = compute_forgetting_metrics(memory, forget_mask)

    return {**metrics, **forget_metrics}



if __name__ == "__main__":
    with open('/LongMemEval/data/longmemeval_s_cleaned.json', 'r') as f:
        data = json.load(f)

    single_session_dataset = []

    for ex in data:
        sample = extract_single_session(ex)
        single_session_dataset.append(sample)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    tagger = A1TaggerLLM("Qwen/Qwen2.5-3B-Instruct")
    gate = M4ForgetGate(emb_dim=embedder.get_sentence_embedding_dimension()).cuda()
    gate.load_state_dict(torch.load("saved_gate/m4_forget_gate.pt"))
    gate.eval()


    results_final = []
    test_data = single_session_dataset[len(single_session_dataset)-NUM_EVAL:]
    for ex in tqdm(test_data, desc="Test sample processing"):
        if len(ex["dialog"]) > 5:
            out = run_A1_M4_pipeline(ex, tagger, gate, embedder)
            results_final.append(out)



    def aggregate_results(results):
        agg = defaultdict(list)

        for res in results:
            for k, v in res.items():
                agg[k].append(v)

        mean_metrics = {
            k: float(np.mean(v)) for k, v in agg.items()
        }
        return mean_metrics

    final_metrics = aggregate_results(results_final)
    final_stats = aggregate_with_ci(final_metrics)

    print(final_stats)
