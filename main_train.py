
from tqdm import tqdm
from MemoryStore import MemoryStore, _normalize
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
from data_pre import extract_single_session
from Tagger import A1TaggerLLM
import torch.nn as nn
from forget_gate import M4ForgetGate

from torch.optim import AdamW
import torch.nn.functional as F

TRAIN_NUM = 450
TRAIN_EVAL = 50
NUM_EPOCHS = 10

class M4TupleDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return (
            it["emb"],
            torch.tensor([it["age"]], dtype=torch.float32),
            torch.tensor([it["bm25"]], dtype=torch.float32),
            torch.tensor(it["label"], dtype=torch.float32)
        )




def build_m4_dataset(data, tagger, embedder):
    dataset = []

    for ex in tqdm(data, desc="Building M4 dataset"):
        dialog = ex["dialog"]
        if len(dialog) <= 5:
            continue

        memory = MemoryStore()
        memory.build(dialog, tagger)

        if len(memory.items) == 0:
            continue

        tuple_texts = [
            f"{it['attribute']} {it['value']}"
            for it in memory.items
        ]

        embs = embedder.encode(
            tuple_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        ages = [
            it["turn_id"] / max(1, len(dialog)-1)
            for it in memory.items
        ]

        q_toks = _normalize(ex["question"]).split()
        bm25_scores = memory.bm25.get_scores(q_toks)

        labels = [it["label"] for it in memory.items]

        for i in range(len(memory.items)):
            dataset.append({
                "emb": embs[i].cpu(),
                "age": ages[i],
                "bm25": bm25_scores[i],
                "label": labels[i]
            })

    return dataset





def train_m4_epoch_batched(loader, gate, opt):
    gate.train()
    total_loss = 0.0
    steps = 0

    for emb, age, bm25, y in loader:
        emb = emb.cuda(non_blocking=True)
        age = age.cuda(non_blocking=True)
        bm25 = bm25.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        _, logits = gate(emb, age, bm25)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps


if __name__ == "__main__":
    with open('/LongMemEval/data/longmemeval_s_cleaned.json', 'r') as f:
        data = json.load(f)

    single_session_dataset = []

    for ex in data:
        sample = extract_single_session(ex)
        single_session_dataset.append(sample)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    tagger = A1TaggerLLM("Qwen/Qwen2.5-3B-Instruct")

    
    m4_dataset = build_m4_dataset(single_session_dataset[:TRAIN_NUM], tagger, embedder)
    torch.save(m4_dataset, "saved_dataset/m4_dataset.pt")

    dataset = M4TupleDataset(m4_dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    gate = M4ForgetGate(emb_dim=embedder.get_sentence_embedding_dimension()).cuda()
    opt = AdamW(gate.parameters(), lr=2e-4, weight_decay=1e-2)

    for epoch in range(NUM_EPOCHS):
        loss = train_m4_epoch_batched(loader, gate, opt)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

    torch.save(gate.state_dict(), "m4_forget_gate.pt")
