from Tagger import A1TaggerLLM
import re
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any




def _normalize(text: str) -> str:
    text = text.replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class MemoryStore:
    def __init__(self):
        self.items = []
        self.bm25 = None
        self.corpus = []
        self.tuple_index = {}


    def build(self, dialog, tagger: A1TaggerLLM):
        """
        Build memory *only from A1 tuples*, NOT raw text.
        """

        for i, turn in enumerate(dialog):
            text = turn["content"]
            tuples = tagger.extract_tuples(text)

            for t in tuples:
                label = 1 if turn.get("has_answer", False) else 0

                mem_item = {
                    "turn_id": i,
                    "attribute": t["attribute"],
                    "value": t["value"],
                    "label": label
                }
                self.items.append(mem_item)

                attr_norm = _normalize(t["attribute"])
                val_norm  = _normalize(t["value"])

                if attr_norm:
                    self.tuple_index.setdefault(attr_norm, []).append(len(self.items)-1)

                if val_norm:
                    self.tuple_index.setdefault(val_norm, []).append(len(self.items)-1)
                    for tok in val_norm.split():
                        self.tuple_index.setdefault(tok, []).append(len(self.items)-1)

                tuple_text = f"{t['attribute']} {t['value']}"
                tuple_tokens = _normalize(tuple_text).split()
                self.corpus.append(tuple_tokens)

        assert len(self.items) == len(self.corpus), \
            f"items={len(self.items)} corpus={len(self.corpus)}"

        self.bm25 = BM25Okapi(self.corpus)

    def tuple_hits(self, question: str):
        """
        Lexical search over attributes/values only.
        """
        q_tokens = _normalize(question).split()
        hits = set()
        for tok in q_tokens:
            if tok in self.tuple_index:
                hits.update(self.tuple_index[tok])
        return hits


    def bm25_topk(self, question: str, k=20):
        """
        BM25 search over A1 tuple memory, not the conversation.
        """
        q_toks = _normalize(question).split()
        scores = self.bm25.get_scores(q_toks)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]
