# M4 Forget Gate

This code trains and evaluates a **Learnable Forget Gate (M4)** that selects which conversational turns to retain. Our pipeline:

1. Uses an LLM-based tagger (`A1TaggerLLM`) to convert each dialogue turn into normalized attribute/value tuples (A1 tuples).
2. Stores tuples in a lexical `MemoryStore` that supports BM25 retrieval and attribute/value lookups.
3. Learns a neural forget gate (`M4ForgetGate`) trained on the tuple embedding, tuple age, and BM25 relevance so the model can down-weight or remove stale/irrelevant memories.
4. Evaluates retrieval and forgetting on LongMemEval sessions with traditional IR metrics.

## Files

| Path | Purpose |
| --- | --- |
| `main_train.py` | Builds the tuple-level training set and trains the forget gate. |
| `main_eval.py` | Runs the A1 + M4 pipeline on held-out sessions and reports metrics. |
| `MemoryStore.py` | BM25-backed store for tuple memories and lexical lookups. |
| `Tagger.py` | Qwen2.5-based tuple extractor for individual dialogue turns. |
| `forget_gate.py` | Compact feed-forward classifier that outputs forget probabilities. |
| `data_pre.py` | Utility to convert LongMemEval records into single-session samples. |
| `retrieval_metrics.py` | hit@k, recall@k, ndcg@k helpers. |
| `ci_metrics.py` | Bootstrap confidence interval utilities. |
| `saved_dataset/` | Serialized tuple dataset cache (`m4_dataset.pt`). |
| `LongMemEval/` | Expected location of the cleaned LongMemEval JSON file. |

## Requirements and Setup Instructions

- Python >= 3.13 (see `pyproject.toml`).
- CUDA-ready GPU with >=16 GB memory (Qwen2.5-3B and MiniLM embeddings run on GPU).
- Access to the Hugging Face Hub to download `Qwen/Qwen2.5-3B-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`.
- System packages for PyTorch + CUDA (install from https://pytorch.org if needed).

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> The tagger loads the Qwen model in bfloat16 with `device_map="auto"`. Make sure your GPU supports bf16 or edit `Tagger.py`.

