# M4 Forget Gate

This code trains and evaluates a **Learnable Forget Gate (M4)** that selects which conversational turns to retain. Our pipeline:

1. Uses an LLM-based tagger (`A1TaggerLLM`) to convert each dialogue turn into normalized attribute/value tuples (A1 tuples).
2. Stores tuples in a lexical `MemoryStore` that supports BM25 retrieval and attribute/value lookups.
3. Learns a neural forget gate (`M4ForgetGate`) trained on the tuple embedding, tuple age, and BM25 relevance so the model can down-weight or remove stale/irrelevant memories.
4. Evaluates retrieval and forgetting on LongMemEval sessions with traditional IR metrics.
