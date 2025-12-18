import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
import re
import json
from typing import List, Dict, Any




class A1TaggerLLM:
    """
    LLM-based A1 Tagger that extracts (entity, attribute, sentiment) tuples per turn.
    """
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage = True, trust_remote_code = True
        )
        self.model.to('cuda')

    def extract_tuples(self, text: str) -> List[Dict[str, Any]]:
        prompt = f"""You are an expert dialogue annotator.

          Given ONE dialogue turn, extract all important attributes and their values for
          the speaker in that turn.

          Follow these rules STRICTLY:

          1. Identify key attributes in the turn (e.g., location, preference, job, relation,
            opinion, dislike, plan, possession, fact, etc).
          2. Include the PERSON NAME (if present) as an attribute-value pair:
                [person]<Alice>
          3. Arrange attributes in DESCENDING order of relevance (most important first).
          4. Output MUST be a LIST of items in the exact format:
                [attribute]<value>
          5. Skip attributes with missing or ambiguous values.
          6. Be concise and do NOT add explanation.
          7. Only use lowercase attribute names (e.g., location, preference, job).
          8. Output MUST be valid and contain ONLY the list, nothing else.

          Dialogue turn:
          "{text}"

          Now produce the final sorted annotation list:
          """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )
        text_out = self.tokenizer.decode(output[0], skip_special_tokens=True)

        pairs = re.findall(r"\[[^]]+\]<[^>]+>", text_out)
        results: List[Dict[str, Any]] = []

        for p in pairs:
            m = re.match(r"\[([^]]+)\]<([^>]+)>", p.strip())
            if not m:
                continue
            attr = m.group(1).strip().lower()
            val = m.group(2).strip()
            results.append({"attribute": attr, "value": val})

        return results
