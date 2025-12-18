import json





def extract_single_session(example):
    """
    Convert one LongMemEval example into a single-session CRS sample.

    Returns:
        {
            "session_id": int,
            "dialog": [ ... turns ... ],
            "question": str,
            "answer": str
        }
    """
    answer_session_ids = example["answer_session_ids"]
    haystack_ids = example["haystack_session_ids"]
    haystack_sessions = example["haystack_sessions"]

    ans_id = answer_session_ids[0]

    idx = haystack_ids.index(ans_id)
    needle_session = haystack_sessions[idx]

    return {
        "session_id": ans_id,
        "dialog": needle_session,
        "question": example["question"],
        "answer": example["answer"]
    }


# if __name__ == "__main__":
#     with open('/LongMemEval/data/longmemeval_s_cleaned.json', 'r') as f:
#         data = json.load(f)

#     single_session_dataset = []

#     for ex in data:
#         sample = extract_single_session(ex)
#         single_session_dataset.append(sample)
