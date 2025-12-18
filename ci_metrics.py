import numpy as np
from collections import defaultdict


def bootstrap_ci(values, n_boot=1000, ci=95):
    """
    values: list of per-query metric values
    """
    values = np.array(values)
    means = []

    n = len(values)
    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def aggregate_with_ci(results, n_boot=1000):
    agg = defaultdict(list)

    for res in results:
        for k, v in res.items():
            agg[k].append(v)

    summary = {}
    for k, values in agg.items():
        mean = float(np.mean(values))
        lo, hi = bootstrap_ci(values, n_boot=n_boot)
        summary[k] = {
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi
        }

    return summary