"""
Check whether expert usage in the real trace is actually skewed (a few
hot experts) or roughly uniform (load-balanced). This tells us whether a
frequency-based allocator strategy has anything real to exploit.

Run from nebula-project/src/ with the venv activated:
    python3 check_expert_skew.py
"""

import pandas as pd

df = pd.read_csv("../data/expert_traces.csv")

print(f"Total activation records: {len(df)}")
print(f"Prompts: {df['prompt_id'].nunique()}, Layers: {df['layer'].nunique()}, "
      f"Distinct experts seen: {df['expert_id'].nunique()}")
print()

for layer in sorted(df['layer'].unique())[:5]:  # first 5 layers as a sample
    counts = df[df['layer'] == layer]['expert_id'].value_counts()
    total = counts.sum()
    top3_share = counts.head(3).sum() / total
    print(f"Layer {layer}: {len(counts)} distinct experts used, "
          f"top expert used {counts.iloc[0]} times ({counts.iloc[0]/total:.1%} of layer's picks), "
          f"top-3 experts account for {top3_share:.1%} of all picks")

print()
print("Rule of thumb: if top-3 share is well above 3/num_experts (i.e. much more")
print("than what 3 equally-popular experts would get), there's real skew to exploit.")
print("If it's close to that number, routing is close to load-balanced/uniform.")
