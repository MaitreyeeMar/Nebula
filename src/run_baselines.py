"""
Run all allocator policies over the collected expert-activation trace and
compare total memory-stall time, hit rates, and overflow events.

Run this from inside nebula-project/src/ with the venv activated:
    python run_baselines.py
"""

import pandas as pd
from memory_model import MemoryTierModel, estimate_expert_size_bytes, estimate_kv_per_token_bytes
from allocators import HBMOnlyAllocator, StaticPartitionAllocator, LRUAllocator, DynamicJointAllocator

# --- Load the trace collected from Colab ---
TRACE_PATH = "../data/expert_traces.csv"
df = pd.read_csv(TRACE_PATH)
print(f"Loaded {len(df)} activation records covering "
      f"{df['prompt_id'].nunique()} prompts and {df['layer'].nunique()} layers")

# --- Size parameters ---
# PLACEHOLDER VALUES. For your report, replace these with real numbers computed
# from the model's config.json (see memory_model.py docstrings for the formulas).
# You can find OLMoE-1B-7B's config at:
#   https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json
EXPERT_SIZE_BYTES = estimate_expert_size_bytes(hidden_size=2048, intermediate_size=1024)
KV_PER_TOKEN_BYTES = estimate_kv_per_token_bytes(num_layers=16, hidden_size=2048)
print(f"Using expert_size={EXPERT_SIZE_BYTES/1e6:.1f}MB, "
      f"kv_per_token={KV_PER_TOKEN_BYTES/1e3:.1f}KB (placeholder values - verify against real config)")

# --- HBM budget: deliberately made small relative to the trace so that
#     memory pressure (and therefore differences between allocators)
#     actually shows up. Tune this until HBM-only starts overflowing but
#     the other allocators mostly don't. ---
HBM_CAPACITY_BYTES = 800 * 1024 * 1024  # 800 MB, tune as needed

mm = MemoryTierModel()

allocators = [
    HBMOnlyAllocator(mm, HBM_CAPACITY_BYTES),
    StaticPartitionAllocator(mm, HBM_CAPACITY_BYTES, expert_ratio=0.5),
    LRUAllocator(mm, HBM_CAPACITY_BYTES),
    DynamicJointAllocator(mm, HBM_CAPACITY_BYTES, expert_ratio=0.5),
]

# --- Build ordered (prompt_id, token) steps and, for each, the set of
#     (layer, expert_id) needed at that step ---
steps = (
    df.groupby(["prompt_id", "token"])
    .apply(lambda g: list(zip(g["layer"], g["expert_id"])))
    .sort_index()
)

for alloc in allocators:
    global_step = 0
    current_prompt = None
    for (prompt_id, token), expert_pairs in steps.items():
        if current_prompt is not None and prompt_id != current_prompt:
            alloc.free_prompt(current_prompt)
        current_prompt = prompt_id

        # Expert weights needed this step
        for layer, expert_id in expert_pairs:
            key = ("expert", layer, expert_id)
            alloc.touch(key, "expert", EXPERT_SIZE_BYTES, global_step, is_new=False)

        # New KV chunk created this step (no read cost to create it)
        kv_key = ("kv", prompt_id, token)
        alloc.touch(kv_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=True)

        # All *previous* KV chunks of this prompt are re-read this step
        # (attention looks at the full history every step)
        for prev_token in range(token):
            prev_kv_key = ("kv", prompt_id, prev_token)
            alloc.touch(prev_kv_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=False)

        global_step += 1

    if current_prompt is not None:
        alloc.free_prompt(current_prompt)

# --- Report ---
results = pd.DataFrame([a.results() for a in allocators])
print("\n=== Results ===")
print(results.to_string(index=False))

results.to_csv("../results/baseline_results.csv", index=False)
print("\nSaved to ../results/baseline_results.csv")
