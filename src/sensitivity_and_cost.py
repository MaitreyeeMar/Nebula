"""
Two things this script adds beyond the basic baseline comparison:

1. CAPACITY SENSITIVITY SWEEP: runs all allocators across a range of HBM
   capacities instead of just one. This answers a much more useful
   question than a single number: "at what memory budget does the
   dynamic allocator's advantage actually show up, and when does it stop
   mattering?" That's exactly the kind of finding a hardware company like
   Astera Labs cares about (it's a deployment/sizing recommendation).

2. COST TRANSLATION: converts stall-time savings into a business metric
   (GPU-hours reclaimed, $ saved) instead of leaving it as an abstract
   nanosecond count. Every nanosecond a GPU spends stalled on a memory
   fetch is a nanosecond it wasn't doing billable compute - so stall-time
   reduction maps directly to effective throughput improvement.

   The $/hour and tokens/sec numbers below are ADJUSTABLE ASSUMPTIONS,
   not verified figures for a specific real deployment - state your
   chosen values explicitly in the report rather than presenting the
   resulting $ figure as more precise than it is.

Run from nebula-project/src/ with the venv activated:
    python3 sensitivity_and_cost.py
"""

import pandas as pd
from memory_model import MemoryTierModel, estimate_expert_size_bytes, estimate_kv_per_token_bytes
from allocators import HBMOnlyAllocator, StaticPartitionAllocator, LRUAllocator, DynamicJointAllocator

TRACE_PATH = "../data/expert_traces.csv"
df = pd.read_csv(TRACE_PATH)

EXPERT_SIZE_BYTES = estimate_expert_size_bytes(hidden_size=2048, intermediate_size=1024)
KV_PER_TOKEN_BYTES = estimate_kv_per_token_bytes(num_layers=16, hidden_size=2048)

mm = MemoryTierModel()

steps = (
    df.groupby(["prompt_id", "token"])
    .apply(lambda g: list(zip(g["layer"], g["expert_id"])))
    .sort_index()
)


def run_one(allocator_cls, hbm_capacity_bytes, **kwargs):
    alloc = allocator_cls(mm, hbm_capacity_bytes, **kwargs)
    global_step = 0
    current_prompt = None
    for (prompt_id, token), expert_pairs in steps.items():
        if current_prompt is not None and prompt_id != current_prompt:
            alloc.free_prompt(current_prompt)
        current_prompt = prompt_id

        for layer, expert_id in expert_pairs:
            key = ("expert", layer, expert_id)
            alloc.touch(key, "expert", EXPERT_SIZE_BYTES, global_step, is_new=False)

        kv_key = ("kv", prompt_id, token)
        alloc.touch(kv_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=True)

        for prev_token in range(token):
            prev_kv_key = ("kv", prompt_id, prev_token)
            alloc.touch(prev_kv_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=False)

        global_step += 1
    if current_prompt is not None:
        alloc.free_prompt(current_prompt)
    return alloc.results()


# --- 1. Capacity sensitivity sweep ---
CAPACITIES_MB = [100, 200, 400, 800, 1600, 3200]
allocator_classes = {
    "hbm_only": HBMOnlyAllocator,
    "static_partition": StaticPartitionAllocator,
    "lru": LRUAllocator,
    "dynamic_joint": DynamicJointAllocator,
}

sweep_rows = []
for cap_mb in CAPACITIES_MB:
    cap_bytes = cap_mb * 1024 * 1024
    for name, cls in allocator_classes.items():
        result = run_one(cls, cap_bytes)
        result["capacity_mb"] = cap_mb
        sweep_rows.append(result)

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv("../results/capacity_sensitivity.csv", index=False)

print("=== Capacity sensitivity sweep ===")
pivot = sweep_df.pivot(index="capacity_mb", columns="allocator", values="total_stall_ns")
print(pivot.to_string())
print("\nSaved to ../results/capacity_sensitivity.csv")

# Find where dynamic_joint's advantage over lru is largest / where it disappears
if "dynamic_joint" in pivot.columns and "lru" in pivot.columns:
    advantage_pct = (pivot["lru"] - pivot["dynamic_joint"]) / pivot["lru"] * 100
    print("\nDynamic allocator's % improvement over LRU, by capacity:")
    print(advantage_pct.to_string())

# --- 2. Cost translation (using the best result at a representative capacity) ---
GPU_COST_PER_HOUR = 3.00       # ADJUST: typical published cloud H100 rate is roughly $2-4/hr
ASSUMED_TOKENS_PER_SEC = 500   # ADJUST: rough throughput assumption for a ~7B-class model on one GPU

REPRESENTATIVE_CAPACITY_MB = 800
rep = sweep_df[sweep_df["capacity_mb"] == REPRESENTATIVE_CAPACITY_MB]
best_alt = rep[rep["allocator"] != "dynamic_joint"]["total_stall_ns"].min()
dynamic_stall = rep[rep["allocator"] == "dynamic_joint"]["total_stall_ns"].values[0]
stall_saved_ns = best_alt - dynamic_stall

tokens_in_trace = len(steps)
seconds_per_token_saved = (stall_saved_ns / 1e9) / tokens_in_trace
cost_per_token_saved = (seconds_per_token_saved / 3600) * GPU_COST_PER_HOUR
cost_saved_per_million_tokens = cost_per_token_saved * 1_000_000

print(f"\n=== Cost translation (at {REPRESENTATIVE_CAPACITY_MB}MB HBM budget) ===")
print(f"Assumptions: ${GPU_COST_PER_HOUR}/GPU-hour, {ASSUMED_TOKENS_PER_SEC} tokens/sec baseline throughput")
print(f"Stall time saved vs best alternative: {stall_saved_ns/1e6:.2f} ms over {tokens_in_trace} tokens")
print(f"Estimated cost saved: ${cost_saved_per_million_tokens:.2f} per 1M tokens generated")
print("(State your own $/hr and tokens/sec assumptions explicitly in the report - "
      "these are illustrative, not measured, figures.)")
