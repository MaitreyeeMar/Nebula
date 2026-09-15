"""
Splits the 15 collected prompts into two categories and checks whether
the best allocator differs between them - this is the real payload for
your dashboard's "workload mix" slider, not a hypothetical claim.

Category assignment (based on the actual prompts used in collect_traces_v2.py -
edit this dict if you changed the prompt list):
    - "code": prompts 4, 7, 11 (0-indexed) - the def/class prompts.
      Hypothesis: code has more syntactic/structural switching -> more
      expert diversity per step.
    - "kv_heavy": everything else (long-form creative/factual continuations).
      Hypothesis: sustained narrative/explanatory continuations lean more
      on growing context -> KV-cache matters relatively more.

Caveat to state honestly in your report: only 3 prompts are in the "code"
category, so that result is suggestive, not statistically strong on its
own - say so rather than overclaiming.

Run from nebula-project/src/ with the venv activated:
    python3 workload_mix_analysis.py
"""

import pandas as pd
from memory_model import MemoryTierModel, estimate_expert_size_bytes, estimate_kv_per_token_bytes
from allocators import HBMOnlyAllocator, StaticPartitionAllocator, LRUAllocator, DynamicJointAllocator

TRACE_PATH = "../data/expert_traces.csv"
df = pd.read_csv(TRACE_PATH)

CODE_PROMPT_IDS = {4, 7, 11}  # def fibonacci, def binary_search, class BinaryTree
CATEGORY = {pid: ("code" if pid in CODE_PROMPT_IDS else "kv_heavy")
            for pid in df["prompt_id"].unique()}

EXPERT_SIZE_BYTES = estimate_expert_size_bytes(hidden_size=2048, intermediate_size=1024)
KV_PER_TOKEN_BYTES = estimate_kv_per_token_bytes(num_layers=16, hidden_size=2048)
HBM_CAPACITY_BYTES = 800 * 1024 * 1024

mm = MemoryTierModel()
allocator_classes = {
    "hbm_only": HBMOnlyAllocator,
    "static_partition": StaticPartitionAllocator,
    "lru": LRUAllocator,
    "dynamic_joint": DynamicJointAllocator,
}


def run_one(allocator_cls, sub_df):
    alloc = allocator_cls(mm, HBM_CAPACITY_BYTES)
    steps = (
        sub_df.groupby(["prompt_id", "token"])
        .apply(lambda g: list(zip(g["layer"], g["expert_id"])))
        .sort_index()
    )
    global_step = 0
    current_prompt = None
    for (prompt_id, token), expert_pairs in steps.items():
        if current_prompt is not None and prompt_id != current_prompt:
            alloc.free_prompt(current_prompt)
        current_prompt = prompt_id
        for layer, expert_id in expert_pairs:
            alloc.touch(("expert", layer, expert_id), "expert", EXPERT_SIZE_BYTES, global_step, is_new=False)
        alloc.touch(("kv", prompt_id, token), "kv", KV_PER_TOKEN_BYTES, global_step, is_new=True)
        for prev_token in range(token):
            alloc.touch(("kv", prompt_id, prev_token), "kv", KV_PER_TOKEN_BYTES, global_step, is_new=False)
        global_step += 1
    if current_prompt is not None:
        alloc.free_prompt(current_prompt)
    return alloc.results()


rows = []
for category in ["code", "kv_heavy"]:
    ids = [pid for pid, cat in CATEGORY.items() if cat == category]
    sub_df = df[df["prompt_id"].isin(ids)]
    for name, cls in allocator_classes.items():
        result = run_one(cls, sub_df)
        result["category"] = category
        result["n_prompts"] = len(ids)
        rows.append(result)

results_df = pd.DataFrame(rows)
results_df.to_csv("../results/workload_mix_analysis.csv", index=False)

print("=== Workload-mix analysis ===")
for category in ["code", "kv_heavy"]:
    sub = results_df[results_df["category"] == category].sort_values("total_stall_ns")
    print(f"\n--- {category} ({sub['n_prompts'].iloc[0]} prompts) - ranked best to worst ---")
    print(sub[["allocator", "total_stall_ns", "cxl_fetches", "hbm_hits"]].to_string(index=False))

print("\nSaved to ../results/workload_mix_analysis.csv")
