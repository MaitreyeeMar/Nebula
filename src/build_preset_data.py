"""
Generates workload-mix results at three capacity presets (Tight/Typical/
Abundant), so the dashboard's "Build your workload" panel can combine
BOTH controls (workload mix slider + memory pressure buttons) with real
data for each combination, not just one fixed capacity.

Run from nebula-project/src/ with the venv activated:
    python3 build_preset_data.py
"""

import json
import pandas as pd
from memory_model import MemoryTierModel, estimate_expert_size_bytes, estimate_kv_per_token_bytes
from allocators import HBMOnlyAllocator, StaticPartitionAllocator, LRUAllocator, DynamicJointAllocator

TRACE_PATH = "../data/expert_traces.csv"
df = pd.read_csv(TRACE_PATH)

CODE_PROMPT_IDS = {4, 7, 11}
CATEGORY = {pid: ("code" if pid in CODE_PROMPT_IDS else "kv_heavy") for pid in df["prompt_id"].unique()}

EXPERT_SIZE_BYTES = estimate_expert_size_bytes(hidden_size=2048, intermediate_size=1024)
KV_PER_TOKEN_BYTES = estimate_kv_per_token_bytes(num_layers=16, hidden_size=2048)

PRESETS = {"tight": 200 * 1024 * 1024, "typical": 800 * 1024 * 1024, "abundant": 3200 * 1024 * 1024}

mm = MemoryTierModel()
allocator_classes = {
    "hbm_only": HBMOnlyAllocator,
    "static_partition": StaticPartitionAllocator,
    "lru": LRUAllocator,
    "dynamic_joint": DynamicJointAllocator,
}


def run_one(allocator_cls, sub_df, capacity_bytes):
    alloc = allocator_cls(mm, capacity_bytes)
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
for preset_name, capacity_bytes in PRESETS.items():
    for category in ["code", "kv_heavy"]:
        ids = [pid for pid, cat in CATEGORY.items() if cat == category]
        sub_df = df[df["prompt_id"].isin(ids)]
        for name, cls in allocator_classes.items():
            result = run_one(cls, sub_df, capacity_bytes)
            result["preset"] = preset_name
            result["category"] = category
            rows.append(result)

preset_df = pd.DataFrame(rows)
preset_df.to_csv("../results/preset_data.csv", index=False)
print(preset_df.pivot_table(index=["preset", "category"], columns="allocator", values="total_stall_ns"))
print("\nSaved to ../results/preset_data.csv")
