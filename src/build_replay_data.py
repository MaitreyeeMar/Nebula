"""
Builds a step-by-step "replay" history comparing LRU vs the dynamic
allocator on ONE real prompt, for the dashboard's animated replay panel.

This is computed ONCE, offline, and saved to JSON - the dashboard just
plays back the saved history, so there's no risk of live computation
lag or failure during your actual demo.

Run from nebula-project/src/ with the venv activated:
    python3 build_replay_data.py
"""

import json
import pandas as pd
from memory_model import MemoryTierModel, estimate_expert_size_bytes, estimate_kv_per_token_bytes
from allocators import LRUAllocator, DynamicJointAllocator

TRACE_PATH = "../data/expert_traces.csv"
REPLAY_PROMPT_ID = 4  # one of the "def fibonacci / def binary_search / class BinaryTree" code prompts -
                       # chosen because that's where the gap between LRU and dynamic is most dramatic

df = pd.read_csv(TRACE_PATH)
sub_df = df[df["prompt_id"] == REPLAY_PROMPT_ID]

EXPERT_SIZE_BYTES = estimate_expert_size_bytes(hidden_size=2048, intermediate_size=1024)
KV_PER_TOKEN_BYTES = estimate_kv_per_token_bytes(num_layers=16, hidden_size=2048)
HBM_CAPACITY_BYTES = 800 * 1024 * 1024

mm = MemoryTierModel()

steps = (
    sub_df.groupby(["prompt_id", "token"])
    .apply(lambda g: list(zip(g["layer"], g["expert_id"])))
    .sort_index()
)


def run_with_history(allocator_cls):
    alloc = allocator_cls(mm, HBM_CAPACITY_BYTES)
    history = []
    all_touches = []
    global_step = 0
    cum_expert_fetches = 0
    cum_kv_fetches = 0
    for (prompt_id, token), expert_pairs in steps.items():
        hits_before, fetches_before = alloc.hbm_hits, alloc.cxl_fetches
        residents_before = set(alloc.hbm_residents.keys())

        for layer, expert_id in expert_pairs:
            key = ("expert", layer, expert_id)
            f_before = alloc.cxl_fetches
            alloc.touch(key, "expert", EXPERT_SIZE_BYTES, global_step, is_new=False)
            if alloc.cxl_fetches > f_before:
                cum_expert_fetches += 1
            all_touches.append((global_step, key))
        kv_key = ("kv", prompt_id, token)
        alloc.touch(kv_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=True)
        all_touches.append((global_step, kv_key))
        for prev_token in range(token):
            prev_key = ("kv", prompt_id, prev_token)
            f_before = alloc.cxl_fetches
            alloc.touch(prev_key, "kv", KV_PER_TOKEN_BYTES, global_step, is_new=False)
            if alloc.cxl_fetches > f_before:
                cum_kv_fetches += 1
            all_touches.append((global_step, prev_key))

        residents_after = set(alloc.hbm_residents.keys())
        evicted_keys = residents_before - residents_after

        eviction_raw_key = None
        eviction_detail = None
        if evicted_keys:
            evicted_key = next(iter(evicted_keys))
            eviction_raw_key = evicted_key
            reuse_score = None
            if evicted_key[0] == "expert" and hasattr(alloc, "_current_freq"):
                reuse_score = round(alloc._current_freq(evicted_key, global_step), 2)
            eviction_detail = {
                "kind": evicted_key[0],
                "key_label": (f"layer {evicted_key[1]}, expert {evicted_key[2]}"
                              if evicted_key[0] == "expert" else f"prompt {evicted_key[1]}, token {evicted_key[2]}"),
                "reuse_score": reuse_score,
            }

        n_experts_resident = sum(1 for v in alloc.hbm_residents.values() if v["kind"] == "expert")
        n_kv_resident = sum(1 for v in alloc.hbm_residents.values() if v["kind"] == "kv")
        history.append({
            "step": global_step,
            "token": token,
            "hits_this_step": alloc.hbm_hits - hits_before,
            "fetches_this_step": alloc.cxl_fetches - fetches_before,
            "cumulative_hits": alloc.hbm_hits,
            "cumulative_fetches": alloc.cxl_fetches,
            "cum_expert_fetches": cum_expert_fetches,
            "cum_kv_fetches": cum_kv_fetches,
            "experts_resident": n_experts_resident,
            "kv_resident": n_kv_resident,
            "eviction": eviction_detail,
            "_eviction_raw_key": eviction_raw_key,
        })
        global_step += 1

    key_to_future_steps = {}
    for step, key in all_touches:
        key_to_future_steps.setdefault(key, []).append(step)

    for h in history:
        raw_key = h.pop("_eviction_raw_key")
        if h["eviction"] is None or raw_key is None:
            continue
        future_steps = [s for s in key_to_future_steps.get(raw_key, []) if s > h["step"]]
        if future_steps:
            h["eviction"]["needed_again_at_step"] = future_steps[0]
            h["eviction"]["steps_until_needed"] = future_steps[0] - h["step"]
        else:
            h["eviction"]["needed_again_at_step"] = None
            h["eviction"]["steps_until_needed"] = None

    return history



data = {
    "lru": run_with_history(LRUAllocator),
    "dynamic_joint": run_with_history(DynamicJointAllocator),
    "prompt_id": REPLAY_PROMPT_ID,
}

# Find the first step where dynamic's hit rate pulls meaningfully ahead of LRU's,
# for the "Jump to divergence" control.
divergence_step = None
for lru_h, dyn_h in zip(data["lru"], data["dynamic_joint"]):
    lru_rate = lru_h["cumulative_hits"] / max(lru_h["cumulative_hits"] + lru_h["cumulative_fetches"], 1)
    dyn_rate = dyn_h["cumulative_hits"] / max(dyn_h["cumulative_hits"] + dyn_h["cumulative_fetches"], 1)
    if dyn_rate - lru_rate > 0.10:
        divergence_step = dyn_h["step"]
        break
data["divergence_step"] = divergence_step

with open("../results/replay_data.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved replay history for prompt {REPLAY_PROMPT_ID}: "
      f"{len(data['lru'])} steps each for LRU and dynamic_joint")
print(f"Final: LRU hits={data['lru'][-1]['cumulative_hits']}, "
      f"dynamic hits={data['dynamic_joint'][-1]['cumulative_hits']}")
print(f"Divergence step: {divergence_step}")
