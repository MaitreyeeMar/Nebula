"""
Allocator policies for the Nebula CXL/MoE simulation.

Each allocator manages a shared HBM budget (hbm_capacity_bytes) across two
kinds of resources:
  - "expert": expert FFN weights, identified by (layer, expert_id)
  - "kv":     per-token KV-cache chunks, identified by (prompt_id, token)

touch(key, kind, size_bytes, step, is_new) is called every time a resource
is needed at simulation step `step`. It returns the CXL access cost (ns)
incurred, if any (0.0 for an HBM hit or a brand-new item being created).
"""

from collections import OrderedDict
from memory_model import MemoryTierModel, TIER_CXL

OVERFLOW_PENALTY_NS = 1_000_000  # heavy penalty representing thrash/failure


class BaseAllocator:
    name = "base"

    def __init__(self, memory_model: MemoryTierModel, hbm_capacity_bytes: int):
        self.mm = memory_model
        self.hbm_capacity_bytes = hbm_capacity_bytes
        self.hbm_used_bytes = 0
        self.hbm_residents = OrderedDict()  # key -> {"kind", "size", "last_used"}
        self.stall_ns = 0.0
        self.cxl_fetches = 0
        self.hbm_hits = 0
        self.overflow_events = 0

    # ---- shared helpers ----
    def _drop(self, key):
        item = self.hbm_residents.pop(key, None)
        if item:
            self.hbm_used_bytes -= item["size"]

    def _place(self, key, kind, size_bytes, step):
        self.hbm_residents[key] = {"kind": kind, "size": size_bytes, "last_used": step}
        self.hbm_used_bytes += size_bytes

    def free_prompt(self, prompt_id):
        """Drop all kv items belonging to a finished prompt."""
        dead = [k for k, v in self.hbm_residents.items()
                if v["kind"] == "kv" and k[0] == prompt_id]
        for k in dead:
            self._drop(k)

    # ---- to override ----
    def touch(self, key, kind, size_bytes, step, is_new):
        raise NotImplementedError

    def results(self):
        return {
            "allocator": self.name,
            "total_stall_ns": self.stall_ns,
            "cxl_fetches": self.cxl_fetches,
            "hbm_hits": self.hbm_hits,
            "overflow_events": self.overflow_events,
        }


class HBMOnlyAllocator(BaseAllocator):
    """No CXL offloading at all. Illustrates why offloading is needed."""
    name = "hbm_only"

    def touch(self, key, kind, size_bytes, step, is_new):
        if key in self.hbm_residents:
            self.hbm_residents[key]["last_used"] = step
            self.hbm_hits += 1
            return 0.0
        if self.hbm_used_bytes + size_bytes <= self.hbm_capacity_bytes:
            self._place(key, kind, size_bytes, step)
            return 0.0
        # No room, no CXL to fall back on -> simulate catastrophic overflow
        self.overflow_events += 1
        self.stall_ns += OVERFLOW_PENALTY_NS
        return OVERFLOW_PENALTY_NS


class StaticPartitionAllocator(BaseAllocator):
    """Fixed byte quota split between expert and kv resources."""
    name = "static_partition"

    def __init__(self, memory_model, hbm_capacity_bytes, expert_ratio=0.5):
        super().__init__(memory_model, hbm_capacity_bytes)
        self.quota = {
            "expert": hbm_capacity_bytes * expert_ratio,
            "kv": hbm_capacity_bytes * (1 - expert_ratio),
        }
        self.used = {"expert": 0, "kv": 0}

    def _evict_lru_of_kind(self, kind):
        oldest_key, oldest_step = None, None
        for k, v in self.hbm_residents.items():
            if v["kind"] == kind and (oldest_step is None or v["last_used"] < oldest_step):
                oldest_key, oldest_step = k, v["last_used"]
        if oldest_key is not None:
            self.used[kind] -= self.hbm_residents[oldest_key]["size"]
            self._drop(oldest_key)
            return True
        return False

    def touch(self, key, kind, size_bytes, step, is_new):
        if key in self.hbm_residents:
            self.hbm_residents[key]["last_used"] = step
            self.hbm_hits += 1
            return 0.0

        cost = 0.0
        if not is_new:
            cost = self.mm.access_cost_ns(TIER_CXL, size_bytes)
            self.cxl_fetches += 1
            self.stall_ns += cost

        while self.used[kind] + size_bytes > self.quota[kind]:
            if not self._evict_lru_of_kind(kind):
                self.overflow_events += 1
                return cost  # leave this item in CXL, quota too small

        self._place(key, kind, size_bytes, step)
        self.used[kind] += size_bytes
        return cost


class LRUAllocator(BaseAllocator):
    """Single global pool, evicts globally least-recently-used regardless of kind."""
    name = "lru"

    def _evict_lru(self):
        if not self.hbm_residents:
            return False
        oldest_key = min(self.hbm_residents, key=lambda k: self.hbm_residents[k]["last_used"])
        self._drop(oldest_key)
        return True

    def touch(self, key, kind, size_bytes, step, is_new):
        if key in self.hbm_residents:
            self.hbm_residents[key]["last_used"] = step
            self.hbm_hits += 1
            return 0.0

        cost = 0.0
        if not is_new:
            cost = self.mm.access_cost_ns(TIER_CXL, size_bytes)
            self.cxl_fetches += 1
            self.stall_ns += cost

        while self.hbm_used_bytes + size_bytes > self.hbm_capacity_bytes:
            if not self._evict_lru():
                self.overflow_events += 1
                return cost

        self._place(key, kind, size_bytes, step)
        return cost


class DynamicJointAllocator(BaseAllocator):
    """
    Predicted-reuse-aware joint allocator (v3).

    Uses a DIFFERENT reuse signal for each resource kind:
      - KV chunks belonging to a still-active prompt are protected from
        eviction (attention reads the full history every step, so
        evicting one guarantees a future miss).
      - Expert weights are scored by a TIME-DECAYED popularity score,
        not a lifetime count. This matters: a lifetime count treats an
        expert that was hot 500 steps ago (in a different prompt, a
        different context) as still "hot" now, which goes stale exactly
        the way pure LFU caching is known to in the systems literature.
        Decaying toward recent activity blends "how popular" with "how
        relevant right now" - closer to hybrid policies like ARC.
    """
    name = "dynamic_joint"

    def __init__(self, memory_model, hbm_capacity_bytes, decay_half_life=150, **kwargs):
        super().__init__(memory_model, hbm_capacity_bytes)
        self.expert_freq = {}  # key -> {"score": float, "last_step": int}
        self.active_prompts = set()
        self.decay_half_life = decay_half_life

    def free_prompt(self, prompt_id):
        super().free_prompt(prompt_id)
        self.active_prompts.discard(prompt_id)

    def _decay(self, elapsed):
        return 0.5 ** (elapsed / self.decay_half_life)

    def _bump_freq(self, key, step):
        entry = self.expert_freq.get(key)
        score = entry["score"] * self._decay(step - entry["last_step"]) if entry else 0.0
        self.expert_freq[key] = {"score": score + 1.0, "last_step": step}

    def _current_freq(self, key, step):
        entry = self.expert_freq.get(key)
        if not entry:
            return 0.0
        return entry["score"] * self._decay(step - entry["last_step"])

    def _eviction_score(self, key, item, step):
        if item["kind"] == "kv":
            return float("inf") if key[1] in self.active_prompts else -1
        return self._current_freq(key, step)

    def _evict_one(self, step):
        if not self.hbm_residents:
            return False
        key = min(self.hbm_residents, key=lambda k: self._eviction_score(k, self.hbm_residents[k], step))
        if self._eviction_score(key, self.hbm_residents[key], step) == float("inf"):
            return False
        self._drop(key)
        return True

    def touch(self, key, kind, size_bytes, step, is_new):
        if kind == "kv" and is_new:
            self.active_prompts.add(key[1])
        if kind == "expert":
            self._bump_freq(key, step)

        if key in self.hbm_residents:
            self.hbm_residents[key]["last_used"] = step
            self.hbm_hits += 1
            return 0.0

        cost = 0.0
        if not is_new:
            cost = self.mm.access_cost_ns(TIER_CXL, size_bytes)
            self.cxl_fetches += 1
            self.stall_ns += cost

        while self.hbm_used_bytes + size_bytes > self.hbm_capacity_bytes:
            if not self._evict_one(step):
                self.overflow_events += 1
                return cost

        self._place(key, kind, size_bytes, step)
        return cost
