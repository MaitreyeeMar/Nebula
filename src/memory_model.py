"""
Memory tier timing model for the Nebula CXL/MoE project.

Calibration sources (document these in your report - this is exactly the
kind of detail that makes your numbers credible to judges):

- HBM / local GPU memory latency: ~100ns. In line with published NUMA
  local-node DRAM access latency figures of roughly 80-100ns.
- CXL-attached memory latency: ~170ns. Astera Labs describes CXL memory
  as behaving like a "NUMA hop" for latency purposes. Published remote-
  node NUMA access latencies range roughly 125-300ns depending on the
  platform (one commonly cited Skylake/DDR4 benchmark: 80ns local vs
  138ns remote, a 72% increase). 170ns sits near the middle of that
  published range.
- CXL link bandwidth: ~64 GB/s, consistent with a PCIe Gen5 x16 link,
  the interconnect width Leo-class CXL controllers support.

These are calibration PARAMETERS, not verified silicon numbers for any
specific chip. If you can run Intel MLC or a similar tool on real
hardware, replace these with your own measurements for extra credibility.
"""

TIER_HBM = "hbm"
TIER_CXL = "cxl"

DEFAULT_CONFIG = {
    TIER_HBM: {
        "latency_ns": 100,
        "bandwidth_gbs": 2000,   # order-of-magnitude HBM figure; adjust as needed
        "capacity_gb": 80,       # e.g. a single H100/A100-class GPU
    },
    TIER_CXL: {
        "latency_ns": 170,
        "bandwidth_gbs": 64,
        "capacity_gb": 512,      # generous CXL expansion capacity
    },
}


class MemoryTierModel:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def access_cost_ns(self, tier, size_bytes):
        """Estimated time (ns) to move size_bytes worth of data from `tier`."""
        cfg = self.config[tier]
        transfer_ns = (size_bytes / (cfg["bandwidth_gbs"] * 1e9)) * 1e9
        return cfg["latency_ns"] + transfer_ns

    def capacity_bytes(self, tier):
        return self.config[tier]["capacity_gb"] * (1024 ** 3)


def estimate_expert_size_bytes(hidden_size, intermediate_size, dtype_bytes=2, matrices=3):
    """
    Rough size of one expert's FFN weights.
    SwiGLU-style experts (gate_proj, up_proj, down_proj) use 3 matrices,
    each hidden_size x intermediate_size.
    Look up hidden_size / intermediate_size in the model's config.json on
    Hugging Face (e.g. allenai/OLMoE-1B-7B-0924) for a real, defensible number.
    """
    params = matrices * hidden_size * intermediate_size
    return params * dtype_bytes


def estimate_kv_per_token_bytes(num_layers, hidden_size, dtype_bytes=2, kv_heads_ratio=1.0):
    """
    Standard KV-cache-per-token formula: 2 (K and V) * num_layers * hidden_size.
    kv_heads_ratio < 1.0 accounts for grouped/multi-query attention if the
    model uses fewer KV heads than query heads (check the model config).
    """
    return int(2 * num_layers * hidden_size * dtype_bytes * kv_heads_ratio)
