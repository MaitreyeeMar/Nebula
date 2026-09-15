# Cache Me If You Can

Nebula @ IIT Bombay submission, CXL-Based Memory Optimization for MoE Models track.
Maitreyee Markale, Khushbu Sidar, Yugratna Shaurya.

Full writeup is in `report.pdf`. This file is just for running the code.

## What this actually does

MoE models need to keep two different things in fast GPU memory: expert
weights and the KV-cache. Most CXL offloading work picks one of these and
gives it a fixed budget. We built an allocator that shares one CXL-backed
budget across both, and decides what to keep based on how likely each thing
is to be reused. KV-cache belonging to a still-running prompt is protected
outright (it will definitely be needed again). Experts are ranked by a
decayed popularity score instead.

Tested on OLMoE-1B-7B, real generation, 15 prompts. Beats plain LRU and a
static 50/50 split by ~16% at an 800MB budget. Numbers and caveats are in
the report - the short version is it works best under real memory pressure
and stops helping (even loses to LRU) once memory is abundant.

## Setup

Needs Python 3.10+ and a working `pip`.

```
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib plotly dash torch transformers
```

QEMU stuff (only needed if you want to re-verify the CXL device topology,
not needed to run the simulation/dashboard):

```
sudo apt install qemu-system-x86 cmake g++ make
```

Two things that aren't obvious and will waste your time if you don't know
them going in:
- QEMU's Type-3 CXL device needs an explicit LSA (Label Storage Area) backing
  object even if you're not using persistence. Without it you get a schema
  error, not a helpful message.
- Total guest RAM + the CXL device's backing memory has to fit under whatever
  RAM your host actually has free, or QEMU just fails to allocate.

## Running it

Order matters here, each script feeds the next one.

```
python3 collect_traces_v2.py        # run this in Colab, not locally - needs a GPU
                                     # writes expert_traces.csv

python3 check_expert_skew.py        # sanity check - confirms expert usage
                                     # is actually skewed, not uniform

python3 run_baselines.py            # the core comparison, all 4 allocators
python3 sensitivity_and_cost.py     # capacity sweep + cost-per-token estimate
python3 workload_mix_analysis.py    # code vs long-form prompt split
python3 build_replay_data.py        # step-by-step eviction history for the dashboard
python3 build_preset_data.py        # memory-pressure presets for the dashboard

pip install dash
python3 dashboard.py                # opens on localhost:8050
```

`dashboard.py` expects a folder called `assets/` sitting next to it with
`style.css` inside. If the dashboard loads with no styling (plain white
background, default buttons), that's almost always because this folder is
missing or in the wrong place, not a code bug. Check that first before
anything else.

## Repo layout

```
memory_model.py       - latency/bandwidth model, calibrated against published
                         NUMA figures and Astera Labs' Leo spec, not measured
allocators.py          - the four policies: HBM-only, static partition, LRU,
                         and ours (dynamic_joint)
run_baselines.py       - runs a trace through all four, main comparison
sensitivity_and_cost.py - HBM capacity sweep, $/token estimate
workload_mix_analysis.py - splits traces by category, reruns everything
build_replay_data.py   - records real per-step eviction decisions for replay
build_preset_data.py   - precomputes tight/typical/abundant presets
collect_traces_v2.py   - Colab script, real OLMoE generation -> CSV trace
dashboard.py           - Plotly Dash app, everything above feeds into this
```

## Things worth knowing before you touch the allocator code

- OLMoE-1B-7B routes top-8 of 64 experts per layer, not top-2. We got this
  wrong on the first pass and it changed the results meaningfully once fixed
  - if you're adapting this to a different model, check its actual routing
  config, don't assume.
- The first version of the dynamic allocator scored experts by lifetime
  usage count and it lost to plain LRU. Turned out old activity from a
  completely different prompt was still counted as "popular." Switched to a
  time-decayed score instead. If you're extending this, watch out for the
  same staleness trap.
- The 800MB figure used throughout is not a real GPU's HBM capacity, it's an
  effective residency budget scaled to make memory pressure visible at the
  size of our trace set. Don't quote it as if it were a real accelerator's
  spec.

## What's not done

Second MoE model (only tested on OLMoE), no physical CXL hardware
measurements (QEMU emulation + published-spec timing model only, see the
report's "Hardware-Level Validation" section for exactly what that does and
doesn't cover), and the capacity sweep only samples 6 points so the 1.6GB-3.2GB
crossover region is not finely resolved.
