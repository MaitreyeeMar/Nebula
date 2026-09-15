"""
Nebula CXL/MoE dashboard - cockpit layout matching the approved design concept.

Setup (one-time):
    pip install dash
    python3 build_preset_data.py      (generates results/preset_data.csv)
    python3 build_replay_data.py      (generates results/replay_data.json)
    Make sure style.css is inside a folder named "assets" next to this file.

Run:
    python3 dashboard.py
Then open http://localhost:8050
"""

import json
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.graph_objects as go

# ---------- Load precomputed real data ----------
capacity_df = pd.read_csv("../results/capacity_sensitivity.csv")
preset_df = pd.read_csv("../results/preset_data.csv")
with open("../results/replay_data.json") as f:
    replay_data = json.load(f)
REPLAY_LEN = len(replay_data["lru"])
DIVERGENCE_STEP = replay_data.get("divergence_step") or 0

ALLOCATOR_ORDER = ["hbm_only", "static_partition", "lru", "dynamic_joint"]
ALLOCATOR_LABELS = {
    "hbm_only": "HBM-only", "static_partition": "Static partition",
    "lru": "LRU", "dynamic_joint": "Dynamic joint (ours)",
}
COLORS = {"hbm_only": "#B85C4A", "static_partition": "#D2853F", "lru": "#33D2C9", "dynamic_joint": "#A6E24B"}

BG, GRID, TEXT, SUBTEXT = "#10151C", "#3A3226", "#F2F0E6", "#8B94A0"
FONT_MONO = "IBM Plex Mono, monospace"

# Must match memory_model.py's estimate_expert_size_bytes(hidden=2048, intermediate=1024)
# and estimate_kv_per_token_bytes(num_layers=16, hidden=2048) - kept literal here so this
# file has no dependency on the simulation modules, just their precomputed outputs.
EXPERT_SIZE_BYTES = 3 * 2048 * 1024 * 2
KV_PER_TOKEN_BYTES = 2 * 16 * 2048 * 2
PRESET_CAPACITY_BYTES = {"tight": 200 * 1024 * 1024, "typical": 800 * 1024 * 1024, "abundant": 3200 * 1024 * 1024}

kv_heavy_by_preset = {p: preset_df[(preset_df["preset"] == p) & (preset_df["category"] == "kv_heavy")].set_index("allocator")
                      for p in PRESET_CAPACITY_BYTES}
code_by_preset = {p: preset_df[(preset_df["preset"] == p) & (preset_df["category"] == "code")].set_index("allocator")
                  for p in PRESET_CAPACITY_BYTES}


def style_fig(fig, title):
    fig.update_layout(
        title=title, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT_MONO, color=TEXT, size=12),
        title_font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT),
        margin=dict(l=45, r=20, t=45, b=35),
        legend=dict(bgcolor=BG, font=dict(color=SUBTEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


capacity_fig = go.Figure()
for alloc in ALLOCATOR_ORDER:
    sub = capacity_df[capacity_df["allocator"] == alloc].sort_values("capacity_mb")
    capacity_fig.add_trace(go.Scatter(
        x=sub["capacity_mb"], y=sub["total_stall_ns"] / 1e9, mode="lines+markers",
        name=ALLOCATOR_LABELS[alloc], line=dict(color=COLORS[alloc], width=2.5), marker=dict(size=6),
    ))
capacity_fig.update_xaxes(title="HBM capacity, MB (log)", type="log")
capacity_fig.update_yaxes(title="Stall, billion ns")
style_fig(capacity_fig, "Advantage shrinks as capacity grows")
capacity_fig.update_layout(height=260, showlegend=False)

app = dash.Dash(__name__)
app.title = "Nebula / Memory Lab"

app.index_string = """
<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""


def bar_track(seg_id_prefix):
    return html.Div(className="bar-track", children=[
        html.Div(id=f"{seg_id_prefix}-expert", className="bar-segment expert", style={"width": "0%"}),
        html.Div(id=f"{seg_id_prefix}-kv", className="bar-segment kv", style={"width": "0%"}),
    ])


app.layout = html.Div(className="wrap", id="top", children=[

    html.Div(className="topbar", children=[
        html.Div(className="brand", children=[
            html.Span(className="brand-title", children="NEBULA / MEMORY LAB"),
            html.Span(className="brand-sub", children="CXL x MoE"),
        ]),
        html.Div(className="nav", children=[
            html.A("Overview", href="#top", className="active"),
            html.A("Replay", href="#compare-section"),
            html.A("Evidence", href="#evidence-panel"),
        ]),
        html.Div(className="badge", children=[
            html.Span(className="badge-dot"), "Precomputed demo",
        ]),
    ]),

    # Row 1: hero + workload builder
    html.Div(className="row-2col", children=[
        html.Div(className="panel", children=[
            html.Div(className="hero-label", children="NEBULA"),
            html.Div(className="hero-body", children=[
                html.Div(children=[
                    html.H1(className="hero-headline", children="One pool. Two competing demands."),
                    html.Div(className="hero-sub", children="CXL-based memory optimization for MoE"),
                    html.Div(className="hero-caption", children="OLMoE-1B-7B, 15 real prompt traces"),
                ]),
                html.Div(className="hero-stat-block", children=[
                    html.Div(id="hero-stat", className="hero-stat", children="16%"),
                    html.Div(className="hero-stat-label", children="overall improvement"),
                    html.A("Metric & baseline definition in Evidence",
                           href="#evidence-panel", className="hero-link"),
                ]),
            ]),
        ]),
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children="Build your workload"),
            html.Div(className="field-label", children="Long-form vs code-heavy"),
            dcc.Slider(id="mix-slider", min=0, max=100, step=5, value=50,
                       marks={0: "Long-form", 100: "Code-heavy"}),
            html.Div(className="field-label", children="Memory pressure"),
            html.Div(className="preset-row", children=[
                html.Button("Tight", id="preset-tight", className="preset-btn selected"),
                html.Button("Typical", id="preset-typical", className="preset-btn"),
                html.Button("Abundant", id="preset-abundant", className="preset-btn"),
            ]),
            dcc.Store(id="preset-store", data="tight"),
            html.Div(id="workload-note", className="note-box", children="Code-like prompts: greater benefit"),
        ]),
    ]),

    # Row 2: comparison + inspector
    html.Div(id="compare-section", className="row-2col", children=[
        html.Div(className="panel", children=[
            html.Div(className="panel-header-row", children=[
                html.Div(className="panel-title", children="Where LRU loses the next hit"),
                html.Button("Jump to divergence", id="jump-button", className="small-btn"),
            ]),
            html.Div(className="compare-cols", children=[
                html.Div(children=[
                    html.Div(className="compare-col-title", children="Plain LRU"),
                    html.Div(className="bar-label", children="GPU memory (on device)"),
                    bar_track("lru-bar1"),
                    html.Div(className="bar-label", children="CXL memory (offloaded)"),
                    bar_track("lru-bar2"),
                    html.Div(id="lru-hitrate", className="compare-stat", children="0%"),
                    html.Div(className="compare-stat-label", children="hit rate on selected trace"),
                ]),
                html.Div(children=[
                    html.Div(className="compare-col-title", children="Joint allocator"),
                    html.Div(className="bar-label", children="GPU memory (on device)"),
                    bar_track("dyn-bar1"),
                    html.Div(className="bar-label", children="CXL memory (offloaded)"),
                    bar_track("dyn-bar2"),
                    html.Div(id="dyn-hitrate", className="compare-stat", children="0%"),
                    html.Div(className="compare-stat-label", children="hit rate on selected trace"),
                ]),
            ]),
            html.Div(className="legend-row", children=[
                html.Span(children=[html.Span(className="legend-dot", style={"background": "#D2853F"}), "Expert weights"]),
                html.Span(children=[html.Span(className="legend-dot", style={"background": "#33D2C9"}), "KV-cache"]),
            ]),
            html.Div(className="playback-row", children=[
                html.Button("\u25B6", id="play-button", n_clicks=0, className="play-btn"),
                html.Button("Step", id="step-button", className="small-btn"),
                html.Div(style={"flex": "1"}, children=[
                    dcc.Slider(id="progress-slider", min=0, max=REPLAY_LEN - 1, step=1, value=0, marks=None),
                ]),
                html.Div(id="step-counter", className="step-counter", children=f"0 / {REPLAY_LEN}"),
                html.Button("Reset", id="reset-button", className="small-btn"),
            ]),
            dcc.Interval(id="replay-interval", interval=250, n_intervals=0, disabled=True),
        ]),

        html.Div(className="panel", children=[
            html.Div(className="panel-header-row", children=[
                html.Div(className="panel-title", children="Decision inspector"),
                html.Span("At current step", className="small-btn", style={"cursor": "default"}),
            ]),
            html.Div(id="inspector-item", className="inspector-item", children="No eviction at step 0"),
            html.Div(id="inspector-desc", className="inspector-desc", children=(
                "The joint allocator weighs future reuse likelihood across expert weights "
                "and KV-cache before evicting anything."
            )),
            html.Div(className="inspector-row", children=[
                html.Span("Reuse score", className="inspector-row-label"),
                html.Span(id="inspector-score", className="inspector-value", children="n/a"),
            ]),
            html.Div(className="inspector-row", children=[
                html.Span("Evicted item", className="inspector-row-label"),
                html.Span(id="inspector-item-label", className="inspector-value", children="n/a"),
            ]),
            html.Div(className="inspector-row", children=[
                html.Span("Outcome", className="inspector-row-label"),
                html.Span(id="inspector-outcome", className="inspector-value", children="n/a"),
            ]),
            html.Div(className="inspector-footer", children="Decision uses past information only."),
        ]),
    ]),

    # Row 3: capacity, cost, evidence
    html.Div(className="row-3col", children=[
        html.Div(className="panel", children=[
            html.Div(className="mini-panel-title", children="When memory is tight"),
            html.Div(className="mini-panel-desc", children="Advantage shrinks as capacity grows."),
            dcc.Graph(figure=capacity_fig, config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.Div(className="mini-panel-title", children="Cost per million tokens"),
            html.Div(className="mini-panel-desc", children="State your own assumptions - these are illustrative."),
            html.Div(className="cost-input-row", children=[
                html.Div(children=[
                    html.Div(className="cost-input-label", children="$ / GPU-hour"),
                    dcc.Input(id="cost-gpu-hour", type="number", value=3.00, step=0.5),
                ]),
                html.Div(children=[
                    html.Div(className="cost-input-label", children="tokens / sec"),
                    dcc.Input(id="cost-tokens-sec", type="number", value=500, step=50),
                ]),
            ]),
            html.Div(id="cost-stat", className="cost-stat", children="$0.00"),
            html.Div(className="mini-panel-desc", children="saved vs best alternative, at Typical pressure"),
        ]),
        html.Div(id="evidence-panel", className="panel", children=[
            html.Div(className="mini-panel-title", children="Evidence check"),
            html.Div(className="evidence-row", children="15 prompts, 3 workload categories (code, kv-heavy, mixed)"),
            html.Div(className="evidence-row", children="Baselines: HBM-only, static 50/50 partition, LRU"),
            html.Div(className="evidence-row", children="Model: OLMoE-1B-7B-0924, 16 layers, top-8 of 64 experts"),
            html.Div(className="evidence-row", children="Timing calibrated from published NUMA/CXL latency figures"),
        ]),
    ]),

    html.Div(className="footer-bar", children=[
        html.Span("IIT BOMBAY | NEBULA x ASTERA LABS"),
        html.Span("Simulation-based; see Evidence panel for methodology"),
    ]),
])


# ---------- Callbacks ----------

def _select_preset_logic(preset):
    classes = {p: "preset-btn" for p in ["tight", "typical", "abundant"]}
    classes[preset] = "preset-btn selected"
    return preset, classes["tight"], classes["typical"], classes["abundant"]


@app.callback(
    Output("preset-store", "data"),
    Output("preset-tight", "className"),
    Output("preset-typical", "className"),
    Output("preset-abundant", "className"),
    Input("preset-tight", "n_clicks"),
    Input("preset-typical", "n_clicks"),
    Input("preset-abundant", "n_clicks"),
    prevent_initial_call=True,
)
def select_preset(*_):
    triggered = ctx.triggered_id
    preset = {"preset-tight": "tight", "preset-typical": "typical", "preset-abundant": "abundant"}[triggered]
    return _select_preset_logic(preset)


@app.callback(
    Output("hero-stat", "children"),
    Output("workload-note", "children"),
    Input("mix-slider", "value"),
    Input("preset-store", "data"),
)
def update_hero(mix_pct, preset):
    t = mix_pct / 100.0
    kv_row, code_row = kv_heavy_by_preset[preset], code_by_preset[preset]
    stall = {a: kv_row.loc[a, "total_stall_ns"] * (1 - t) + code_row.loc[a, "total_stall_ns"] * t
             for a in ALLOCATOR_ORDER}
    best_other = min(stall[a] for a in ALLOCATOR_ORDER if a != "dynamic_joint")
    improvement = (best_other - stall["dynamic_joint"]) / best_other * 100

    kv_imp = (kv_row.loc[[a for a in ALLOCATOR_ORDER if a != "dynamic_joint"], "total_stall_ns"].min()
              - kv_row.loc["dynamic_joint", "total_stall_ns"]) / kv_row.loc[[a for a in ALLOCATOR_ORDER if a != "dynamic_joint"], "total_stall_ns"].min() * 100
    code_imp = (code_row.loc[[a for a in ALLOCATOR_ORDER if a != "dynamic_joint"], "total_stall_ns"].min()
                - code_row.loc["dynamic_joint", "total_stall_ns"]) / code_row.loc[[a for a in ALLOCATOR_ORDER if a != "dynamic_joint"], "total_stall_ns"].min() * 100
    ratio = code_imp / kv_imp if kv_imp > 0 else 1

    return f"{improvement:.0f}%", f"Code-like prompts: ~{ratio:.1f}x greater benefit ({preset} pressure)"


def _move_playhead_logic(trigger, current_value):
    if trigger == "replay-interval":
        return (current_value + 1) % REPLAY_LEN
    if trigger == "step-button":
        return min(current_value + 1, REPLAY_LEN - 1)
    if trigger == "reset-button":
        return 0
    if trigger == "jump-button":
        return DIVERGENCE_STEP
    return current_value


@app.callback(
    Output("progress-slider", "value"),
    Input("replay-interval", "n_intervals"),
    Input("step-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    Input("jump-button", "n_clicks"),
    State("progress-slider", "value"),
    prevent_initial_call=True,
)
def move_playhead(n_intervals, step_clicks, reset_clicks, jump_clicks, current_value):
    return _move_playhead_logic(ctx.triggered_id, current_value)


@app.callback(
    Output("replay-interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("replay-interval", "disabled"),
)
def toggle_play(n_clicks, currently_disabled):
    if n_clicks == 0:
        return True, "\u25B6"
    now_disabled = not currently_disabled
    return now_disabled, ("\u25B6" if now_disabled else "\u23F8")


def bar_widths(experts_n, kv_n):
    total = max(experts_n + kv_n, 1)
    return f"{experts_n / total * 100:.1f}%", f"{kv_n / total * 100:.1f}%"


@app.callback(
    Output("lru-bar1-expert", "style"), Output("lru-bar1-kv", "style"),
    Output("lru-bar2-expert", "style"), Output("lru-bar2-kv", "style"),
    Output("lru-hitrate", "children"),
    Output("dyn-bar1-expert", "style"), Output("dyn-bar1-kv", "style"),
    Output("dyn-bar2-expert", "style"), Output("dyn-bar2-kv", "style"),
    Output("dyn-hitrate", "children"),
    Output("step-counter", "children"),
    Output("inspector-item", "children"),
    Output("inspector-score", "children"),
    Output("inspector-item-label", "children"),
    Output("inspector-outcome", "children"),
    Input("progress-slider", "value"),
)
def update_replay_view(step):
    lru_h = replay_data["lru"][step]
    dyn_h = replay_data["dynamic_joint"][step]

    lru_e_w, lru_k_w = bar_widths(lru_h["experts_resident"], lru_h["kv_resident"])
    dyn_e_w, dyn_k_w = bar_widths(dyn_h["experts_resident"], dyn_h["kv_resident"])
    lru_e2_w, lru_k2_w = bar_widths(lru_h["cum_expert_fetches"], lru_h["cum_kv_fetches"])
    dyn_e2_w, dyn_k2_w = bar_widths(dyn_h["cum_expert_fetches"], dyn_h["cum_kv_fetches"])

    lru_rate = lru_h["cumulative_hits"] / max(lru_h["cumulative_hits"] + lru_h["cumulative_fetches"], 1) * 100
    dyn_rate = dyn_h["cumulative_hits"] / max(dyn_h["cumulative_hits"] + dyn_h["cumulative_fetches"], 1) * 100

    ev = dyn_h.get("eviction")
    if ev is None:
        item_text, score_text, label_text, outcome_text = "No eviction at this step", "n/a", "n/a", "n/a"
    else:
        item_text = f"Evicted a {ev['kind']} item"
        score_text = str(ev["reuse_score"]) if ev["reuse_score"] is not None else "n/a (kv, protected-while-active rule)"
        label_text = ev["key_label"]
        if ev["steps_until_needed"] is not None:
            outcome_text = f"Needed again {ev['steps_until_needed']} steps later"
        else:
            outcome_text = "Never needed again in this trace (good call)"

    return (
        {"width": lru_e_w}, {"width": lru_k_w},
        {"width": lru_e2_w}, {"width": lru_k2_w},
        f"{lru_rate:.0f}%",
        {"width": dyn_e_w}, {"width": dyn_k_w},
        {"width": dyn_e2_w}, {"width": dyn_k2_w},
        f"{dyn_rate:.0f}%",
        f"{step} / {REPLAY_LEN}",
        item_text, score_text, label_text, outcome_text,
    )


@app.callback(
    Output("cost-stat", "children"),
    Input("cost-gpu-hour", "value"),
    Input("cost-tokens-sec", "value"),
)
def update_cost(gpu_hour, tokens_sec):
    if not gpu_hour or not tokens_sec:
        return "$0.00"
    typical_kv, typical_code = kv_heavy_by_preset["typical"], code_by_preset["typical"]
    best_alt = min(typical_kv.loc[a, "total_stall_ns"] for a in ALLOCATOR_ORDER if a != "dynamic_joint")
    dyn_stall = typical_kv.loc["dynamic_joint", "total_stall_ns"]
    stall_saved_ns = max(best_alt - dyn_stall, 0)
    seconds_saved_per_token = (stall_saved_ns / 1e9) / max(REPLAY_LEN, 1)
    cost_per_token = (seconds_saved_per_token / 3600) * gpu_hour
    return f"${cost_per_token * 1_000_000:.2f}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

