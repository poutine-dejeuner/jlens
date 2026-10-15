#!/usr/bin/env python3
"""Interactive Plotly dashboard for Pythia Jacobian lens analysis.

Single unified view: all scalar metrics over training steps, toggleable.
"""
import json, argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def load_val_loss(results_dir: Path) -> dict:
    with open(results_dir / "val_loss_all.json") as f:
        entries = json.load(f)
    return {int(e["step"].replace("step", "")): e for e in entries}


def load_spectra_attrs(results_dir: Path) -> dict:
    """{metric_name: {step: np.array([n_layers])}}"""
    import h5py
    step_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", "")),
    )
    metrics = {}
    n_layers = None
    for d in step_dirs:
        step = int(d.name.replace("step", ""))
        sf = d / "spectra.h5"
        if not sf.exists():
            continue
        with h5py.File(sf, "r") as f:
            if n_layers is None:
                n_layers = f.attrs.get("n_layers", sum(1 for k in f.keys() if k.startswith("layer_")))
            for layer_idx in range(n_layers):
                lg = f[f"layer_{layer_idx}"]
                for attr_name in lg.attrs:
                    if attr_name not in metrics:
                        metrics[attr_name] = {}
                    if step not in metrics[attr_name]:
                        metrics[attr_name][step] = np.full(n_layers, np.nan)
                    metrics[attr_name][step][layer_idx] = float(lg.attrs[attr_name])
    return metrics, n_layers


def normalize(arr):
    """Min-max normalize to [0,1]."""
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if amax == amin:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def build_dashboard(results_dir: Path, cross_join_path: Path, output_path: Path):
    val_loss = load_val_loss(results_dir)
    attrs, n_layers = load_spectra_attrs(results_dir)
    loss_steps = np.array(sorted(val_loss.keys()))
    loss_vals = np.array([val_loss[s]["loss"] for s in loss_steps])
    ppl_vals = np.array([val_loss[s]["ppl"] for s in loss_steps])

    # --- Collect all metrics by layer-mean and per-layer ---
    # Each metric: {step: [layers]} -> mean over layers
    metric_means = {}  # name -> (steps, values, visible_default)
    metric_per_layer = {}  # name -> {layer: (steps, values)}

    default_visible = {"loss", "ppl", "coherence", "effective_rank", "power_law_alpha", "n_spikes"}

    for mname in sorted(attrs.keys()):
        mdata = attrs[mname]
        msteps = sorted(mdata.keys())
        # Mean over layers
        means = np.array([np.nanmean(mdata[s]) for s in msteps])
        if np.all(np.isnan(means)):
            continue
        metric_means[mname] = (msteps, means, mname in default_visible)
        # Per-layer
        metric_per_layer[mname] = {}
        for li in range(n_layers):
            vals = np.array([mdata[s][li] for s in msteps])
            if np.all(np.isnan(vals)):
                continue
            metric_per_layer[mname][li] = (msteps, vals)

    # --- Build figure ---
    fig = go.Figure()

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]

    # --- PANEL 1: Mean metrics + Loss (always visible initially) ---
    # Loss (primary y)
    fig.add_trace(go.Scatter(
        x=loss_steps, y=loss_vals,
        mode="lines+markers", name="Val Loss",
        line=dict(color="black", width=2.5),
        yaxis="y1",
        hovertemplate="Step %{x}<br>Loss: %{y:.4f}<extra></extra>",
        visible=True,
    ))
    # PPL (secondary y)
    fig.add_trace(go.Scatter(
        x=loss_steps, y=ppl_vals,
        mode="lines", name="PPL",
        line=dict(color="gray", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Step %{x}<br>PPL: %{y:.1f}<extra></extra>",
        visible=True,
    ))

    # Mean spectral metrics
    color_idx = 0
    for mname, (msteps, mvals, visible) in metric_means.items():
        if mname in ("loss", "ppl"):
            continue
        fig.add_trace(go.Scatter(
            x=msteps, y=mvals,
            mode="lines+markers", name=f"{mname} (mean)",
            line=dict(color=colors[color_idx % len(colors)], width=1.5),
            yaxis="y1",
            hovertemplate=f"Step %{{x}}<br>{mname}: %{{y:.4f}}<extra></extra>",
            visible=True if visible else "legendonly",
        ))
        color_idx += 1

    # --- PANEL 1B: Per-layer metrics (hidden by default) ---
    for mname in sorted(metric_per_layer.keys()):
        for li in sorted(metric_per_layer[mname].keys()):
            msteps, mvals = metric_per_layer[mname][li]
            fig.add_trace(go.Scatter(
                x=msteps, y=mvals,
                mode="lines", name=f"{mname} L{li}",
                line=dict(color=colors[li % len(colors)], width=0.8),
                opacity=0.5,
                yaxis="y3" if "sigma" in mname.lower() else "y1",
                hovertemplate=f"Step %{{x}}<br>{mname} L{li}: %{{y:.4f}}<extra></extra>",
                visible="legendonly",
            ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="Pythia-12B: Spectral Evolution + Training Loss",
            font=dict(size=18),
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top", y=0.98,
            xanchor="left", x=1.01,
            font=dict(size=10),
            itemsizing="constant",
        ),
        xaxis=dict(
            title="Training Step",
            type="log",
            dtick=1,
        ),
        yaxis=dict(
            title="Value (raw)",
            side="left",
        ),
        yaxis2=dict(
            title="Perplexity",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        yaxis3=dict(
            title="σ / Overlap",
            overlaying="y",
            side="right",
            position=0.93,
            showgrid=False,
        ),
        margin=dict(r=220),
        width=1400,
        height=800,
    )

    # --- Buttons: normalize toggle, metric group toggles ---
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01, y=1.12,
                buttons=[
                    dict(label="Raw Values", method="relayout", args=["yaxis.title.text", "Value (raw)"]),
                    dict(label="Min-Max Normalized", method="update",
                         args=[{"yaxis.title.text": "Value (normalized)"}]),
                ],
            ),
            dict(
                type="buttons",
                direction="right",
                x=0.01, y=1.06,
                buttons=[
                    dict(label="Show All", method="update",
                         args=[{"visible": [True] * len(fig.data)}]),
                    dict(label="Means Only", method="update",
                         args=[{"visible": [
                             t.visible is True or t.visible == True
                             if "mean" in t.name or t.name in ("Val Loss", "PPL")
                             else (t.visible if t.visible != "legendonly" else "legendonly")
                             for t in fig.data
                         ]}]),
                    dict(label="Loss Only", method="update",
                         args=[{"visible": [
                             True if t.name in ("Val Loss", "PPL") else "legendonly"
                             for t in fig.data
                         ]}]),
                ],
            ),
        ]
    )

    # --- Write HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pythia-12B: J-Lens Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  #container {{ max-width: 1500px; margin: 0 auto; padding: 10px; }}
  .info-box {{
    background: #f0f4f8; border-left: 4px solid #1f77b4;
    padding: 10px 16px; margin-bottom: 10px; border-radius: 4px;
    font-size: 13px; color: #333;
  }}
</style>
</head>
<body>
<div id="container">
  <div class="info-box">
    <strong>Pythia-12B-deduped Jacobian Lens Dashboard</strong><br>
    Hover for values. Double-click legend items to isolate. Toggle between raw and normalized views.<br>
    <em>Mean</em> traces are layer-averaged. <em>L#</em> traces are individual layers (hidden by default).
  </div>
  <div id="main"></div>
</div>
<script>
Plotly.newPlot("main", {json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)}, {{
  responsive: true,
  displaylogo: false,
}});
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard → {output_path} ({len(html):,} bytes)")
    print(f"  {len(fig.data)} traces, {len(loss_steps)} checkpoints, {n_layers} layers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--cross-join", default="eval_data/12b_cross_join.npz")
    parser.add_argument("--output", default="figures/pythia-12b/dashboard.html")
    args = parser.parse_args()
    build_dashboard(Path(args.results_dir), Path(args.cross_join), Path(args.output))
