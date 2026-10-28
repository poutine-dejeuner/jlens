"""Shared playback for the spectra dashboards: vertical, log-space interpolation.

Plotly's built-in frame transitions tween in pixel space and pair points by
index, which produces kinks. Instead, every checkpoint's log10(y) values are
embedded once and a small requestAnimationFrame loop blends adjacent
checkpoints point-by-point on a fixed x-grid (x never changes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STEP_MS = 700

PLAYBACK_JS = """
(function () {
const gd = document.getElementById("{plot_id}");
const LOGY = __LOGY__;
const LOGSPACE = __LOGSPACE__;
const STEP_MS = __STEP_MS__;
const nFrames = LOGY.length;
const traceIdx = LOGY[0].map((_, k) => k);
let cur = 0, playing = false, raf = null, segStart = 0, busy = false;

function blend(a, b, t) {
    return traceIdx.map((k) => {
        const ya = LOGY[a][k], yb = LOGY[b][k];
        const f = LOGSPACE ? (v) => Math.pow(10, v) : (v) => v;
        if (t === 0 || a === b) return ya.map((v) => (v === null ? null : f(v)));
        return ya.map((v, i) => {
            const w = yb[i];
            return v === null || w === null ? null : f(v + (w - v) * t);
        });
    });
}
function render(a, b, t) {
    if (busy) return Promise.resolve();
    busy = true;
    return Plotly.restyle(gd, {y: blend(a, b, t)}, traceIdx).finally(() => { busy = false; });
}
function setSlider(i) { Plotly.relayout(gd, {"sliders[0].active": i}); }
function stop() {
    playing = false;
    if (raf !== null) { window.cancelAnimationFrame(raf); raf = null; }
}
function tick(now) {
    if (!playing) return;
    let t = (now - segStart) / STEP_MS;
    if (t >= 1) {
        cur = (cur + 1) % nFrames;
        segStart = now;
        t = 0;
        setSlider(cur);
    }
    // At the last checkpoint hold, then jump (no backwards sweep) to the first.
    render(cur, Math.min(cur + 1, nFrames - 1), t);
    raf = window.requestAnimationFrame(tick);
}
gd.on("plotly_buttonclicked", (ev) => {
    if (ev.button.label === "Loop checkpoints") {
        stop();
        cur = Math.max(0, gd.layout.sliders[0].active || 0);
        playing = true;
        segStart = performance.now();
        raf = window.requestAnimationFrame(tick);
    } else if (ev.button.label === "Pause") {
        stop();
        window.setTimeout(() => render(cur, cur, 0), 60);
    }
});
gd.on("plotly_sliderchange", (ev) => {
    stop();
    let i = ev.slider && ev.step ? ev.slider.steps.indexOf(ev.step) : -1;
    if (i < 0) i = gd.layout.sliders[0].active || 0;
    cur = i;
    window.setTimeout(() => render(cur, cur, 0), 60);
});
})();
"""


def _axis_name(axis_ref: str | None, kind: str) -> str:
    ref = axis_ref or kind
    return kind + "axis" + ref[1:] if len(ref) > 1 else kind + "axis"


def add_axis_scale_menus(figure, reference_x, log_y, log_space: bool) -> None:
    """Add dropdowns toggling each axis between log and linear, with data-fitted ranges."""
    extents: dict[str, list[float]] = {}

    def update(name: str, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size:
            lo = extents.setdefault(name, [np.inf, -np.inf])
            lo[0] = min(lo[0], float(values.min()))
            lo[1] = max(lo[1], float(values.max()))

    for k, trace in enumerate(figure.data):
        if reference_x[k] is not None:
            update(_axis_name(trace.xaxis, "x"), reference_x[k])
        ys = np.array([[np.nan if v is None else v for v in frame[k]] for frame in log_y], dtype=float)
        update(_axis_name(trace.yaxis, "y"), 10.0 ** ys if log_space else ys)

    def ranges(name: str) -> tuple[list[float], list[float]]:
        lo, hi = extents[name]
        positive = lo if lo > 0 else hi * 1e-6
        log_range = [float(np.log10(positive)), float(np.log10(hi))]
        pad = 0.04 * (hi - min(lo, 0.0))
        linear_range = [min(lo, 0.0) if lo >= 0 else lo - pad, hi + pad]
        return log_range, linear_range

    menus = [m.to_plotly_json() for m in figure.layout.updatemenus]
    for index, (kind, x_pos) in enumerate((("x", 0.58), ("y", 0.75))):
        names = sorted(n for n in extents if n.startswith(kind + "axis"))
        if not names:
            continue
        current = (figure.layout[names[0]].type or "linear")
        buttons = []
        for scale in ("log", "linear"):
            args = {}
            for name in names:
                log_range, linear_range = ranges(name)
                args[f"{name}.type"] = scale
                args[f"{name}.range"] = log_range if scale == "log" else linear_range
                args[f"{name}.autorange"] = False
            buttons.append({"label": f"{kind}-axis: {scale}", "method": "relayout", "args": [args]})
        menus.append({
            "type": "dropdown", "x": x_pos, "y": 1.18, "xanchor": "left", "yanchor": "top",
            "buttons": buttons, "active": 0 if current == "log" else 1, "showactive": True,
        })
    # Stack all dropdowns vertically in the right margin, above the legend.
    row = 0
    for menu in menus:
        if menu.get("type") == "dropdown":
            menu.update({"x": 1.02, "xanchor": "left", "y": 1.0 - 0.055 * row, "yanchor": "top",
                         "font": {"size": 10}, "pad": {"l": 0, "r": 0, "t": 0, "b": 0}})
            row += 1
    figure.update_layout(updatemenus=menus, legend={"x": 1.02, "xanchor": "left", "y": 1.0 - 0.055 * row - 0.02, "yanchor": "top"})


def write_dashboard(figure, output: Path, step_ms: int = STEP_MS, log_space: bool = True) -> None:
    """Write `figure` as standalone HTML with custom vertical-interpolation playback.

    `log_space=True` blends log10(y) (positive data); False blends y linearly (zeros allowed).
    """
    frames = list(figure.frames)
    if not frames:
        raise ValueError("Figure has no frames to play back.")
    n_traces = len(figure.data)
    steps = [frame.name for frame in frames]

    reference_x: list[np.ndarray | None] = [None] * n_traces
    log_y: list[list[list[float | None]]] = []
    for frame in frames:
        if len(frame.data) != n_traces:
            raise ValueError("Every frame must update every trace.")
        per_trace = []
        for k, trace in enumerate(frame.data):
            x = np.asarray(trace.x if trace.x is not None else [], dtype=float)
            y = np.asarray(trace.y if trace.y is not None else [], dtype=float)
            if x.size:
                if reference_x[k] is None:
                    reference_x[k] = x
                elif not np.array_equal(reference_x[k], x):
                    raise ValueError(f"x-grid of trace {k} changes between frames (step {frame.name}).")
            if log_space:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.log10(np.where(y > 0, y, np.nan))
                per_trace.append([round(float(v), 4) if np.isfinite(v) else None for v in values])
            else:
                per_trace.append([float(f"{v:.5g}") if np.isfinite(v) else None for v in y])
        log_y.append(per_trace)

    for k, trace in enumerate(figure.data):
        if reference_x[k] is not None:
            trace.x = reference_x[k]
            trace.y = [None if v is None else (10.0 ** v if log_space else v) for v in log_y[0][k]]

    slider = figure.layout.sliders[0].to_plotly_json()
    def _format_step(step: str) -> str:
        try:
            return f"{int(step):,}"
        except ValueError:
            return step

    slider["steps"] = [{"label": _format_step(step), "method": "skip", "args": []} for step in steps]
    figure.update_layout(sliders=[slider])
    figure.frames = []

    add_axis_scale_menus(figure, reference_x, log_y, log_space)

    script = (
        PLAYBACK_JS.replace("__LOGY__", json.dumps(log_y, separators=(",", ":")))
        .replace("__LOGSPACE__", "true" if log_space else "false")
        .replace("__STEP_MS__", str(step_ms))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs="inline", full_html=True, post_script=script)
