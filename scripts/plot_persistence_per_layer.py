#!/usr/bin/env python3
"""Plot per-layer persistence (H0-H3) for singleton vs contextual token
embeddings across Qwen3.6-27B's 64 layers, compared to random baseline."""

import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = '/data/jlens/qwen3.6-27b/persistence_per_layer.pkl'
OUT_DIR = Path('/mnt/home/jacob/vl/repos/jlens/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load ----
with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

sng = data['singleton']
ctx = data['contextual']
rand = data['random']
cfg = data['config']
n_layers = cfg['n_layers']
rand_arr = np.array(rand)
rand_mean = rand_arr.mean(axis=0)
rand_std = rand_arr.std(axis=0)

# Per-layer arrays (omit H0 which is always 224 for all)
layers = np.arange(n_layers)
sng_h1 = np.array([sng[l][1] for l in range(n_layers)])
sng_h2 = np.array([sng[l][2] for l in range(n_layers)])
sng_h3 = np.array([sng[l][3] for l in range(n_layers)])
ctx_h1 = np.array([ctx[l][1] for l in range(n_layers)])
ctx_h2 = np.array([ctx[l][2] for l in range(n_layers)])
ctx_h3 = np.array([ctx[l][3] for l in range(n_layers)])

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle('Per-Layer Persistence: Qwen3.6-27B Token Embeddings vs Random',
             fontsize=14, fontweight='bold')

colors = {'singleton': '#2196F3', 'contextual': '#FF5722', 'random': '#757575'}
alpha_fill = 0.15

# Panel 1: H1
ax = axes[0, 0]
ax.fill_between([0, n_layers-1],
                [rand_mean[1]-rand_std[1]]*2,
                [rand_mean[1]+rand_std[1]]*2,
                color=colors['random'], alpha=alpha_fill, label='Random H1')
ax.axhline(rand_mean[1], color=colors['random'], linestyle='--', linewidth=1)
ax.plot(layers, sng_h1, 'o-', color=colors['singleton'], markersize=3, linewidth=1.2, label='Singleton')
ax.plot(layers, ctx_h1, 's-', color=colors['contextual'], markersize=3, linewidth=1.2, label='Contextual')
ax.set_ylabel('Number of H1 bars')
ax.set_title(f'H1 (1-cycles) — Random: {rand_mean[1]:.0f}±{rand_std[1]:.0f}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 2: H2
ax = axes[0, 1]
ax.fill_between([0, n_layers-1],
                [rand_mean[2]-rand_std[2]]*2,
                [rand_mean[2]+rand_std[2]]*2,
                color=colors['random'], alpha=alpha_fill, label='Random H2')
ax.axhline(rand_mean[2], color=colors['random'], linestyle='--', linewidth=1)
ax.plot(layers, sng_h2, 'o-', color=colors['singleton'], markersize=3, linewidth=1.2, label='Singleton')
ax.plot(layers, ctx_h2, 's-', color=colors['contextual'], markersize=3, linewidth=1.2, label='Contextual')
ax.set_ylabel('Number of H2 bars')
ax.set_title(f'H2 (2-cycles) — Random: {rand_mean[2]:.0f}±{rand_std[2]:.0f}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 3: H3
ax = axes[1, 0]
ax.fill_between([0, n_layers-1],
                [rand_mean[3]-rand_std[3]]*2,
                [rand_mean[3]+rand_std[3]]*2,
                color=colors['random'], alpha=alpha_fill, label='Random H3')
ax.axhline(rand_mean[3], color=colors['random'], linestyle='--', linewidth=1)
ax.plot(layers, sng_h3, 'o-', color=colors['singleton'], markersize=3, linewidth=1.2, label='Singleton')
ax.plot(layers, ctx_h3, 's-', color=colors['contextual'], markersize=3, linewidth=1.2, label='Contextual')
ax.set_xlabel('Layer')
ax.set_ylabel('Number of H3 bars')
ax.set_title(f'H3 (3-cycles) — Random: {rand_mean[3]:.0f}±{rand_std[3]:.0f}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 4: ctx/sng H1 ratio
ax = axes[1, 1]
ratio = np.where(sng_h1 > 0, ctx_h1 / sng_h1, np.nan)
ax.plot(layers, ratio, 'o-', color='#9C27B0', markersize=3, linewidth=1.2)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(layers, 1.0, ratio, where=(ratio > 1.0),
                color='#FF5722', alpha=0.15, label='ctx > sng')
ax.fill_between(layers, ratio, 1.0, where=(ratio < 1.0),
                color='#2196F3', alpha=0.15, label='ctx < sng')
# Shade background: early (0-35) vs deep (36-63)
ax.axvspan(0, 35, alpha=0.05, color='orange', label='Early (0-35)')
ax.axvspan(35, 63, alpha=0.05, color='purple', label='Deep (36-63)')
ax.set_xlabel('Layer')
ax.set_ylabel('Contextual / Singleton H1 ratio')
ax.set_title('Context vs Singleton: H1 Ratio')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = OUT_DIR / 'persistence_per_layer.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved to {outpath}')
plt.close()

# ---- Second figure: Combined H1-H3 on one plot with z-scores ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Per-Layer Persistence: z-scores vs Random', fontsize=14, fontweight='bold')

# Compute z-scores
sng_z1 = (sng_h1 - rand_mean[1]) / rand_std[1]
sng_z2 = (sng_h2 - rand_mean[2]) / rand_std[2]
sng_z3 = (sng_h3 - rand_mean[3]) / rand_std[3]
ctx_z1 = (ctx_h1 - rand_mean[1]) / rand_std[1]
ctx_z2 = (ctx_h2 - rand_mean[2]) / rand_std[2]
ctx_z3 = (ctx_h3 - rand_mean[3]) / rand_std[3]

# Singleton z-scores
ax1.plot(layers, sng_z1, 'o-', color='#2196F3', markersize=3, linewidth=1.2, label=f'H1 (mean {sng_h1.mean():.0f})')
ax1.plot(layers, sng_z2, 's-', color='#4CAF50', markersize=3, linewidth=1.2, label=f'H2 (max {sng_h2.max()})')
ax1.plot(layers, sng_z3, '^-', color='#FF9800', markersize=3, linewidth=1.2, label=f'H3 (max {sng_h3.max()})')
ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Layer')
ax1.set_ylabel('z-score vs random')
ax1.set_title('Singleton')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Contextual z-scores
ax2.plot(layers, ctx_z1, 'o-', color='#FF5722', markersize=3, linewidth=1.2, label=f'H1 (mean {ctx_h1.mean():.0f})')
ax2.plot(layers, ctx_z2, 's-', color='#4CAF50', markersize=3, linewidth=1.2, label=f'H2 (max {ctx_h2.max()})')
ax2.plot(layers, ctx_z3, '^-', color='#FF9800', markersize=3, linewidth=1.2, label=f'H3 (max {ctx_h3.max()})')
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Layer')
ax2.set_ylabel('z-score vs random')
ax2.set_title('Contextual')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
outpath2 = OUT_DIR / 'persistence_zscore.png'
fig.savefig(outpath2, dpi=150, bbox_inches='tight')
print(f'Saved to {outpath2}')
plt.close()

# ---- Third figure: SVD spectrum comparison at key layers ----
# (No data for this directly — would need eigenvalues from the SVD step)
# Skip for now; could be added later

print('Done!')
print(f'Key numbers:')
print(f'  Random H1: {rand_mean[1]:.0f}±{rand_std[1]:.0f}')
print(f'  Singleton H1: {sng_h1.min()}-{sng_h1.max()} (mean {sng_h1.mean():.1f})')
print(f'  Contextual H1: {ctx_h1.min()}-{ctx_h1.max()} (mean {ctx_h1.mean():.1f})')
print(f'  Singleton Layer 59 anomaly: H1={sng[59][1]}, H2={sng[59][2]}, H3={sng[59][3]}')
