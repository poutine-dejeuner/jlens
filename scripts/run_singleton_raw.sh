#!/usr/bin/env bash
#SBATCH --job-name=sng-raw
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sng_raw_%j.out
#SBATCH --error=logs/sng_raw_%j.err

# Compute singleton forward passes for all 248K tokens on Qwen 27B.
# Save raw hidden states + token manifest for circularity analysis.
set -euo pipefail
echo "=== Singleton Raw Representations ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import torch, numpy as np, time, json, h5py
from pathlib import Path
from transformers import AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
BATCH_SIZE = 512
VOCAB_SIZE = 248320
D_MODEL = 5120

print("Loading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.6-27B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
print(f"  Loaded in {time.time()-t0:.1f}s")

# Compute singleton representations
print(f"\nComputing singleton forward passes for {VOCAB_SIZE} tokens...")
print(f"  Batch size: {BATCH_SIZE}")

all_hidden = np.zeros((VOCAB_SIZE, D_MODEL), dtype=np.float16)
t_start = time.time()

for start in tqdm(range(0, VOCAB_SIZE, BATCH_SIZE), desc="singleton"):
    end = min(start + BATCH_SIZE, VOCAB_SIZE)
    token_ids = list(range(start, end))
    
    # Create batch: each is a single-token sequence
    # Pad to same length (1 token each)
    input_ids = torch.tensor([[tid] for tid in token_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        # Last layer hidden state at position 0
        hidden = outputs.hidden_states[-1][:, 0, :]  # (batch, d_model)
        all_hidden[start:end] = hidden.float().cpu().numpy().astype(np.float16)

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"  Shape: {all_hidden.shape}, dtype: {all_hidden.dtype}")

# Save
print("\nSaving...")
out_path = DATA_DIR / 'singleton' / 'representations.h5'
with h5py.File(out_path, 'w') as f:
    f.create_dataset('representations', data=all_hidden, compression='gzip', compression_opts=4)
    f.create_dataset('token_ids', data=np.arange(VOCAB_SIZE, dtype=np.int32))
    f.attrs['vocab_size'] = VOCAB_SIZE
    f.attrs['d_model'] = D_MODEL
    f.attrs['date'] = time.strftime('%Y-%m-%d %H:%M:%S')

mb = out_path.stat().st_size / 1024 / 1024
print(f"  Saved {out_path} ({mb:.0f} MB)")

# Also save a compact manifest (just token IDs + decoded strings)
from transformers import AutoTokenizer
print("Building manifest...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
id_to_token = {v: k for k, v in tok.get_vocab().items()}

manifest = {
    'token_ids': list(range(VOCAB_SIZE)),
    'decoded': [id_to_token[i] for i in range(VOCAB_SIZE)],
    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
}
manifest_path = DATA_DIR / 'singleton' / 'token_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)
print(f"  Saved {manifest_path} (len={len(manifest['decoded'])})")

# Cleanup
del model
torch.cuda.empty_cache()
print("\nDone!")
PYEOF
