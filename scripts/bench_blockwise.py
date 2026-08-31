"""Benchmark per-block Jacobian computation."""
import os
os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import triton
triton.knobs.build.impl = "dummy"

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-160m-deduped"
STEP = 143000

print(f"Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=f"step{STEP}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=f"step{STEP}",
    torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
model.eval()
D = model.config.hidden_size
L = len(model.gpt_neox.layers)
print(f"Model: {L} layers, d_model={D}")

prompt = "The capital of France is Paris."
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
S = tokens.input_ids.shape[1]
last_pos = S - 1
print(f"Seq len: {S}, last_pos: {last_pos}")

# Forward pass to get hidden states
position_ids = torch.arange(S, device=model.device).unsqueeze(0)

with torch.no_grad():
    outputs = model(**tokens, output_hidden_states=True)
hs = [h.clone() for h in outputs.hidden_states]
# hs[0] = embedding output, hs[ℓ+1] = output of layer ℓ, hs[L] = final

# --- Benchmark: per-block Jacobian for ONE block ---
k = L // 2  # middle layer
print(f"\n--- Benchmarking block {k} Jacobian ---")

block = model.gpt_neox.layers[k]

# Test: run block forward on the saved hidden state
# The block takes position_embeddings (cos, sin) from the model's rotary_emb
h_in = hs[k].clone().detach().requires_grad_(True)  # (1, S, D)

# Generate position embeddings
rotary_emb = model.gpt_neox.rotary_emb
position_embeddings = rotary_emb(h_in, position_ids)

# Fix attention mask dtype (must be float or bool, not long)
attn_mask = tokens.attention_mask.to(dtype=torch.float16)

# Also need causal mask - the block expects it through the model's internal processing
# Actually, SDPA needs a 4D mask. Let me use the model's internal mask generation.
# For simplicity, pass a boolean causal mask
causal_mask = torch.tril(torch.ones(S, S, device=model.device, dtype=torch.bool))
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

t0 = time.perf_counter()
# Block expects: (hidden_states, attention_mask, position_ids, position_embeddings)
h_out = block(
    h_in, attention_mask=causal_mask, 
    position_ids=position_ids, position_embeddings=position_embeddings,
)
if isinstance(h_out, tuple):
    h_out = h_out[0]
torch.cuda.synchronize()
t_fwd = time.perf_counter() - t0
print(f"  Block forward: {t_fwd*1000:.1f}ms")

# Test: one vjp row
grad_output = torch.zeros_like(h_out)
grad_output[0, last_pos, 0] = 1.0

t0 = time.perf_counter()
grads = torch.autograd.grad(
    outputs=h_out, inputs=h_in,
    grad_outputs=grad_output,
    retain_graph=True,
    create_graph=False, allow_unused=False,
)[0]
torch.cuda.synchronize()
t_one = time.perf_counter() - t0
print(f"  One vjp row: {t_one*1000:.1f}ms")
print(f"  Full block Jacobian ({D} rows): {t_one*D:.1f}s")
print(f"  {L} blocks: {t_one*D*L:.1f}s")

# Test: multiple rows to measure stable timing
print("\n  Measuring 10 rows...")
times = []
for d in range(10):
    grad_output = torch.zeros_like(h_out)
    grad_output[0, last_pos, d] = 1.0
    t0 = time.perf_counter()
    grads = torch.autograd.grad(
        outputs=h_out, inputs=h_in,
        grad_outputs=grad_output,
        retain_graph=(d < 9),
        create_graph=False, allow_unused=False,
    )[0]
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
avg = sum(times) / len(times)
print(f"  Average: {avg*1000:.1f}ms per row")
print(f"  Per block: {avg*D:.1f}s")
print(f"  All {L} blocks: {avg*D*L:.1f}s")
print(f"  Plus {L-1} matrix multiplies: negligible")

# Compare: old approach (full model backprop)
print(f"\n--- Comparison ---")
print(f"  Old (full graph): ~7.6s/layer × 12 = ~91s/prompt")
print(f"  New (per-block):  {avg*D:.1f}s/block × {L} = {avg*D*L:.1f}s/prompt")
print(f"  Speedup: {91/(avg*D*L):.1f}x")
