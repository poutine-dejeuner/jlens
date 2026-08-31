"""Optimized Jacobian: compute J^T J directly via double-backward (JVP + VJP).

Key insight: we don't need the full D×D Jacobian J. We need:
1. J̄ (mean Jacobian) = (1/N) Σ J(x) — accumulated online
2. J̄^T J̄ eigenvalues — for coherence κ_ℓ from Gram spectrum
3. Σ eigenvalues — for covariance spectrum

For the Gram matrix G = J^T J, we can compute it efficiently:
  G[i,j] = (J^T J)[i,j] = ⟨J_{:,i}, J_{:,j}⟩

This is the inner product of the i-th and j-th columns of J.
Each column J_{:,i} is ∂h_final/∂(h_ℓ)_i, which is the gradient of
h_final w.r.t. the i-th dimension of h_ℓ.

Alternatively, for any vector v: G v = J^T (J v)
  - J v = jvp(h_final, h_ℓ, v) — forward-mode differentiation
  - J^T u = vjp(h_final, h_ℓ, u) — reverse-mode differentiation
  - So G v = vjp(jvp(v)) — a "double autodiff" operation

But torch doesn't easily support JVP + VJP chaining for full models.

Simpler approach: compute G directly from vjp outputs.
For each dimension d of h_final:
  ∂(h_final)_d / ∂h_ℓ is a vector of size D (from vjp with one-hot grad_output)
  This is one row of J. Call it r_d = J[d, :] (size D).
  
Then G = J^T J has entries G[i,j] = Σ_d J[d,i] * J[d,j]
Which is: the matrix with rows r_d, G = Σ_d r_d^T r_d

So we can accumulate G online during the sequential vjp:
  After computing row r_d, update G += outer(r_d, r_d)
  And accumulate J̄[d,:] += r_d
  
With this, we accumulate both J̄ and G (hence J̄^T J̄) from the same vjp calls.
We also accumulate per-prompt G to compute Σ later.
  
This doesn't reduce the vjp count but consolidates the work.

--- BETTER APPROACH: Randomized SVD ---

Even better: we can compute the top-k eigenvalues of G = J^T J using
randomized algorithms without ever forming J explicitly.

For a D×D matrix J, to compute the top-k eigenvalues:
1. Generate random matrix Ω of size D×k
2. Compute Y = J Ω via JVP (k directional derivatives)
3. QR factorize Y
4. Compute B = Q^T J^T J Q via VJP then JVP

But this requires JVP, which needs forward-mode AD...

--- PRACTICAL APPROACH: Exploit the residual chain ---

Since h_{ℓ+1} = h_ℓ + F_ℓ(h_ℓ), we have:
  J_ℓ = ∂h_final/∂h_ℓ = ∂h_final/∂h_{ℓ+1} · (I + ∂F_ℓ/∂h_ℓ)
       = J_{ℓ+1} · (I + ∂F_ℓ/∂h_ℓ)

Starting from J_{L-1} = I + ∂F_{L-1}/∂h_{L-1} (a single block's Jacobian),
we can compute J_ℓ recursively downward.

The per-block Jacobian M_ℓ = I + ∂F_ℓ/∂h_ℓ only involves ONE transformer block,
which is much cheaper to differentiate through (smaller graph).

But we still need ∂F_ℓ/∂h_ℓ, which is a D×D Jacobian of a transformer block.

--- PRACTICAL APPROACH: Just use fewer dimensions ---

Another option: instead of full D=768, use a random projection or
compute only the first k principal components.

Actually, wait. Let me check: the paper computes J̄_ℓ as the D×D Jacobian
of h_final w.r.t h_ℓ, AND the Gram matrix J̄^T J̄. 

For coherence κ_ℓ = (Σ λ_i)^2 / (D · Σ λ_i^2), we need ALL eigenvalues.
For q_eff, we need the MP bulk fraction.
For power-law tail, we need the full eigenvalue distribution.

So we DO need the full spectrum (all D eigenvalues). We can't reduce dimension.

Given the constraints, the fastest approach is:
1. Use the sequential vjp as implemented
2. Compute in fp16 (mixed precision for Jacobian accumulation)
3. Check if we can reduce prompt count (1000 may be overkill for initial analysis)

Let me check if fp32 is needed for vjp stability, or if fp16 works.
"""

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

# Test: single vjp row in fp16 vs fp32
prompt = "The capital of France is Paris, and the capital of Germany"
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
S = tokens.input_ids.shape[1]
last_pos = S - 1

hooked = {}
def hook_fn(module, inputs, output):
    hooked["h"] = output[0] if isinstance(output, tuple) else output

hook = model.gpt_neox.layers[6].register_forward_hook(hook_fn)
outputs = model(**tokens, output_hidden_states=True)
h_final = outputs.hidden_states[-1]
h_mid = hooked["h"]
hook.remove()

print(f"\n--- Timing comparison ---")
for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
    hf = h_final.to(dtype)
    hm = h_mid.to(dtype)
    
    times = []
    for d in range(10):
        grad_output = torch.zeros_like(hf)
        grad_output[0, last_pos, d] = 1.0
        
        t0 = time.perf_counter()
        grads = torch.autograd.grad(
            outputs=hf, inputs=hm,
            grad_outputs=grad_output,
            retain_graph=(d < 9),
            create_graph=False, allow_unused=False,
        )[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    avg = sum(times) / len(times)
    print(f"  {dtype_name}: {avg*1000:.1f}ms per row → {avg*D:.1f}s per layer → {avg*D*12:.0f}s per prompt")

print(f"\n--- Checking if we can reduce model overhead ---")
print(f"  Current: load model once, do 12 layers x D dims per prompt")
print(f"  Alternative: per-layer backward through only remaining layers")
print(f"  This would cut compute by ~2x (each layer only backprops through fewer blocks)")
print(f"  But requires separate forward pass per layer...")
