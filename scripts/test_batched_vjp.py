"""Test batched vjp: compute all D rows of the Jacobian in one call."""
import os
os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import triton
triton.knobs.build.impl = "dummy"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-160m-deduped"
STEP = 143000

print(f"Loading {MODEL_ID} @ step{STEP}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=f"step{STEP}", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=f"step{STEP}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
D = model.config.hidden_size  # 768

prompt = "The capital of France is"
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
S = tokens.input_ids.shape[1]
last_pos = S - 1

# Forward + hook at layer 6
hooked = {}
def hook_fn(module, inputs, output):
    if isinstance(output, tuple):
        hooked["h"] = output[0]
    else:
        hooked["h"] = output

hook = model.gpt_neox.layers[6].register_forward_hook(hook_fn)

outputs = model(**tokens, output_hidden_states=True)
h_final = outputs.hidden_states[-1]  # (1, S, D)
h_mid = hooked["h"]  # (1, S, D)

hook.remove()

print(f"h_final: {h_final.shape}, h_mid: {h_mid.shape}")

# --- Method 1: Sequential (benchmark) ---
import time

jac_seq = torch.zeros(D, D, device=h_final.device, dtype=h_final.dtype)

# Test just 30 rows for timing
N_TEST = 30
t0 = time.perf_counter()
for d in range(N_TEST):
    grad_output = torch.zeros_like(h_final)
    grad_output[0, last_pos, d] = 1.0
    grads = torch.autograd.grad(
        outputs=h_final, inputs=h_mid,
        grad_outputs=grad_output,
        retain_graph=(d < N_TEST - 1),
        create_graph=False, allow_unused=False,
    )[0]
    jac_seq[d] = grads[0, last_pos, :]
t_seq = time.perf_counter() - t0
print(f"Sequential ({N_TEST} rows): {t_seq:.3f}s → {(t_seq/N_TEST)*D:.1f}s for all {D} rows")

# --- Method 2: Batched grad_outputs ---
# We want ∂(h_final[0, last_pos, :]) / ∂(h_mid[0, last_pos, :])
# h_final[0, last_pos, :] is shape (D,) so the Jacobian is (D, D)
# 
# Trick: use vmap-like batching - create grad_output of shape (D, 1, S, D)
# where grad_output[d, 0, last_pos, d] = 1.0
# But this gives gradient w.r.t. a (D,)-shaped output, not per-dimension

# Better: create a proxy function that maps h_mid → h_final[last_pos, :]
# Then use torch.autograd.functional.jacobian

print("\n--- Method 2: torch.autograd.functional.jacobian ---")

# Re-do forward with fresh hook
hooked2 = {}
def hook_fn2(module, inputs, output):
    if isinstance(output, tuple):
        hooked2["h"] = output[0]
    else:
        hooked2["h"] = output

hook2 = model.gpt_neox.layers[6].register_forward_hook(hook_fn2)

# For the functional API, we need to define a function f: h_mid → h_final_last
# But h_mid feeds into the rest of the model. We can't easily extract this.

# Alternative: define f(h_mid) that manually runs the remaining layers
# This is complex. Let me try the batched grad_output approach instead.

# --- Method 3: Batched grad_output with autograd.grad ---
# The trick: create grad_output with shape (D, 1, S, D)
# and compute gradients w.r.t. h_mid expanded to (D, 1, S, D)
# but autograd.grad doesn't support batched inputs natively.

# Actually, we CAN use torch.func:
print("\n--- Method 3: torch.func.vmap + torch.func.grad ---")

from torch.func import vmap, grad

# Redo forward
hooked3 = {}
def hook_fn3(module, inputs, output):
    if isinstance(output, tuple):
        hooked3["h"] = output[0]
    else:
        hooked3["h"] = output

hook3 = model.gpt_neox.layers[6].register_forward_hook(hook_fn3)
outputs3 = model(**tokens, output_hidden_states=True)
h_final3 = outputs3.hidden_states[-1]  # (1, S, D)
h_mid3 = hooked3["h"]  # (1, S, D)
hook3.remove()

# The function we want to differentiate: 
# given h_mid[0, last_pos] → h_final[0, last_pos]
# But h_final depends on the FULL h_mid (because of attention to other positions)
# 
# For the last position specifically, ∂(h_final[0,last_pos,d']) / ∂(h_mid[0,j,k])
# is non-zero only for j ≤ last_pos due to causal masking.
#
# Let's compute the full D×D Jacobian at the last position:
# J[d', k] = ∂(h_final[0,last_pos,d']) / ∂(h_mid[0,last_pos,k])

# We can't easily use torch.func because the model forward is stateful
# and involves attention. Let me try a different approach.

# --- Method 4: Double backward trick ---
# The most efficient method: compute J^T J directly without forming J.
# We need J^T J for spectra, not J itself.
# 
# For a matrix J (D×D), the Gram matrix G = J^T J can be computed via:
#   G v = J^T (J v) for any v
# We can use this for randomized SVD.

# But first, let me measure if the batched vjp actually works.
# The autograd.grad docs say grad_outputs should match outputs shape.
# But we could reshape h_final to treat it as D separate scalar outputs?

print("\n--- Method 4: Batched grad_output (reshape trick) ---")

hook4 = model.gpt_neox.layers[6].register_forward_hook(
    lambda m, i, o: hooked3.update({"h4": o[0] if isinstance(o, tuple) else o})
)
outputs4 = model(**tokens, output_hidden_states=True)
h_final4 = outputs4.hidden_states[-1]  # (1, S, D)
h_mid4 = hooked3["h4"]  # (1, S, D)
hook4_target = None
for h in [hook4]:
    h.remove()

# The key insight: autograd.grad with a matrix grad_output
# treats each output dimension separately.
# If outputs has shape (1, S, D) and grad_outputs has shape (1, S, D),
# the gradient is the sum of element-wise products.
# 
# But we need PER-DIMENSION gradients, not the sum.
# So we need D separate calls (or vmap).

# Let me try: create grad_output as (D, 1, S, D) and also expand h_final
# as (D, 1, S, D) so each "batch element" is a separate vjp.

# Actually this is exactly what torch.func.vmap does.
# Let me try a simpler approach: compute J^T J directly via double backward.

print("\n--- Testing J^T J computation via double vjp ---")
print("This is the key optimization: we don't need J itself, just J^T J and J mean.")

# J: (D, D) mapping h_mid[last_pos] → h_final[last_pos]
# G = J^T J: (D, D)

# For any vector v (D,), G v = J^T (J v)
# We compute J v via vjp: g = ∂((h_final[last_pos] ⋅ (J v)))/∂h_mid[last_pos]
# Wait that's circular. Let me think again.

# J v is the directional derivative of h_final[last_pos] in direction v
# J v = ∂(h_final[last_pos]) / ∂(h_mid[last_pos]) ⋅ v
# This equals: d/dt (h_final[last_pos](h_mid[last_pos] + t*v)) at t=0
# Which is just the standard vjp!

# So: given v, compute u = vjp(h_final[last_pos], h_mid[last_pos], v)
# Then: J^T u = vjp(h_final[last_pos], h_mid[last_pos])^T ⋅ u
# But the VJP operator is J^T, so: J^T u = vjp(h_final[last_pos], h_mid[last_pos], u) 
# Wait no. The vjp gives J^T times the cotangent.
# v = J^T w where w is the cotangent for h_final and v is for h_mid.
# So J^T w = vjp.
# Therefore J v = the *forward* sensitivity, which equals...
# The JVP (Jacobian-vector product)!

# JVP: given tangent t for h_mid, compute pushforward to h_final
# J t = jvp(h_final[last_pos], h_mid[last_pos], t)

# VJP: given cotangent w for h_final, compute pullback to h_mid
# J^T w = vjp(h_final[last_pos], h_mid[last_pos], w)

# So: G = J^T J. For any vector v:
#   G v = J^T (J v) = vjp(jvp(v))
# This means: forward-mode differentiate then reverse-mode pull back.

print("Computing G = J^T J via JVP + VJP (double autodiff)...")

# Do one more forward pass
hook5 = model.gpt_neox.layers[6].register_forward_hook(
    lambda m, i, o: hooked3.update({"h5": o[0] if isinstance(o, tuple) else o})
)
outputs5 = model(**tokens, output_hidden_states=True)
h_final5 = outputs5.hidden_states[-1]  # (1, S, D)
h_mid5 = hooked3["h5"]  # (1, S, D)
hook5.remove()

# Extract last-position vectors
hf = h_final5[0, last_pos, :]  # (D,)
hm = h_mid5[0, last_pos, :]    # (D,)

# Test: compute one column of G using double backward
v = torch.randn(D, device=hf.device, dtype=hf.dtype)
v = v / v.norm()

# JVP: compute J v
# We need: d(h_final[last_pos]) / d(h_mid) ⋅ v
# Using torch.autograd.forward_ad:
with torch.autograd.forward_ad.dual_level():
    hm_dual = torch.autograd.forward_ad.make_dual(hm, v)
    # h_mid is NOT an input at this point. We'd need to re-run the model.
    pass

print("JVP requires forward-mode AD which needs model re-execution.")
print("This is complex for a full transformer model.")
print()
print("--- Conclusion ---")
print("The vmap approach (Method 3) is the most promising:")
print("  vmap(grad(sum_{d}(h_final[0,last_pos,d]), argnums=0))(h_mid)")
print("But this requires encapsulating the 'rest of the model' as a pure function.")
print()
print("For now, the sequential approach works. Let's benchmark per-row cost.")
print(f"Per-row time: {t_seq/N_TEST*1000:.1f}ms")
print(f"Full Jacobian (768 rows): {t_seq/N_TEST*D:.1f}s per layer")
print(f"12 layers: {t_seq/N_TEST*D*12:.1f}s per prompt")
print(f"1000 prompts: {t_seq/N_TEST*D*12*1000/3600:.1f} hours per checkpoint")
print(f"153 checkpoints: {t_seq/N_TEST*D*12*1000*153/3600:.1f} hours total")
