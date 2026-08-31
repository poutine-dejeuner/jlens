"""Test torch.compile for vjp acceleration."""
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

prompt = "The capital of France is Paris, and the capital of Germany is Berlin."
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
S = tokens.input_ids.shape[1]
last_pos = S - 1

hooked = {}
def hook_fn(module, inputs, output):
    hooked["h"] = output[0] if isinstance(output, tuple) else output

hook = model.gpt_neox.layers[6].register_forward_hook(hook_fn)
outputs = model(**tokens, output_hidden_states=True)
h_final6 = outputs.hidden_states[-1]
h_mid6 = hooked["h"]
hook.remove()

print(f"\nCurrent approach: {D} sequential autograd.grad calls per layer")
print(f"Each call backprops through ~6 layers of the model\n")

# Can we batch? Instead of one-hot grad_outputs, use a matrix
# grad_output of shape (D, 1, S, D) where grad_output[d,:,:,:] is 
# the d-th standard basis vector?
#
# Actually: autograd.grad can handle this! We can reshape h_final
# to expose the D dimension as a batch dimension and use batched grad.

print("Test: can we batch vjp calls using expanded h_final?")

# New forward
hooked2 = {}
def hook_fn2(module, inputs, output):
    hooked2["h"] = output[0] if isinstance(output, tuple) else output
hook2 = model.gpt_neox.layers[6].register_forward_hook(hook_fn2)
outputs2 = model(**tokens, output_hidden_states=True)
h_final = outputs2.hidden_states[-1]  # (1, S, D)
h_mid = hooked2["h"]  # (1, S, D)
hook2.remove()

# We want: for each d, ∂(h_final[0,last_pos,d]) / ∂(h_mid[0,last_pos,:])
# grad_outputs should have the same shape as outputs: (1, S, D)
# We want individual gradients, not the sum.

# Trick: expand h_final along a new batch dimension
# h_final_expanded: (D, 1, S, D) where batch d is the original
# But then h_mid must also be batch d... 
# The issue is that h_final[0,:,d] depends on the ENTIRE h_mid, not per-batch.
# We'd need D separate copies of the model, which is impractical.

# Alternative: just use grad_output with the full identity matrix?
# grad_output shape (D, 1, S, D): for batch d, grad_output[d,0,last_pos,d]=1
# But then autograd.grad sums over the output dimensions...

# Let me test: what if we make grad_output have extra first dim?
print("\nAttempting batched grad_output...")
try:
    grad_output = torch.zeros(10, 1, S, D, device=h_final.device, dtype=h_final.dtype)
    for d in range(10):
        grad_output[d, 0, last_pos, d] = 1.0
    
    # Need to also expand h_final or the outputs to match
    h_final_exp = h_final.unsqueeze(0).expand(10, 1, S, D)  # not in graph
    # This won't work - the expanded tensor isn't in the autograd graph
    pass
except Exception as e:
    print(f"  Not attempted: {e}")

# Let me try the simplest acceleration: use the fact that we only need
# the Gram matrix, and Gram = Σ rows^T rows can be computed via
# randomized power iteration without forming J explicitly.

print("\n\n--- Randomized eigenvalue computation ---")
print("Goal: compute eigenvalues of J^T J without forming J.")
print()
print("For a D×D Gram matrix G = J^T J:")
print("  1. Draw random matrix Ω of size D×(k+p) where k = desired rank, p = oversampling")
print("  2. Compute Y = J Ω via k+p JVPs (directional derivatives)")
print("  3. QR factorize: Y = Q R")
print("  4. Compute B = Q^T G Q via: (J Q)^T (J Q) using VJPs")
print("  5. Eigendecompose B (small, (k+p)×(k+p))")
print("  6. Eigenvalues of B ≈ top eigenvalues of G")
print()
print("But step 2 requires JVP (forward-mode AD), which needs:")
print("  - Per-tangent: one forward pass through remaining layers")
print("  - For k+p ~= 50 tangents: 50 forward passes")
print("  - Each forward: ~50ms → 2.5s total for JVPs")
print("  - Plus VJPs: 50 vjp calls → ~0.25s")
print("  - Total per layer: ~3s (vs 7.6s sequential)")
print("  - 12 layers × 3s = 36s/prompt (vs 91s)")
print()
print("BUT: JVP in PyTorch requires torch.autograd.forward_ad which has")
print("limited support. Let me check if it works for transformer models...")

# Test JVP
print("\nTesting torch.autograd.forward_ad with a simple model...")

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(768, 256)
        self.linear2 = torch.nn.Linear(256, 128)
    
    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

simple = SimpleNet().cuda().half()
x = torch.randn(1, 768, device="cuda", dtype=torch.float16)

# Test JVP
tangent = torch.randn_like(x)
with torch.autograd.forward_ad.dual_level():
    dual_x = torch.autograd.forward_ad.make_dual(x, tangent)
    dual_y = simple(dual_x)
    _, jvp_result = torch.autograd.forward_ad.unpack_dual(dual_y)
    print(f"  Simple net JVP works! Shape: {jvp_result.shape}")

# Test with the actual model
print("\nTesting JVP with transformer layer...")
hooked3 = {}
def hook_fn3(module, inputs, output):
    hooked3["h"] = output[0] if isinstance(output, tuple) else output
hook3 = model.gpt_neox.layers[6].register_forward_hook(hook_fn3)
outputs3 = model(**tokens, output_hidden_states=True)
h_mid3 = hooked3["h"]  # (1, S, D)
hook3.remove()

# We need forward_ad to propagate through the remaining layers.
# Let me test if forward_ad works with a transformer block
# by re-running just one attention layer through forward_ad

print("  Testing single attention layer JVP...")
layer7 = model.gpt_neox.layers[7]
# Forward the hidden state through layer 7 with dual numbers
x_in = h_mid3[0, last_pos:last_pos+1, :]  # (1, D) - take last position
tangent = torch.randn_like(x_in)
try:
    with torch.autograd.forward_ad.dual_level():
        dual_x = torch.autograd.forward_ad.make_dual(x_in, tangent)
        # layer7 expects (batch, seq, hidden) and may need attention_mask
        # This is getting complex...
        print("  Skipping - need full input preparation")
except Exception as e:
    print(f"  Error: {e}")
