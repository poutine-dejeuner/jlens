"""Test: can we do a full model forward + backward on B300?"""
import os
os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import triton
triton.knobs.build.impl = "dummy"

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

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
    torch_dtype=torch.float16,  # bf16 has RoPE issues on B300
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded. Dtype: {model.dtype}")

# Try a forward pass
prompt = "The capital of France is"
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Forward pass...")
with torch.no_grad():
    outputs = model(**tokens, output_hidden_states=True)
print(f"Forward OK, hidden states: {len(outputs.hidden_states)} layers")

# Try a backward pass with a proper hook (what triggers the Jacobian computation)
hooked = {}
def hook_fn(module, inputs, output):
    hooked["h"] = output
    # Don't detach - keep it in the graph for backward

hook = model.gpt_neox.layers[6].register_forward_hook(hook_fn)

# Forward again with hook
tokens2 = tokenizer("Another test sentence for backward.", return_tensors="pt").to(model.device)
print("Forward + hook...")
out2 = model(**tokens2)
h_final2 = out2.logits

print("Backward pass...")
loss2 = h_final2.sum()
loss2.backward()
print("Backward OK - hooks work with autograd!")

hook.remove()

print("All OK!")
