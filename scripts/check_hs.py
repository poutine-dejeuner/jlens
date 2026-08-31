"""Check hidden state indexing for Pythia (GPT-NeoX) model."""
import os
os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import triton; triton.knobs.build.impl = "dummy"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-160m-deduped"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision="step143000", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision="step143000", 
    torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
model.eval()
L = len(model.gpt_neox.layers)
print(f"Layers: {L}")

tokens = tokenizer("Hello world test", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**tokens, output_hidden_states=True)
hs = out.hidden_states
print(f"Hidden states: {len(hs)} (expected {L+2}: embed + {L} blocks + final_ln)")
for i, h in enumerate(hs):
    print(f"  hs[{i}]: {list(h.shape)}")
print()
print("Differences between consecutive hidden states (should be non-zero if they change):")
for i in range(len(hs)-1):
    diff = (hs[i+1] - hs[i]).norm().item()
    print(f"  Δ[{i}→{i+1}] = {diff:.6f}")
print()
# Check: hs[0] should be embedding output (before any layer)
# hs[1..L] should be layer 0..L-1 outputs
# hs[L+1] should be final layer norm output
# If hs[L] == hs[L+1] then final_ln is included in the last block output
print(f"hs[{L}] (last block) vs hs[{L+1}] (should be final_ln):")
print(f"  diff: {(hs[L] - hs[L+1]).norm().item():.6f}")
# Also check: does hs[L] == hs[L-1] + residual? (i.e. does last block output include final_ln?)
# Pythia-160m doesn't have a separate final_ln block - it's applied in the model wrapper
