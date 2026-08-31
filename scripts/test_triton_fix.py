"""Test if we can disable Triton JIT compilation."""
import os
os.environ["TRITON_BUILD_WITHOUT_CC"] = "1"

# Try setting triton knob
try:
    import triton
    # Check what knobs exist
    print(f"Triton version: {triton.__version__}")
    # Try to set the build impl to use no compilation
    if hasattr(triton.knobs, "build"):
        triton.knobs.build.impl = "dummy"
        print("Set triton.knobs.build.impl = 'dummy'")
except Exception as e:
    print(f"Triton knob set error: {e}")

# Also set TORCH_COMPILE_DISABLE 
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Test a simple autograd operation 
x = torch.randn(10, 10, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
y = x.sum()
y.backward()
print("Basic autograd: OK")

# Test a slightly more complex hook-based operation 
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
if torch.cuda.is_available():
    model = model.cuda()
    x = torch.randn(1, 10, device="cuda")
else:
    x = torch.randn(1, 10)

hooked = {}
def hook(module, inputs, output):
    output = output.detach()
    output.requires_grad_(True)
    hooked["out"] = output
    return output

model.linear.register_forward_hook(hook)
out = model(x)
print(f"Hook output shape: {hooked['out'].shape}")

# Backprop through the hook
loss = out.sum()
loss.backward()
print("Hook-based autograd: OK")

print("\nAll tests passed!")
