"""Quick validation script for Jacobian computation approach.

Tests:
1. Can we load Pythia-160M on B300?
2. Can we hook residual stream states?
3. What's the fastest way to compute ∂h_final/∂h_ℓ?
   a) torch.autograd.functional.jacobian (built-in)
   b) Sequential vjp per output dimension
   c) Single backward pass with grad_outputs = identity
4. How much memory does each approach use?
5. How long does one prompt take?

Run via: uv run python scripts/validate_jacobian.py
"""

import time
import torch
import gc

MODEL_ID = "EleutherAI/pythia-160m-deduped"
STEP = 143000  # final checkpoint
PROMPT = "The capital of France is Paris. The capital of Germany is Berlin."


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print(f"\nLoading {MODEL_ID} @ step{STEP}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=f"step{STEP}", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=f"step{STEP}",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Architecture
    n_layers = len(model.gpt_neox.layers)
    d_model = model.config.hidden_size
    print(f"Layers: {n_layers}, d_model: {d_model}")

    # Tokenize prompt
    tokens = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    S = input_ids.shape[1]
    print(f"Prompt length: {S} tokens")

    # --- Test 1: Basic forward pass with hidden state capture ---
    print("\n--- Test 1: Forward pass with hidden states ---")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    print(f"Forward pass: {time.time() - t0:.3f}s")
    print(f"Hidden states: {len(outputs.hidden_states)} layers")
    for i, h in enumerate(outputs.hidden_states):
        print(f"  L{i}: shape={list(h.shape)}, norm={h.norm().item():.2f}")

    # --- Test 2: Compute ∂h_final/∂h_layer using torch.autograd.functional.jacobian ---
    print("\n--- Test 2: torch.autograd.functional.jacobian ---")

    # We need a function that takes h_ℓ and returns h_final
    # But the model isn't designed for this. Instead, we can:
    # - Capture h_ℓ via hook
    # - Recomputed h_final from h_ℓ by running remaining layers

    # Simpler: compute Jacobian of h_final w.r.t input embeddings
    # (this validates the autodiff approach)

    def get_final_hidden(input_ids, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1][0, -1, :]  # last position only (D,)

    try:
        t0 = time.time()
        jac = torch.autograd.functional.jacobian(
            lambda ids: get_final_hidden(ids, attention_mask),
            input_ids,
            vectorize=True,  # use vmap for efficiency
        )
        print(f"  ∂h_final/∂input_ids via vectorized jacobian: {time.time() - t0:.1f}s")
        print(f"  Shape: {list(jac.shape)}")
        # jac shape: (D, 1, S, V) = (768, 1, S, vocab_size) — huge!

        del jac
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  vectorize=True failed: {e}")
        print("  Trying without vectorize...")
        try:
            t0 = time.time()
            jac = torch.autograd.functional.jacobian(
                lambda ids: get_final_hidden(ids, attention_mask),
                input_ids,
                vectorize=False,
            )
            print(f"  ∂h_final/∂input_ids: {time.time() - t0:.1f}s")
            del jac
        except Exception as e2:
            print(f"  non-vectorized also failed: {e2}")

    gc.collect()
    torch.cuda.empty_cache()

    # --- Test 3: Hook intermediate hidden states, compute Jacobian via vjp ---
    print("\n--- Test 3: Per-layer Jacobian via hook + sequential vjp ---")

    target_layer = n_layers // 2  # middle layer
    hidden_state = {}

    def hook_fn(module, inputs, output):
        # output is the residual stream after this layer: (1, S, D)
        hidden_state["h"] = output.detach()
        hidden_state["h"].requires_grad_(True)

    hook = model.gpt_neox.layers[target_layer].register_forward_hook(hook_fn)

    try:
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_final = outputs.hidden_states[-1]  # (1, S, D)
        h_ell = hidden_state["h"]  # (1, S, D)

        print(f"  h_final shape: {list(h_final.shape)}")
        print(f"  h_{target_layer} shape: {list(h_ell.shape)}")

        # Sequential vjp: compute one row of the Jacobian at a time
        # Jacobian: ∂(h_final)[0, -1, :] / ∂(h_ell)[0, -1, :]  (D × D)
        # For the last position only
        D = d_model
        jac = torch.zeros(D, D, device=h_ell.device, dtype=h_ell.dtype)

        t0 = time.time()
        for d in range(D):
            grad_output = torch.zeros_like(h_final)
            grad_output[0, -1, d] = 1.0  # only the d-th dim of last position

            grads = torch.autograd.grad(
                outputs=h_final,
                inputs=h_ell,
                grad_outputs=grad_output,
                retain_graph=(d < D - 1),
                create_graph=False,
            )[0]  # (1, S, D)

            # Gradient w.r.t. last position of h_ell
            jac[d] = grads[0, -1, :]

        elapsed = time.time() - t0
        print(f"  Sequential vjp ({D} passes): {elapsed:.1f}s ({elapsed/D*1000:.1f}ms per dim)")

        # Check Jacobian properties
        print(f"  J shape: {list(jac.shape)}")
        print(f"  ‖J‖_F: {jac.norm().item():.2f}")
        print(f"  tr(J)/d: {jac.trace().item()/D:.3f}")
        jtj = jac.T @ jac
        eigvals = torch.linalg.eigvalsh(jtj.float())
        print(f"  J⊤J eigenvalues: min={eigvals[0].item():.4f}, max={eigvals[-1].item():.2f}")

        del jac, jtj, grads
        gc.collect()
        torch.cuda.empty_cache()

    finally:
        hook.remove()

    # --- Test 4: All layers at once (single forward, multi-backward) ---
    print(f"\n--- Test 4: All {n_layers} layers via single forward pass ---")

    hidden_states = {}

    def make_hook(idx):
        def hook_fn(module, inputs, output):
            hidden_states[idx] = output.detach()
            hidden_states[idx].requires_grad_(True)
        return hook_fn

    hooks = []
    for layer_idx in range(n_layers):
        h = model.gpt_neox.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    try:
        t0 = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_final = outputs.hidden_states[-1]
        fwd_time = time.time() - t0
        print(f"  Forward pass: {fwd_time:.3f}s")

        # Now compute Jacobian for each layer
        D = d_model
        last_pos = attention_mask.sum().item() - 1

        for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            h_ell = hidden_states[layer_idx]

            t0 = time.time()

            # Compute all rows at once: vmap-style but manually
            # For efficiency, compute vjp for all D output dims in batches
            jac = torch.zeros(D, D, device=h_ell.device, dtype=h_ell.dtype)

            # Batch vjps: compute gradients for a batch of output dimensions
            batch_size = 64  # process 64 dims at a time
            for start in range(0, D, batch_size):
                end = min(start + batch_size, D)
                n_batch = end - start

                # Create grad_outputs for batch
                grad_outputs = torch.zeros(
                    1, S, D, device=h_final.device, dtype=h_final.dtype
                )
                for i, d in enumerate(range(start, end)):
                    grad_outputs[0, last_pos, d] = 1.0

                # We need separate vjp per dimension within the batch
                # torch.autograd.grad doesn't natively batch this.
                # Fall back to sequential for now (but could use functorch.vmap)
                for i, d in enumerate(range(start, end)):
                    go = torch.zeros_like(h_final)
                    go[0, last_pos, d] = 1.0
                    grads = torch.autograd.grad(
                        outputs=h_final,
                        inputs=h_ell,
                        grad_outputs=go,
                        retain_graph=(d < D - 1),
                        create_graph=False,
                    )[0]
                    jac[d] = grads[0, last_pos, :]

            bwd_time = time.time() - t0
            jtj = jac.T @ jac
            eigvals = torch.linalg.eigvalsh(jtj.float())
            print(
                f"  L{layer_idx:02d}: {bwd_time:.1f}s "
                f"tr(J)/d={jac.trace().item()/D:.3f} "
                f"λ_min={eigvals[0].item():.4f} λ_max={eigvals[-1].item():.2f}"
            )

            del jac, jtj
            gc.collect()
            torch.cuda.empty_cache()

    finally:
        for h in hooks:
            h.remove()

    # --- Test 5: Can we get J⊤J without full J using Hutchinson? ---
    print("\n--- Test 5: Hutchinson trace estimator for tr(J⊤J) ---")
    # tr(J⊤J) = E[v⊤ J⊤J v] for v ~ N(0,I)
    # Compute J⊤J v via double vjp: first vjp gives J⊤v, second gives J(J⊤v)
    # Then dot with v gives v⊤J⊤Jv = ‖Jv‖²

    target_layer = n_layers // 2
    hidden_state_test5 = {}

    def hook5(module, inputs, output):
        hidden_state_test5["h"] = output.detach()
        hidden_state_test5["h"].requires_grad_(True)

    hook5_handle = model.gpt_neox.layers[target_layer].register_forward_hook(hook5)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_final = outputs.hidden_states[-1]
        h_ell = hidden_state_test5["h"]

        last_pos = attention_mask.sum().item() - 1

        # Hutchinson: for k random probes, estimate tr(J⊤J) ≈ (1/k) Σ ‖J v_i‖²
        k_probes = 10
        estimates = []

        t0 = time.time()
        for _ in range(k_probes):
            v = torch.randn(D, device=h_ell.device, dtype=h_ell.dtype)

            # Create grad_output for position last_pos, direction v
            go = torch.zeros_like(h_final)
            go[0, last_pos, :] = v

            # J⊤v = ∂(v⊤ h_final[-1])/∂h_ell
            # This is a scalar output → single vjp
            v_dot_h = torch.dot(h_final[0, last_pos, :], v)
            grads = torch.autograd.grad(
                outputs=v_dot_h,
                inputs=h_ell,
                create_graph=False,
            )[0]  # (1, S, D)

            jtv = grads[0, last_pos, :]  # (D,)
            estimates.append(torch.dot(jtv, jtv).item())

        h_time = time.time() - t0
        tr_jtj_est = sum(estimates) / k_probes
        tr_jtj_true = torch.sum(jac**2).item()  # from Test 3
        print(f"  Hutchinson ({k_probes} probes): {h_time:.3f}s")
        print(f"  tr(J⊤J) estimate: {tr_jtj_est:.2f}")
        print(f"  tr(J⊤J) true (from Test 3): {tr_jtj_true:.2f}")
        print(f"  Error: {abs(tr_jtj_est - tr_jtj_true) / tr_jtj_true * 100:.1f}%")

    finally:
        hook5_handle.remove()

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
