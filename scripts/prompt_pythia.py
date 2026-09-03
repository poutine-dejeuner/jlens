#!/usr/bin/env python3
"""Prompt Pythia models via vLLM OpenAI-compatible API.

Can run standalone (starts vLLM server + queries) or as a client
talking to an already-running server.

Usage:
    # Start server + run prompts (all-in-one via SLURM):
    python scripts/prompt_pythia.py --model EleutherAI/pythia-160m-deduped

    # Client-only (server already running):
    python scripts/prompt_pythia.py --base-url http://localhost:8000/v1

    # Batch from a file of prompts:
    python scripts/prompt_pythia.py --prompt-file prompts.txt --max-tokens 256

Environment:
    VLLM_PYTHON: path to vllm venv python (default: ~/vllm-env/bin/python)
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ── client-side imports (lightweight) ──────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def load_prompts(source: Path, n: int | None = None) -> list[str]:
    """Load prompts from a text file (one per line) or generate simple ones."""
    if source.exists():
        with open(source) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if n:
            lines = lines[:n]
        return lines
    # Fallback: generate generic prompts
    defaults = [
        "Explain the concept of gradient descent in machine learning.",
        "Write a short poem about artificial intelligence.",
        "What are the key differences between Python and JavaScript?",
        "Describe the transformer architecture in simple terms.",
        "Summarize the history of the Linux operating system.",
    ]
    return defaults[:n] if n else defaults


# ── vLLM server launcher ───────────────────────────────────────────────────

VLLM_VENV = Path(os.environ.get("VLLM_PYTHON", Path.home() / "vllm-env" / "bin" / "python"))


def start_vllm_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
) -> subprocess.Popen:
    """Launch vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        str(VLLM_VENV), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]

    print(f"[vLLM] launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


def wait_for_server(base_url: str, timeout: float = 300.0) -> bool:
    """Poll /v1/models until vLLM responds."""
    import urllib.request

    deadline = time.time() + timeout
    url = f"{base_url}/models"
    print(f"[client] waiting for server at {base_url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(" ready!")
            return True
        except Exception:
            time.sleep(2)
            print(".", end="", flush=True)
    print(" TIMEOUT")
    return False


# ── client ─────────────────────────────────────────────────────────────────

def run_prompts(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    system: str | None = None,
) -> list[dict]:
    """Send prompts to vLLM and collect responses."""
    if OpenAI is None:
        print("[client] openai package not available, using raw HTTP")
        return _run_prompts_raw(base_url, model, prompts, max_tokens, temperature, system)

    client = OpenAI(base_url=base_url, api_key="not-needed")
    results = []
    try:
        models = client.models.list()
        available = [m.id for m in models]
        print(f"[client] available models: {available}")
    except Exception:
        pass

    for i, prompt in enumerate(prompts):
        t0 = time.time()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content
        usage = resp.usage
        results.append({
            "idx": i,
            "prompt": prompt,
            "completion": text,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            },
            "elapsed_s": round(elapsed, 2),
        })
        print(f"[{i+1}/{len(prompts)}] {usage.prompt_tokens if usage else '?'}→"
              f"{usage.completion_tokens if usage else '?'} tok, "
              f"{elapsed:.1f}s | {text[:80].strip()!r}...")

    return results


def _run_prompts_raw(base_url, model, prompts, max_tokens, temperature, system):
    """Fallback: raw requests without openai package."""
    import urllib.request

    results = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        elapsed = time.time() - t0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        results.append({
            "idx": i,
            "prompt": prompt,
            "completion": text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
            "elapsed_s": round(elapsed, 2),
        })
        print(f"[{i+1}/{len(prompts)}] {usage.get('prompt_tokens','?')}→"
              f"{usage.get('completion_tokens','?')} tok, "
              f"{elapsed:.1f}s | {text[:80].strip()!r}...")

    return results


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Pythia models via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # All-in-one: start vLLM + query (needs SLURM with GPU)
          python scripts/prompt_pythia.py --model EleutherAI/pythia-160m-deduped --launch

          # Client-only (connect to running server)
          python scripts/prompt_pythia.py --base-url http://node-xx:8000/v1

          # Custom prompts
          python scripts/prompt_pythia.py --prompts "What is 2+2?" "Write haiku about GPUs"
        """),
    )
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped",
                        help="Model name or path")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--launch", action="store_true",
                        help="Launch vLLM server before querying")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-file", type=Path,
                        help="File with one prompt per line")
    parser.add_argument("--prompts", nargs="*",
                        help="Prompt strings (inline)")
    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of default prompts if none provided")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system", type=str, default=None,
                        help="System message")
    parser.add_argument("--output", type=Path,
                        help="Save results to JSON file")
    parser.add_argument("--wait-timeout", type=float, default=300.0,
                        help="Seconds to wait for server")

    args = parser.parse_args()

    # Gather prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        prompts = load_prompts(args.prompt_file, args.num_prompts)
    else:
        prompts = load_prompts(Path("/dev/null"), args.num_prompts)

    print(f"[client] {len(prompts)} prompts loaded")

    # Optionally launch server
    server_proc = None
    if args.launch:
        if not VLLM_VENV.exists():
            print(f"[error] vLLM venv not found at {VLLM_VENV}")
            print(f"  Set VLLM_PYTHON env var or create ~/vllm-env")
            sys.exit(1)
        server_proc = start_vllm_server(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        if not wait_for_server(args.base_url, timeout=args.wait_timeout):
            server_proc.kill()
            sys.exit(1)

    try:
        results = run_prompts(
            base_url=args.base_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system=args.system,
        )
    finally:
        if server_proc:
            server_proc.kill()
            server_proc.wait()

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[client] results saved to {args.output}")

    # Summary
    total_prompt_tok = sum(r["usage"]["prompt_tokens"] or 0 for r in results)
    total_completion_tok = sum(r["usage"]["completion_tokens"] or 0 for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"\n[summary] {len(results)} prompts | "
          f"{total_prompt_tok} prompt tok | {total_completion_tok} completion tok | "
          f"{total_time:.1f}s total | {total_time/len(results):.1f}s avg")


if __name__ == "__main__":
    main()
