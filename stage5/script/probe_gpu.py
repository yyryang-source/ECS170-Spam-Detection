#!/usr/bin/env python3
"""Probe optional GPU training in a child process and save factual evidence.

A backend can report available while its first backward pass aborts in native
code. Keeping that attempt in a subprocess protects the main CPU experiments.
The script exits successfully when it writes a report, even if the optional
GPU check fails; inspect the report's ``status`` and ``passed`` fields.

Example (from the project root)::

    python script/probe_gpu.py --device mps --output results/gpu_probe.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHILD_PROGRAM = r'''
import json
import sys
import numpy as np
import scipy.sparse as sp
import torch
from local_code.stage5.model import GCN, GraphOperator

device = sys.argv[1]
torch.manual_seed(42)
graph = np.array([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]], dtype=np.float32)
inv = np.diag(graph.sum(1) ** -0.5)
operator = GraphOperator.from_scipy(sp.coo_matrix(inv @ graph @ inv), device=device)
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)
y = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
model = GCN(2, 4, 2, dropout=0.).to(device)
print(json.dumps({'event': 'initialized', 'device': device, 'backend': operator.backend}), flush=True)
logits = model(x, operator)
loss = torch.nn.functional.cross_entropy(logits, y)
if device == 'mps':
    torch.mps.synchronize()
else:
    torch.cuda.synchronize()
print(json.dumps({'event': 'forward_complete', 'loss': float(loss.detach().cpu())}), flush=True)
loss.backward()
if device == 'mps':
    torch.mps.synchronize()
else:
    torch.cuda.synchronize()
finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all().cpu()) for p in model.parameters())
print(json.dumps({'event': 'backward_complete', 'finite_gradients': finite}), flush=True)
if not finite:
    raise SystemExit(2)
'''


def probe(device: str = "auto", timeout: float = 30.) -> dict:
    capabilities = {
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    selected = device
    if selected == "auto":
        selected = "cuda" if capabilities["cuda_available"] else "mps"
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Optional Stage 5-6 GPU forward/backward smoke test",
        "test": "Three-node normalized graph; GCN(2, 4, 2); cross-entropy backward",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "capabilities": capabilities,
        "requested_device": device,
        "selected_device": selected,
        "timeout_seconds": timeout,
        "passed": False,
    }
    if not capabilities[f"{selected}_available"]:
        report.update(status="unavailable", elapsed_seconds=0.,
                      reason=f"PyTorch reports {selected} unavailable")
        return report
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD_PROGRAM, selected],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        report.update(status="timeout", elapsed_seconds=round(time.perf_counter() - start, 3),
                      stdout=decoded(exc.stdout), stderr=decoded(exc.stderr),
                      reason="Child was killed after the configured timeout")
        return report
    report.update(
        elapsed_seconds=round(time.perf_counter() - start, 3),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    events = []
    for line in completed.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    report["events"] = events
    if completed.returncode < 0:
        report["status"] = "native_crash"
        report["termination_signal"] = signal.Signals(-completed.returncode).name
        report["exception"] = next((line for line in completed.stderr.splitlines() if line.strip()), "")
    elif completed.returncode == 0 and any(
        event.get("event") == "backward_complete" and event.get("finite_gradients")
        for event in events
    ):
        report.update(status="passed", passed=True)
    else:
        report["status"] = "failed"
    report["experiment_policy"] = (
        "GPU smoke test passed; this does not constitute a full dataset training run."
        if report["passed"] else
        "Use CPU for the reported experiments; no successful GPU training result is claimed."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cuda"), default="auto")
    parser.add_argument("--timeout", type=float, default=30.)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "gpu_probe.json")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = probe(args.device, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "device": result["selected_device"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
