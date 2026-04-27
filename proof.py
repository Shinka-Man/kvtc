#!/usr/bin/env python3
"""A/B benchmark for baseline vLLM vs KVTC-enabled vLLM."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Explain in detail how transformer attention works, including grouped-query attention, "
    "RoPE positional encodings, KV cache growth during autoregressive decoding, the main causes "
    "of GPU memory pressure at long context, and why PCA-based KV cache compression can reduce "
    "serving cost while preserving output quality. Include equations where useful."
)


@dataclass
class MemorySample:
    timestamp: float
    used_mib: float


@dataclass
class MemoryPoller:
    """Poll GPU memory usage while a benchmark run is in flight."""

    poll_ms: int = 50
    samples: list[MemorySample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def summary(self, free_timestamp: float | None = None) -> dict[str, float | None]:
        if not self.samples:
            return {
                "memory_initial_mib": None,
                "memory_final_mib": None,
                "memory_peak_mib": None,
                "memory_tail_avg_mib": None,
                "memory_post_free_avg_mib": None,
            }
        used = [sample.used_mib for sample in self.samples]
        tail_start = max(int(len(used) * 0.7), 0)
        tail_values = used[tail_start:] or used
        post_free_values = []
        if free_timestamp is not None:
            post_free_values = [sample.used_mib for sample in self.samples if sample.timestamp >= free_timestamp]
        return {
            "memory_initial_mib": used[0],
            "memory_final_mib": used[-1],
            "memory_peak_mib": max(used),
            "memory_tail_avg_mib": statistics.mean(tail_values),
            "memory_post_free_avg_mib": statistics.mean(post_free_values) if post_free_values else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            value = query_gpu_memory_mib()
            if value is not None:
                self.samples.append(MemorySample(timestamp=time.perf_counter(), used_mib=value))
            self._stop.wait(self.poll_ms / 1000.0)


def query_gpu_memory_mib() -> float | None:
    """Read current GPU memory usage via nvidia-smi."""

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    if not first_line:
        return None
    try:
        return float(first_line.strip())
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark baseline vLLM against KVTC-enabled vLLM.")
    parser.add_argument("--mode", choices=["compare", "worker"], default="compare")
    parser.add_argument("--backend", choices=["baseline", "kvtc"], default="baseline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--head-group-size", type=int, default=1)
    parser.add_argument("--bit-budget-ratio", type=float, default=0.25)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--calibration-path", default="kvtc_vllm_calibration.pt")
    parser.add_argument("--poll-ms", type=int, default=50)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-triton", action="store_true")
    parser.add_argument("--auto-activate", action="store_true",
                        help="Activate KVTC decode path after prefill (default: passthrough).")
    parser.add_argument("--language-model-only", action="store_true",
                        help="For VLM-class models (e.g. Qwen3_5ForConditionalGeneration), "
                             "skip the multimodal preprocessor.")
    parser.add_argument("--quantization", default=None,
                        help="vLLM quantization arg (e.g. modelopt for NVFP4).")
    parser.add_argument("--speculative-config", default=None,
                        help="vLLM speculative-config JSON (e.g. for MTP draft head).")
    parser.add_argument("--output-json", default="")
    return parser


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - only hit outside vLLM environments.
        raise RuntimeError("vLLM is not installed in this environment.") from exc

    from kvtc.calibrate_vllm import DEFAULT_WARMUP_PROMPTS, VLLMCalibrationCollector, calibrate_vllm_model
    from kvtc.vllm_backend import hook_engine, free_kv_cache_engine

    class _EngineHookProxy:
        """Shim mimicking KVTCHandle for the proof.py reporting code (vLLM 0.19+ MP variant)."""
        def __init__(self, info):
            self.info = info or {}
            self.active = bool(self.info.get("auto_activate", False))
            self.free_timestamp = None
            self.num_layers = int(self.info.get("num_layers", 0))

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        async_scheduling=False,
        enable_prefix_caching=False,
    )
    if getattr(args, "language_model_only", False):
        llm_kwargs["language_model_only"] = True
    if getattr(args, "quantization", None):
        llm_kwargs["quantization"] = args.quantization
    if getattr(args, "speculative_config", None):
        llm_kwargs["speculative_config"] = json.loads(args.speculative_config)
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    hook = None
    calibration_entries = 0
    if args.backend == "kvtc":
        calibration_path = Path(args.calibration_path)
        if calibration_path.exists():
            calibration = VLLMCalibrationCollector.load(calibration_path)
        else:
            calibration = calibrate_vllm_model(
                llm,
                DEFAULT_WARMUP_PROMPTS,
                bit_budget_ratio=args.bit_budget_ratio,
                head_group_size=args.head_group_size,
                rope_theta=args.rope_theta,
                save_path=calibration_path,
            )
        calibration.sink_tokens = args.sink_tokens
        calibration.window_tokens = args.window_tokens
        calibration_entries = len(calibration.entries)
        # auto_activate disabled by default; pass --auto-activate to enable KVTC decode.
        auto_act = bool(getattr(args, "auto_activate", False))
        hook_info = hook_engine(llm, calibration, auto_activate=auto_act, use_triton=not args.no_triton)
        print(f"[KVTC HOOK INSTALL] {hook_info}", flush=True)
        hook = _EngineHookProxy(hook_info)

    llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)

    poller = MemoryPoller(poll_ms=args.poll_ms)
    poller.start()
    started_at = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params, use_tqdm=False)
    finished_at = time.perf_counter()
    poller.stop()

    generation = outputs[0].outputs[0]
    token_ids = list(getattr(generation, "token_ids", []))
    text = generation.text
    duration_s = finished_at - started_at
    memory_stats = poller.summary(free_timestamp=None if hook is None else hook.free_timestamp)

    result = {
        "backend": args.backend,
        "model": args.model,
        "prompt_tokens_estimate": None,
        "generated_tokens": len(token_ids),
        "text": text,
        "token_ids": token_ids,
        "duration_ms": duration_s * 1000.0,
        "tokens_per_second": len(token_ids) / duration_s if duration_s > 0 else None,
        "auto_activated": bool(hook and hook.active),
        "free_timestamp": None if hook is None else hook.free_timestamp,
        "calibration_entries": calibration_entries,
        **memory_stats,
    }
    return result


def _shared_worker_args(args: argparse.Namespace) -> list[str]:
    return [
        "--mode",
        "worker",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--head-group-size",
        str(args.head_group_size),
        "--bit-budget-ratio",
        str(args.bit_budget_ratio),
        "--rope-theta",
        str(args.rope_theta),
        "--sink-tokens",
        str(args.sink_tokens),
        "--window-tokens",
        str(args.window_tokens),
        "--calibration-path",
        args.calibration_path,
        "--poll-ms",
        str(args.poll_ms),
    ]


def _run_subprocess(script_path: Path, args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [sys.executable, str(script_path), *_shared_worker_args(args), "--backend", backend]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.no_triton:
        command.append("--no-triton")

    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} run failed with exit code {completed.returncode}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"Did not find RESULT_JSON output for {backend} run.\n{completed.stdout}")


def _token_match_ratio(left: list[int], right: list[int]) -> float:
    if not left and not right:
        return 1.0
    total = max(len(left), len(right), 1)
    matches = sum(1 for lhs, rhs in zip(left, right) if lhs == rhs)
    return matches / total


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    baseline = _run_subprocess(script_path, args, "baseline")
    kvtc = _run_subprocess(script_path, args, "kvtc")

    comparison = {
        "baseline": baseline,
        "kvtc": kvtc,
        "text_exact_match": baseline["text"] == kvtc["text"],
        "token_match_ratio": _token_match_ratio(baseline["token_ids"], kvtc["token_ids"]),
        "latency_ratio": (
            kvtc["duration_ms"] / baseline["duration_ms"]
            if baseline["duration_ms"]
            else None
        ),
        "tail_vram_delta_mib": (
            None
            if baseline["memory_tail_avg_mib"] is None or kvtc["memory_tail_avg_mib"] is None
            else baseline["memory_tail_avg_mib"] - kvtc["memory_tail_avg_mib"]
        ),
        "post_free_vram_delta_mib": (
            None
            if baseline["memory_tail_avg_mib"] is None or kvtc["memory_post_free_avg_mib"] is None
            else baseline["memory_tail_avg_mib"] - kvtc["memory_post_free_avg_mib"]
        ),
    }

    print(f"Model: {args.model}")
    print(f"Prompt length: {len(args.prompt.split())} words")
    print(f"Baseline latency: {baseline['duration_ms']:.1f} ms")
    print(f"KVTC latency: {kvtc['duration_ms']:.1f} ms")
    if comparison["latency_ratio"] is not None:
        print(f"Latency ratio (KVTC / baseline): {comparison['latency_ratio']:.2f}x")
    print(f"Exact output match: {comparison['text_exact_match']}")
    print(f"Token match ratio: {comparison['token_match_ratio']:.3f}")
    if comparison["tail_vram_delta_mib"] is not None:
        print(f"Tail VRAM freed: {comparison['tail_vram_delta_mib']:.1f} MiB")
    if comparison["post_free_vram_delta_mib"] is not None:
        print(f"Post-free VRAM freed: {comparison['post_free_vram_delta_mib']:.1f} MiB")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.output_json}")
    return comparison


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "worker":
        result = run_worker(args)
        print("RESULT_JSON=" + json.dumps(result))
        return

    comparison = run_compare(args)
    print("RESULT_JSON=" + json.dumps(comparison))


if __name__ == "__main__":
    main()
