#!/usr/bin/env python3
"""Run the end-to-end experiment pipeline for the simplified project."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a subprocess and fail loudly on errors."""
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def load_checkpoint_metrics(path: Path) -> dict:
    """Read metrics from a saved trainer checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("metrics", {})


def maybe_run_train(
    python_exec: str,
    config_path: str,
    checkpoint_name: str,
    extra_flags: list[str] | None = None,
    force: bool = False,
    cuda_visible_devices: str | None = None,
) -> Path:
    """Train a model variant unless its checkpoint already exists."""
    checkpoint_path = ROOT / "outputs" / "checkpoints" / checkpoint_name
    if checkpoint_path.exists() and not force:
        print(f"Skipping existing checkpoint: {checkpoint_path}")
        return checkpoint_path

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = [python_exec, "-m", "src.trainer", "--config", config_path, "--checkpoint_name", checkpoint_name]
    if extra_flags:
        cmd.extend(extra_flags)
    run_command(cmd, env=env)
    return checkpoint_path


def maybe_run_baselines(python_exec: str, config_path: str, force: bool = False) -> Path:
    """Run the baseline benchmark script unless cached results already exist."""
    output_path = ROOT / "outputs" / "baseline_results.json"
    if output_path.exists() and not force:
        print(f"Skipping existing baseline results: {output_path}")
        return output_path

    run_command([python_exec, "-u", "scripts/run_baselines.py", "--config", config_path])
    return output_path


def maybe_run_inference(
    python_exec: str,
    config_path: str,
    model_path: Path,
    user_id: str,
    top_n: int,
    enable_llm_profile: bool = False,
) -> str:
    """Run a demo inference and return stdout for the final summary."""
    cmd = [
        python_exec,
        "-m",
        "src.inference",
        "--config",
        config_path,
        "--model_path",
        str(model_path),
        "--user_id",
        user_id,
        "--top_n",
        str(top_n),
    ]
    if enable_llm_profile:
        cmd.append("--enable_llm_profile")

    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--user_id", default="AE22XHMBOBJBXUFCTNYLFMD4UKMA")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Rerun every stage even if outputs already exist")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_main", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--enable_llm_profile_demo", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "config": args.config,
        "python_exec": args.python_exec,
        "baselines": {},
        "main_model": {},
        "ablations": {},
        "inference_demo": {},
    }

    if not args.skip_baselines:
        baseline_path = maybe_run_baselines(args.python_exec, args.config, force=args.force)
        summary["baselines"] = json.loads(baseline_path.read_text())

    main_checkpoint = checkpoint_dir / "final_main.pt"
    legacy_main_checkpoint = checkpoint_dir / "best_model.pt"
    if not main_checkpoint.exists() and legacy_main_checkpoint.exists() and not args.force:
        shutil.copy2(legacy_main_checkpoint, main_checkpoint)

    if not args.skip_main:
        main_checkpoint = maybe_run_train(
            args.python_exec,
            args.config,
            "final_main.pt",
            extra_flags=None,
            force=args.force,
        )

    if main_checkpoint.exists():
        summary["main_model"] = {
            "checkpoint": str(main_checkpoint.relative_to(ROOT)),
            "metrics": load_checkpoint_metrics(main_checkpoint),
        }

    if not args.skip_ablations:
        ablation_specs = [
            ("no_image", ["--no_image"]),
            ("no_text", ["--no_text"]),
            ("no_relation", ["--no_relation"]),
            ("no_rerank", ["--no_rerank"]),
        ]
        for name, flags in ablation_specs:
            checkpoint_path = maybe_run_train(
                args.python_exec,
                args.config,
                f"ablation_{name}.pt",
                extra_flags=flags,
                force=args.force,
            )
            summary["ablations"][name] = {
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "metrics": load_checkpoint_metrics(checkpoint_path),
            }

    if not args.skip_inference and main_checkpoint.exists():
        summary["inference_demo"]["standard"] = maybe_run_inference(
            args.python_exec,
            args.config,
            main_checkpoint,
            args.user_id,
            args.top_n,
            enable_llm_profile=False,
        )
        if args.enable_llm_profile_demo:
            summary["inference_demo"]["llm_profile"] = maybe_run_inference(
                args.python_exec,
                args.config,
                main_checkpoint,
                args.user_id,
                args.top_n,
                enable_llm_profile=True,
            )

    ablation_path = output_dir / "ablation_results.json"
    ablation_path.write_text(json.dumps(summary.get("ablations", {}), indent=2, ensure_ascii=False))

    summary_path = output_dir / "full_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved pipeline summary to {summary_path}")


if __name__ == "__main__":
    main()
