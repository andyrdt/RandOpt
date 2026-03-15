#!/usr/bin/env python3
"""Summarize compact RandOpt run artifacts from codex/experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_results(experiments_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(experiments_dir.glob("countdown_*/results.json")):
        with path.open() as f:
            data = json.load(f)
        data["_results_path"] = str(path)
        data["_run_dir"] = path.parent.name
        runs.append(data)
    return runs


def summarize_runs(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        ensemble = run.get("ensemble_results", {})
        ensemble_items = sorted(
            (
                (int(key), value.get("accuracy"))
                for key, value in ensemble.items()
                if value.get("accuracy") is not None
            ),
            key=lambda item: item[0],
        )
        sigma_stats = run.get("sigma_stats", {})
        samples = run.get("all_sampled_perturbations", [])
        rewards = [sample.get("train_reward") for sample in samples if sample.get("train_reward") is not None]
        base_train = run.get("base_train_accuracy")
        hit_rate_gt_base = None
        hit_rate_gt_base_plus_005 = None
        max_reward = None
        mean_reward = None
        if rewards:
            max_reward = max(rewards)
            mean_reward = sum(rewards) / len(rewards)
        if rewards and base_train is not None:
            hit_rate_gt_base = sum(reward > base_train for reward in rewards) / len(rewards)
            hit_rate_gt_base_plus_005 = sum(reward > (base_train + 0.05) for reward in rewards) / len(rewards)
        best_sigma = run.get("best_sigma")
        best_sigma_mean = None
        if best_sigma is not None:
            best_sigma_mean = sigma_stats.get(str(best_sigma), {}).get("mean")
        ensemble_accuracies = [accuracy for _, accuracy in ensemble_items]
        best_ensemble_accuracy = max(ensemble_accuracies) if ensemble_accuracies else None
        smallest_k = ensemble_items[0][0] if ensemble_items else None
        smallest_k_accuracy = ensemble_items[0][1] if ensemble_items else None
        middle_k = ensemble_items[len(ensemble_items) // 2][0] if ensemble_items else None
        middle_k_accuracy = ensemble_items[len(ensemble_items) // 2][1] if ensemble_items else None
        largest_k = ensemble_items[-1][0] if ensemble_items else None
        largest_k_accuracy = ensemble_items[-1][1] if ensemble_items else None
        mean_reward_delta = None
        if mean_reward is not None and base_train is not None:
            mean_reward_delta = mean_reward - base_train
        single_sigma_value = None
        if len(sigma_stats) == 1:
            single_sigma_value = float(next(iter(sigma_stats.keys())))
        rows.append(
            {
                "run_dir": run["_run_dir"],
                "model": run.get("model"),
                "perturb_seed_mode": run.get("perturb_seed_mode"),
                "perturb_scale_mode": run.get("perturb_scale_mode", "absolute"),
                "skip_base_test_eval": run.get("skip_base_test_eval", False),
                "skip_ensemble_eval": run.get("skip_ensemble_eval", False),
                "train_samples": run.get("train_samples"),
                "test_samples": run.get("test_samples"),
                "num_sigmas": len(sigma_stats),
                "single_sigma_value": single_sigma_value,
                "base_train_accuracy": run.get("base_train_accuracy"),
                "base_test_accuracy": run.get("base_test_accuracy"),
                "mean_sampled_train_reward": mean_reward,
                "mean_reward_delta_vs_base_train": mean_reward_delta,
                "max_sampled_train_reward": max_reward,
                "hit_rate_gt_base_train": hit_rate_gt_base,
                "hit_rate_gt_base_train_plus_0.05": hit_rate_gt_base_plus_005,
                "best_sigma": best_sigma,
                "best_sigma_mean_reward": best_sigma_mean,
                "smallest_k": smallest_k,
                "smallest_k_accuracy": smallest_k_accuracy,
                "middle_k": middle_k,
                "middle_k_accuracy": middle_k_accuracy,
                "largest_k": largest_k,
                "largest_k_accuracy": largest_k_accuracy,
                "k1_accuracy": ensemble.get("1", {}).get("accuracy"),
                "k2_accuracy": ensemble.get("2", {}).get("accuracy"),
                "k4_accuracy": ensemble.get("4", {}).get("accuracy"),
                "best_ensemble_accuracy": best_ensemble_accuracy,
                "num_sampled_perturbations": len(run.get("all_sampled_perturbations", [])),
                "results_path": run["_results_path"],
            }
        )
    return rows


def summarize_sigmas(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        for sigma_str, stats in sorted(run.get("sigma_stats", {}).items(), key=lambda item: float(item[0])):
            rows.append(
                {
                    "run_dir": run["_run_dir"],
                    "model": run.get("model"),
                    "perturb_seed_mode": run.get("perturb_seed_mode"),
                    "perturb_scale_mode": run.get("perturb_scale_mode", "absolute"),
                    "skip_base_test_eval": run.get("skip_base_test_eval", False),
                    "skip_ensemble_eval": run.get("skip_ensemble_eval", False),
                    "sigma": float(sigma_str),
                    "mean_reward": stats.get("mean"),
                    "count": stats.get("count"),
                    "results_path": run["_results_path"],
                }
            )
    return rows


def summarize_samples(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        base_train = run.get("base_train_accuracy")
        for sample in run.get("all_sampled_perturbations", []):
            reward = sample.get("train_reward")
            rows.append(
                {
                    "run_dir": run["_run_dir"],
                    "model": run.get("model"),
                    "perturb_seed_mode": run.get("perturb_seed_mode"),
                    "perturb_scale_mode": run.get("perturb_scale_mode", "absolute"),
                    "skip_base_test_eval": run.get("skip_base_test_eval", False),
                    "skip_ensemble_eval": run.get("skip_ensemble_eval", False),
                    "sigma": sample.get("sigma"),
                    "seed": sample.get("seed"),
                    "train_reward": reward,
                    "base_train_accuracy": base_train,
                    "above_base_train": None if reward is None or base_train is None else reward > base_train,
                    "above_base_train_plus_0.05": None if reward is None or base_train is None else reward > (base_train + 0.05),
                    "results_path": run["_results_path"],
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("codex/experiments"),
        help="Directory containing countdown_* result subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex/analysis"),
        help="Directory to write compact summary tables",
    )
    args = parser.parse_args()

    runs = load_results(args.experiments_dir)
    run_rows = summarize_runs(runs)
    sigma_rows = summarize_sigmas(runs)
    sample_rows = summarize_samples(runs)

    write_csv(run_rows, args.output_dir / "run_summary.csv")
    write_csv(sigma_rows, args.output_dir / "sigma_summary.csv")
    write_csv(sample_rows, args.output_dir / "sample_summary.csv")

    print(f"Wrote {len(run_rows)} run summaries to {args.output_dir / 'run_summary.csv'}")
    print(f"Wrote {len(sigma_rows)} sigma summaries to {args.output_dir / 'sigma_summary.csv'}")
    print(f"Wrote {len(sample_rows)} sampled perturbations to {args.output_dir / 'sample_summary.csv'}")


if __name__ == "__main__":
    main()
