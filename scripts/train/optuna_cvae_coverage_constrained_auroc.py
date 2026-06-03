"""Tune CVAE hyperparameters with coverage-constrained target AUROC.

The objective is AUROC when cov90 mean absolute error is below the threshold.
Trials above the threshold receive a penalty proportional to the excess
coverage error, so Optuna still learns from near-miss configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml


BASE_HIDDEN_LAYERS = [512, 512, 256, 256]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.output_root / 'optuna_study.db'}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        return run_trial(trial, args)

    study.optimize(objective, n_trials=args.n_trials)
    write_study_outputs(study, args.output_root)
    print(f"best trial: {study.best_trial.number}")
    print(f"best objective: {study.best_value:.6f}")
    print(json.dumps(study.best_trial.params, indent=2), flush=True)


def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    params = suggest_params(trial, args)
    run_dir = args.output_root / f"trial_{trial.number:04d}"
    eval_dir = run_dir / "eval"
    cov_dir = run_dir / "coverage"
    config_dir = run_dir / "_config"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    cov_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    apply_trial_config(config, args, params, run_dir)
    trial_config = config_dir / "config.yaml"
    trial_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    train_cmd = [
        sys.executable,
        "scripts/train/train_conditional_vae.py",
        "--config",
        str(trial_config),
        "--disable-early-stopping",
    ]
    train_returncode = run_command(train_cmd, run_dir / "train.log", env=env)
    trial.set_user_attr("train_returncode", train_returncode)
    if train_returncode != 0:
        trial.set_user_attr("status", "train_failed")
        return args.failure_value

    checkpoint_path = run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists() or args.use_latest_checkpoint:
        checkpoint_path = run_dir / "checkpoint_latest.pt"
    if not checkpoint_path.exists():
        trial.set_user_attr("status", "missing_checkpoint")
        return args.failure_value
    trial.set_user_attr("checkpoint", str(checkpoint_path))

    common_eval_args = [
        "--checkpoint",
        str(checkpoint_path),
        "--target-group",
        args.target_group,
        "--dataset-name",
        args.dataset_name,
        "--dataset-root",
        args.dataset_root,
        "--device",
        args.device,
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.max_rows_per_split is not None:
        common_eval_args.extend(["--max-rows", str(args.max_rows_per_split)])

    auroc_cmd = [
        sys.executable,
        "scripts/eval/evaluate_cvae_prior_probability.py",
        *common_eval_args,
        "--variant",
        "optuna_cvae",
        "--output-dir",
        str(eval_dir),
        "--num-samples",
        str(args.auroc_num_samples),
    ]
    auroc_returncode = run_command(auroc_cmd, run_dir / "eval_auroc.log", env=env)
    trial.set_user_attr("auroc_returncode", auroc_returncode)
    if auroc_returncode != 0:
        trial.set_user_attr("status", "auroc_eval_failed")
        return args.failure_value

    metrics_path = eval_dir / f"optuna_cvae_{args.target_group}_metrics.json"
    if not metrics_path.exists():
        trial.set_user_attr("status", "missing_auroc_metrics")
        return args.failure_value
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    auroc = metrics.get("auroc")
    if auroc is None:
        trial.set_user_attr("status", "missing_auroc")
        return args.failure_value

    coverage_cmd = [
        sys.executable,
        "scripts/eval/evaluate_cvae_coverage.py",
        *common_eval_args,
        "--variant",
        "optuna_cvae",
        "--output-dir",
        str(cov_dir),
        "--num-samples",
        str(args.coverage_num_samples),
        "--interval-levels",
        "0.9",
        "--save-csv",
    ]
    coverage_returncode = run_command(coverage_cmd, run_dir / "eval_coverage.log", env=env)
    trial.set_user_attr("coverage_returncode", coverage_returncode)
    if coverage_returncode != 0:
        trial.set_user_attr("status", "coverage_eval_failed")
        return args.failure_value

    coverage_path = cov_dir / f"optuna_cvae_{args.target_group}_test_coverage.json"
    if not coverage_path.exists():
        trial.set_user_attr("status", "missing_coverage_metrics")
        return args.failure_value
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    cov90 = extract_cov90(coverage)
    if cov90 is None:
        trial.set_user_attr("status", "missing_cov90")
        return args.failure_value

    cov90_error = float(cov90["mean_abs_coverage_error"])
    cov90_width = float(cov90["mean_interval_width"])
    excess = max(0.0, cov90_error - args.max_cov90_abs_error)
    objective = float(auroc) - args.coverage_penalty * excess

    trial.set_user_attr("status", "ok")
    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("metric_auroc", float(auroc))
    trial.set_user_attr("metric_cov90_abs_error", cov90_error)
    trial.set_user_attr("metric_cov90_width", cov90_width)
    trial.set_user_attr("metric_coverage_ok", cov90_error <= args.max_cov90_abs_error)
    trial.set_user_attr("objective", objective)
    for key, value in metrics.items():
        if isinstance(value, int | float | str | bool) or value is None:
            trial.set_user_attr(f"metric_{key}", value)
    return objective


def extract_cov90(coverage: dict[str, Any]) -> dict[str, Any] | None:
    aggregate = coverage.get("aggregate", {})
    rows = aggregate.get("by_interval_level") or []
    for row in rows:
        if abs(float(row.get("interval_level", -1.0)) - 0.9) < 1e-9:
            return row
    return None


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    width_multiplier = trial.suggest_categorical(
        "width_multiplier", args.width_multipliers
    )
    return {
        "beta": trial.suggest_float("beta", args.beta_min, args.beta_max, log=True),
        "learning_rate": trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", args.weight_decay_min, args.weight_decay_max, log=True
        ),
        "latent_dim": trial.suggest_categorical("latent_dim", args.latent_dims),
        "beta_warmup_steps": trial.suggest_categorical(
            "beta_warmup_steps", args.beta_warmup_steps
        ),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "width_multiplier": width_multiplier,
        "hidden_layers": [
            max(1, int(value * width_multiplier)) for value in BASE_HIDDEN_LAYERS
        ],
    }


def apply_trial_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    params: dict[str, Any],
    run_dir: Path,
) -> None:
    config["output_dir"] = str(run_dir)
    config["device"] = args.device
    config["seed"] = args.seed + args.seed_stride * int(run_dir.name.split("_")[-1])
    data_config = config.setdefault("data", {})
    data_config["dataset_root"] = args.dataset_root
    data_config["dataset_name"] = args.dataset_name
    data_config["target_group"] = args.target_group
    data_config["source_groups"] = args.source_groups
    if args.max_rows_per_split is not None:
        data_config["max_rows_per_split"] = args.max_rows_per_split

    model_config = config.setdefault("model", {})
    model_config["latent_dim"] = params["latent_dim"]
    model_config["dropout"] = params["dropout"]
    model_config["encoder_hidden_layers"] = params["hidden_layers"]
    model_config["prior_hidden_layers"] = params["hidden_layers"]
    model_config["decoder_hidden_layers"] = params["hidden_layers"]

    train_config = config.setdefault("train", {})
    train_config["steps"] = args.steps
    train_config["batch_size"] = params["batch_size"]
    train_config["learning_rate"] = params["learning_rate"]
    train_config["weight_decay"] = params["weight_decay"]
    train_config["beta"] = params["beta"]
    train_config["beta_warmup_steps"] = params["beta_warmup_steps"]
    train_config["num_workers"] = args.num_workers
    train_config.setdefault("early_stopping", {})["enabled"] = False

    logging_config = config.setdefault("logging", {})
    logging_config["log_every"] = args.log_every
    logging_config["validate_every"] = args.validate_every
    logging_config["checkpoint_every"] = args.checkpoint_every or args.steps
    logging_config["save_best"] = True


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    return int(process.returncode)


def write_study_outputs(study: optuna.Study, output_root: Path) -> None:
    best = {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
    }
    (output_root / "best_trial.json").write_text(json.dumps(best, indent=2) + "\n")
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(output_root / "trials.csv", index=False)
    write_ranked_trials(trials, output_root)


def write_ranked_trials(trials: pd.DataFrame, output_root: Path) -> None:
    metric_cols = {
        "user_attrs_metric_auroc": "auroc",
        "user_attrs_metric_cov90_abs_error": "cov90_abs_error",
        "user_attrs_metric_cov90_width": "cov90_width",
        "user_attrs_metric_coverage_ok": "coverage_ok",
        "user_attrs_checkpoint": "checkpoint",
        "user_attrs_run_dir": "run_dir",
        "user_attrs_status": "status",
    }
    keep = ["number", "value", "state", *[c for c in metric_cols if c in trials.columns]]
    for col in trials.columns:
        if col.startswith("params_"):
            keep.append(col)
    ranked = trials.loc[:, [c for c in keep if c in trials.columns]].rename(columns=metric_cols)
    if "coverage_ok" in ranked.columns:
        ranked = ranked.sort_values(
            by=["coverage_ok", "auroc"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        ranked = ranked.sort_values("value", ascending=False, na_position="last")
    ranked.to_csv(output_root / "ranked_trials.csv", index=False)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train/train_conditional_vae_eddi_z16_beta010.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/optuna_cvae_coverage_constrained_auroc"))
    parser.add_argument("--study-name", default="cvae_coverage_constrained_auroc")
    parser.add_argument("--storage", help="Optuna storage URL. Defaults to output-root/optuna_study.db.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--target-group", default="diabetes")
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument(
        "--source-groups",
        nargs="+",
        default=["questionnaire_without_disease", "dietary"],
    )
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--auroc-num-samples", type=int, default=200)
    parser.add_argument("--coverage-num-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--beta-min", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=0.3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.2)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--beta-warmup-steps", nargs="+", type=int, default=[500, 1000, 1500])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1024])
    parser.add_argument("--width-multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--max-cov90-abs-error", type=float, default=0.05)
    parser.add_argument("--coverage-penalty", type=float, default=10.0)
    parser.add_argument("--failure-value", type=float, default=0.0)
    parser.add_argument("--use-latest-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
