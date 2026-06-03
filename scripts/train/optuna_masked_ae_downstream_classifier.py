#!/usr/bin/env python
"""Tune masked-AE downstream classifier hyperparameters with Optuna."""

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


HEAD_ARCHITECTURES = {
    "linear": [],
    "h128": [128],
    "h256": [256],
    "h256_256": [256, 256],
    "h512_256": [512, 256],
}


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
    print(f"best valid AUROC: {study.best_value:.6f}")
    print(json.dumps(study.best_trial.params, indent=2), flush=True)


def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    params = suggest_params(trial, args)
    run_dir = args.output_root / f"trial_{trial.number:04d}"
    config_dir = run_dir / "_config"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    apply_trial_config(config, args, params, run_dir, trial.number)
    trial_config = config_dir / "config.yaml"
    trial_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/train/train_masked_ae_downstream_classifier.py",
        "--config",
        str(trial_config),
    ]
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return_code = run_command(cmd, run_dir / "train.log", env=env)
    trial.set_user_attr("train_returncode", return_code)
    trial.set_user_attr("run_dir", str(run_dir))
    if return_code != 0:
        trial.set_user_attr("status", "train_failed")
        return 0.0
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        trial.set_user_attr("status", "missing_summary")
        return 0.0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trial.set_user_attr("status", "ok")
    for key, value in summary.items():
        if isinstance(value, int | float | str | bool) or value is None:
            trial.set_user_attr(f"metric_{key}", value)
    return float(summary.get("best_valid_auroc", 0.0))


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    head_name = trial.suggest_categorical("head_architecture", args.head_architectures)
    return {
        "freeze_backbone": trial.suggest_categorical("freeze_backbone", args.freeze_backbone_choices),
        "class_weight": trial.suggest_categorical("class_weight", args.class_weight_choices),
        "head_architecture": head_name,
        "head_hidden_layers": HEAD_ARCHITECTURES[head_name],
        "dropout": trial.suggest_float("dropout", args.dropout_min, args.dropout_max),
        "learning_rate": trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            args.weight_decay_min,
            args.weight_decay_max,
            log=True,
        ),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
    }


def apply_trial_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    params: dict[str, Any],
    run_dir: Path,
    trial_number: int,
) -> None:
    config["output_dir"] = str(run_dir)
    config["device"] = args.device
    config["seed"] = args.seed + args.seed_stride * trial_number
    data_config = config.setdefault("data", {})
    data_config["dataset_root"] = args.dataset_root
    data_config["dataset_name"] = args.dataset_name
    data_config["source_groups"] = args.source_groups
    data_config["target_disease"] = args.target_disease
    if args.max_rows_per_split is not None:
        data_config["max_rows_per_split"] = args.max_rows_per_split
    elif "max_rows_per_split" in data_config:
        del data_config["max_rows_per_split"]

    model_config = config.setdefault("model", {})
    model_config["pretrained_autoencoder_path"] = str(args.pretrained_autoencoder_path)
    model_config["variant_name"] = args.variant_name
    model_config["freeze_backbone"] = params["freeze_backbone"]
    model_config["class_weight"] = params["class_weight"]
    model_config["head_hidden_layers"] = params["head_hidden_layers"]
    model_config["dropout"] = params["dropout"]

    train_config = config.setdefault("train", {})
    train_config["steps"] = args.steps
    train_config["batch_size"] = params["batch_size"]
    train_config["learning_rate"] = params["learning_rate"]
    train_config["weight_decay"] = params["weight_decay"]
    train_config["num_workers"] = args.num_workers

    logging_config = config.setdefault("logging", {})
    logging_config["log_every"] = args.log_every
    logging_config["validate_every"] = args.validate_every
    logging_config["checkpoint_every"] = args.checkpoint_every or args.steps


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
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials.to_csv(output_root / "trials.csv", index=False)
    complete = trials[trials["state"] == "COMPLETE"].copy()
    if complete.empty:
        return
    complete.sort_values("value", ascending=False).head(20).to_csv(
        output_root / "top_trials.csv",
        index=False,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train/train_masked_ae_downstream_diabetes.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/optuna/masked_ae_downstream"))
    parser.add_argument("--study-name", default="masked_ae_downstream")
    parser.add_argument("--storage")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--source-groups", nargs="+", default=["questionnaire_without_disease", "dietary"])
    parser.add_argument("--target-disease", default="diabetes")
    parser.add_argument("--pretrained-autoencoder-path", type=Path, required=True)
    parser.add_argument("--variant-name", default="selected")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--head-architectures", nargs="+", default=["linear", "h128", "h256", "h256_256", "h512_256"])
    parser.add_argument("--freeze-backbone-choices", nargs="+", type=parse_bool, default=[False, True])
    parser.add_argument("--class-weight-choices", nargs="+", default=["balanced", "none"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-2)
    parser.add_argument("--dropout-min", type=float, default=0.0)
    parser.add_argument("--dropout-max", type=float, default=0.4)
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value!r}")


if __name__ == "__main__":
    main()
