#!/usr/bin/env python3
"""Audit Stage 4 historical results; separate real runs from --quick diagnostics.

This reads saved evidence only. It does not train or re-score any models.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


MODELS = ("rnn", "lstm", "gru")
EXPERIMENTS = {
    "random50_full": {"embedding": 50, "initialization": "random"},
    "glove50_full": {"embedding": 50, "initialization": "glove"},
    "full": {"embedding": 100, "initialization": "random"},
    "glove100_full": {"embedding": 100, "initialization": "glove"},
    "glove200_full": {"embedding": 200, "initialization": "glove"},
    "glove300_full": {"embedding": 300, "initialization": "glove"},
}
ROOT = "results/stage4/"


class EvidenceError(ValueError):
    """An expected file or cross-check is missing or inconsistent."""


def require(condition: bool, message: str):
    if not condition:
        raise EvidenceError(message)


def as_json(archive: ZipFile, path: str):
    try:
        return json.loads(archive.read(path))
    except KeyError as exc:
        raise EvidenceError(f"Missing required file: {path}") from exc
    except (ValueError, UnicodeError) as exc:
        raise EvidenceError(f"Invalid JSON in {path}: {exc}") from exc


def as_csv(archive: ZipFile, path: str):
    try:
        text = archive.read(path).decode("utf-8-sig")
    except KeyError as exc:
        raise EvidenceError(f"Missing required file: {path}") from exc
    except UnicodeError as exc:
        raise EvidenceError(f"Invalid UTF-8 CSV in {path}") from exc
    rows = list(csv.DictReader(io.StringIO(text)))
    require(bool(rows), f"Empty CSV file: {path}")
    return rows


def numeric(row, column, path):
    try:
        number = float(row[column])
        require(math.isfinite(number), f"Nonfinite value {column} in {path}")
        return number
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"Invalid number for {column} in {path}") from exc


def compare(actual, expected, message, *, atol=1e-5):
    require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=atol),
            f"{message}: saved={actual} recomputed={expected}")


def inspect_archive(archive: ZipFile) -> dict:
    names = archive.namelist()
    require(len(names) == len(set(names)), "Archive contains duplicate ZIP paths")
    present = set(names)
    require(not archive.testzip(), "Archive contains a corrupt member")

    experiments = {}
    task_count = 0
    for run, config in EXPERIMENTS.items():
        prefix = f"{ROOT}{run}/"
        summary = as_json(archive, prefix + "summary.json")
        require(summary.get("quick") is False,
                f"{run}: flagged as --quick; cannot use in report")
        require(set(summary.get("models", [])) == set(MODELS),
                f"{run}: missing one or more RNN/LSTM/GRU models")
        result = {"embedding": config["embedding"],
                  "initialization": config["initialization"]}

        for task, csvfile, length in (
            ("classification", "test_metrics.csv", 5),
            ("generation", "validation_metrics.csv", 30),
        ):
            taskdir = prefix + task + "/"
            settings = as_json(archive, taskdir + "settings.json")
            vocab = as_json(archive, taskdir + "vocab.json")
            require(settings.get("quick") is False,
                    f"{run}/{task}: quick settings are not final evidence")
            expected_size = 30000 if task == "classification" else 4491
            require(len(vocab) == settings.get("vocab_size") == expected_size,
                    f"{run}/{task}: unexpected vocab size")
            if task == "classification":
                require((settings.get("train"), settings.get("validation"),
                         settings.get("instructor_test")) == (22500, 2500, 25000),
                        f"{run}: the IMDb train/val/official test sizes changed")
            else:
                require((settings.get("train"), settings.get("validation")) ==
                        (1460, 162), f"{run}: unexpected joke train/val split")
                require(settings.get("no_official_test_split") is True,
                        f"{run}: joke task must not claim an official test split")

            rows = as_csv(archive, taskdir + csvfile)
            require(len(rows) == 3 and {r.get("model") for r in rows} == set(MODELS),
                    f"{run}/{task}: missing or duplicate model rows")
            histories = {}
            for model in MODELS:
                path = taskdir + model + "_history.csv"
                history = as_csv(archive, path)
                epochs = [numeric(row, "epoch", path) for row in history]
                require(epochs == list(range(1, length + 1)),
                        f"{run}/{task}/{model}: expected {length} complete epochs")
                for row in history:
                    numeric(row, "train_loss", path)
                    numeric(row, "val_loss", path)
                require(taskdir + model + "_loss.png" in present,
                        f"{run}/{task}/{model}: missing convergence plot")
                histories[model] = history

            metrics = {}
            for row in rows:
                model = row["model"]
                numeric(row, "parameters", taskdir + csvfile)
                hist = histories[model]
                if task == "classification":
                    val = numeric(row, "best_val_macro_f1", taskdir + csvfile)
                    best = max(float(x["val_macro_f1"]) for x in hist)
                    compare(val, best, f"{run} {model}: best validation F1")
                    for column in ("accuracy", "macro_precision", "macro_recall",
                                   "macro_f1"):
                        value = numeric(row, column, taskdir + csvfile)
                        require(0 <= value <= 1, f"{run}/{model}: {column} out of range")
                else:
                    loss = numeric(row, "loss", taskdir + csvfile)
                    best = min(float(x["val_loss"]) for x in hist)
                    compare(loss, best, f"{run} {model}: best validation loss")
                    ppl = numeric(row, "perplexity", taskdir + csvfile)
                    compare(ppl, math.exp(loss), f"{run} {model}: perplexity")
                    accuracy = numeric(row, "token_accuracy", taskdir + csvfile)
                    require(0 <= accuracy <= 1,
                            f"{run}/{model}: token accuracy out of range")
                metrics[model] = row

            if task == "generation":
                examples = as_json(archive, taskdir + "generation_examples.json")
                require(set(examples) == set(MODELS),
                        f"{run}: generation example model coverage incomplete")
                for model in MODELS:
                    require(len(examples[model]) == 3,
                            f"{run}/{model}: need 3 three-word prompts")
                    for ex in examples[model]:
                        prompt = ex.get("prompt", "").strip()
                        reference = ex.get("training_reference", "").strip()
                        generated = ex.get("generated", "").strip()
                        require(len(prompt.split()) == 3,
                                f"{run}/{model}: prompt is not three words")
                        require(reference.lower().startswith(prompt.lower()),
                                f"{run}/{model}: reference does not share prompt")
                        require(generated.lower().startswith(prompt.lower()),
                                f"{run}/{model}: output does not retain prompt")
            result[task] = metrics
            task_count += len(rows)
        experiments[run] = result

    # The legacy unscoped results/stage4/ directory is an earlier one-epoch
    # --quick run and MUST NOT be included in the 36 official experiment rows.
    quick_summary = ROOT + "summary.json"
    diagnostic_files = [
        name for name in names
        if name.startswith(ROOT)
        and len(name[len(ROOT):].split("/")) <= 2
        and not any(name.startswith(ROOT + run + "/") for run in EXPERIMENTS)
    ]
    warnings = []
    if quick_summary in present:
        quick = as_json(archive, quick_summary)
        if quick.get("quick") is True:
            warnings.append(
                "Ignored top-level one-epoch --quick diagnostic results; "
                "do not submit these as final metrics.")
        else:
            warnings.append("Top-level historical results were not audited.")
    require(task_count == 36, f"Expected 36 official trained task/model rows, found {task_count}")

    official = []
    joke = []
    for config, results in experiments.items():
        for model, row in results["classification"].items():
            official.append({
                "run": config, "model": model,
                "validation_f1": float(row["best_val_macro_f1"]),
                "test_macro_f1": float(row["macro_f1"]),
            })
        for model, row in results["generation"].items():
            joke.append({
                "run": config, "model": model,
                "validation_ppl": float(row["perplexity"]),
            })
    by_validation = max(official, key=lambda item: item["validation_f1"])
    by_test = max(official, key=lambda item: item["test_macro_f1"])
    by_ppl = min(joke, key=lambda item: item["validation_ppl"])
    return {
        "status": "pass",
        "official_runs": len(EXPERIMENTS),
        "verified_model_task_runs": task_count,
        "diagnostic_files_ignored": len(diagnostic_files),
        "warnings": warnings,
        "classifier_selected_by_validation": by_validation,
        "highest_observed_imdb_test_f1": by_test,
        "lowest_joke_validation_perplexity": by_ppl,
        "source": "Saved Stage 4 experiment CSV/JSON/PNG files only; no retraining.",
    }


def create_submission_zip(source: ZipFile, destination: Path):
    """Create a clean evidence archive with only six complete non-quick runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(ROOT + run + "/" for run in EXPERIMENTS)
    names = sorted(n for n in source.namelist()
                   if n.startswith(prefixes)
                   and n.endswith((".csv", ".json", ".png")))
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as output:
        for name in names:
            output.writestr(name, source.read(name))
    return len(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("stage4_report_results.zip"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--clean-output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        with ZipFile(args.archive) as archive:
            result = inspect_archive(archive)
            if args.clean_output is not None:
                result["clean_archive_members"] = create_submission_zip(
                    archive, args.clean_output)
                result["clean_archive"] = str(args.clean_output)
    except (OSError, EvidenceError, ValueError) as exc:
        parser.exit(1, f"Stage 4 evidence audit FAILED: {exc}\n")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
