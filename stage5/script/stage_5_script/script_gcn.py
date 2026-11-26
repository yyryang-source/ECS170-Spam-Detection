#!/usr/bin/env python3
"""Stage 5 entry point preserving Dataset/Method/Setting/Evaluate/Result classes."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage5.training import Config, write_json
from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Setting_GCN_Experiment import Setting_GCN_Experiment
from local_code.stage_5_code.Evaluate_GCN import Evaluate_GCN
from local_code.stage_5_code.Result_Saver import Result_Saver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing cora, citeseer, pubmed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--datasets", nargs="+", choices=["cora", "citeseer", "pubmed"], default=["cora", "pubmed", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Initialization/dropout seeds; split remains fixed")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--ablation", action="store_true", help="Also run width 32 with identical seeds and splits")
    parser.add_argument("--raw-features", action="store_true", help="Disable default row-L1 normalization")
    args = parser.parse_args()
    all_rows = []
    variants = [("baseline", args.hidden_dim)]
    if args.ablation:
        if args.hidden_dim == 32:
            parser.error("--ablation needs a baseline hidden dimension other than 32")
        variants.append(("hidden32", 32))
    for dataset in args.datasets:
        loader = Dataset_Loader(seed=args.split_seed, dName=dataset, dDescription="Instructor citation network")
        loader.dataset_source_folder_path = args.data_root
        loader.normalize_features = not args.raw_features
        # Load once per dataset while still using the framework's load/run contract.
        cached = loader.load()
        loader.load = lambda data=cached: data
        for variant, width in variants:
            for seed in args.seeds:
                out = args.output_dir / dataset / variant / f"seed_{seed}"
                config = Config(hidden_dim=width, depth=args.depth, dropout=args.dropout,
                                learning_rate=args.lr, weight_decay=args.weight_decay,
                                max_epochs=args.epochs, patience=args.patience, seed=seed,
                                device=args.device, threads=args.threads)
                experiment = Setting_GCN_Experiment()
                experiment.prepare(loader, Method_GCN(config, out), Result_Saver(out / "framework_result.json"), Evaluate_GCN())
                metrics = experiment.load_run_save_evaluate()
                all_rows.append({"dataset": dataset, "variant": variant, "seed": seed, **metrics})
                write_json(args.output_dir / "runs.json", all_rows)
    summary = []
    for dataset in args.datasets:
        for variant, _ in variants:
            rows = [r for r in all_rows if r["dataset"] == dataset and r["variant"] == variant]
            group = {"dataset": dataset, "variant": variant, "seeds": args.seeds, "split_seed": args.split_seed, "n_runs": len(rows)}
            for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "best_epoch", "training_seconds"]:
                values = [r[metric] for r in rows]
                group[metric + "_mean"] = float(np.mean(values))
                group[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary.append(group)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
