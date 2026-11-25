"""Run Stage 5 GCN experiments (Cora / Citeseer / Pubmed).

Instructor source is always preferred. Public Planetoid is a reference only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.stage5.data import NAMES, load_graph
from script.stage5.model import CONFIGS
from script.stage5.train import run_experiments


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=(*NAMES, "all"), required=True)
    p.add_argument("--source", choices=("local", "planetoid"), default="planetoid")
    p.add_argument("--data-root", default=str(ROOT / "data" / "stage_5_data"),
                   help="Local .npz/.zip, or directory containing cora.npz etc")
    p.add_argument("--output-dir", default=str(ROOT / "results" / "stage5"))
    p.add_argument("--allow-generated-split", action="store_true",
                   help="Explicitly create a stratified split when no official masks exist")
    p.add_argument("--configs", default="baseline,deep,wide,no_dropout")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--quick", action="store_true",
                   help="Only two epochs per config; NEVER use scores for report")
    p.add_argument("--check-data", action="store_true",
                   help="Validate graph and splits, but do not train")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    configs = tuple(z.strip() for z in args.configs.split(",") if z.strip())
    if not configs or len(set(configs)) != len(configs) or any(z not in CONFIGS for z in configs):
        p.error(f"--configs must contain distinct names from {tuple(CONFIGS)}")
    if min(args.epochs, args.patience) < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("Epochs, patience, learning rate must be positive; weight decay >= 0")
    args.config_names = configs
    if args.quick:
        args.epochs = min(args.epochs, 2)
    return args


def local_path(root, name):
    path = Path(root)
    if path.is_dir():
        candidate = path / f"{name}.npz"
        if candidate.is_file():
            return candidate
        archive = path / "stage_5_data.zip"
        if archive.is_file():
            return archive
        raise FileNotFoundError(f"Missing {candidate} or {archive}")
    return path


def main():
    args = parse_args()
    datasets = NAMES if args.dataset == "all" else (args.dataset,)
    for name in datasets:
        data_root = local_path(args.data_root, name) if args.source == "local" else args.data_root
        graph = load_graph(
            name, source=args.source, data_root=data_root,
            seed=args.seed, allow_generated_split=args.allow_generated_split)
        print(json.dumps({"dataset": name, "provenance": graph.provenance},
                         indent=2), flush=True)
        if args.check_data:
            print("DATA_VALIDATED:", name, flush=True)
            continue
        out = Path(args.output_dir) / name / ("quick" if args.quick else "full")
        run_experiments(graph, out, args.config_names, args)


if __name__ == "__main__":
    main()
