"""Run all Stage 5 datasets via the provided ECS170 base-class interfaces."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.stage5.run_stage5 import parse_args, local_path
from script.stage5.data import NAMES
from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Method_GCN_Cora import Method_GCN_Cora
from local_code.stage_5_code.Method_GCN_Citeseer import Method_GCN_Citeseer
from local_code.stage_5_code.Method_GCN_Pubmed import Method_GCN_Pubmed
from local_code.stage_5_code.Setting_GCN_Experiment import Setting_GCN_Experiment
from local_code.stage_5_code.Evaluate_Node_Classification import Evaluate_Node_Classification
from local_code.stage_5_code.Result_Saver import Result_Saver

METHODS = {
    "cora": Method_GCN_Cora,
    "citeseer": Method_GCN_Citeseer,
    "pubmed": Method_GCN_Pubmed
}


def main():
    args = parse_args()
    datasets = NAMES if args.dataset == "all" else (args.dataset,)
    for name in datasets:
        root = local_path(args.data_root, name) if args.source == "local" else args.data_root
        data = Dataset_Loader(name, args.source, str(root), args.seed,
                              args.allow_generated_split)
        if args.check_data:
            graph = data.load()
            print(json.dumps({"DATA_VALIDATED": name,
                              "provenance": graph.provenance}, indent=2))
            continue
        folder = (Path(args.output_dir) / name /
                  ("quick" if args.quick else "full"))
        model = METHODS[name](args, folder)
        saver = Result_Saver(folder)
        evaluator = Evaluate_Node_Classification()
        setting = Setting_GCN_Experiment()
        setting.prepare(data, model, saver, evaluator)
        setting.print_setup_summary()
        scores, _ = setting.load_run_save_evaluate()
        print(f"FINISHED {name}:", scores)


if __name__ == "__main__":
    main()
