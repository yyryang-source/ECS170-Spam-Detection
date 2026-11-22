"""Stage 4 instructor-template entry point (same Dataset/Method/Setting style as Stage 3).

From repository root:
    python script/stage_4_script/script_rnn.py --quick --device cpu
    python script/stage_4_script/script_rnn.py --task both --device cpu
    python script/stage_4_script/script_rnn.py --glove ~/Desktop/glove.2024.wikigiga.200d.zip --embedding-dim 200
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Evaluate_RNN import Evaluate_RNN
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_RNN_Experiment import Setting_RNN_Experiment


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/stage_4_data.zip")
    p.add_argument("--task", choices=("both", "classification", "generation"),
                   default="both")
    p.add_argument("--models", default="rnn,lstm,gru")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--glove", default=None)
    p.add_argument("--embedding-dim", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-review-tokens", type=int, default=256)
    p.add_argument("--max-joke-tokens", type=int, default=64)
    p.add_argument("--max-vocab", type=int, default=30000)
    p.add_argument("--max-joke-vocab", type=int, default=10000)
    p.add_argument("--epochs-classification", type=int, default=5)
    p.add_argument("--epochs-generation", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.models = [name.strip().lower() for name in args.models.split(",")]
    if not args.models or any(x not in ("rnn", "lstm", "gru") for x in args.models):
        parser.error("--models must contain rnn,lstm,gru")
    if len(args.models) != len(set(args.models)):
        parser.error("--models must not contain duplicates")
    if min(args.embedding_dim, args.hidden_dim, args.layers,
           args.batch_size, args.max_review_tokens, args.max_vocab,
           args.max_joke_vocab, args.epochs_classification,
           args.epochs_generation) < 1 or args.max_joke_tokens < 5:
        parser.error("Model/training parameters must be positive; joke length >= 5")
    if not (0 <= args.dropout < 1) or args.lr <= 0 or args.top_k < 1:
        parser.error("Invalid dropout, learning rate, or top-k")
    args.output_dir = args.output_dir or str(
        ROOT / "results" / "stage4" /
        ("template_quick" if args.quick else "template_full"))

    dataset_obj = Dataset_Loader("Stage4 instructor text", "IMDb and jokes")
    dataset_obj.dataset_source_folder_path = args.data
    method_obj = Method_RNN("RNN/LSTM/GRU", "PyTorch Stage 4")
    method_obj.args = args
    evaluator_obj = Evaluate_RNN("Stage 4 metrics", "")
    result_obj = Result_Saver("Stage 4 summary", "")
    result_obj.result_destination_folder_path = args.output_dir
    setting_obj = Setting_RNN_Experiment("Stage 4 train/validate/test", "")
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluator_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print("Results:", metrics)
    return metrics


if __name__ == "__main__":
    main()
