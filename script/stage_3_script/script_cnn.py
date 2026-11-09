"""ECS170 official-template Stage 3 CNN experiment entrypoint.

Example (from repository root):
python script/stage_3_script/script_cnn_mnist.py --epochs 15
python script/stage_3_script/script_cnn_orl.py --source local --data-root C:\\data\\orl
python script/stage_3_script/script_cnn_cifar.py --epochs 15
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_CNN_Experiment import Setting_CNN_Experiment
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification
from local_code.stage_3_code.Result_Saver import Result_Saver

MODELS = {
    "mnist": Method_CNN_MNIST,
    "orl": Method_CNN_ORL,
    "cifar10": Method_CNN_CIFAR,
}


def main(default_dataset="mnist"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=MODELS, default=default_dataset)
    parser.add_argument("--source", choices=("standard", "local"),
                        default="standard")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Instructor Stage 3 ZIP, NPZ, or class-directory path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--channels-last", action="store_true",
                        help="Use channels-last convolution weights for CPU training")
    parser.add_argument("--configs", default="baseline,residual,kernel5")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny subset smoke test; results are NOT report scores")
    args = parser.parse_args()

    if args.dataset == "orl" and args.source != "local":
        parser.error("ORL needs --source local --data-root PATH to instructor faces")
    if args.source == "local" and not args.data_root:
        parser.error("--source local requires --data-root PATH")
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size and patience must be >= 1")
    configs = tuple(x.strip() for x in args.configs.split(",") if x.strip())
    if not configs or any(x not in ("baseline", "residual", "kernel5")
                          for x in configs) or len(set(configs)) != len(configs):
        parser.error("Use distinct --configs baseline,residual,kernel5")
    if args.learning_rate <= 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid learning rate or dropout")

    # ---- object initialization (same pattern as Stage 1 script_mlp.py) ----
    data_obj = Dataset_Loader(args.dataset, "ECS170 Stage 3 CNN",
                              dataset_key=args.dataset,
                              source=args.source, quick=args.quick)
    data_obj.dataset_source_folder_path = (
        args.data_root if args.data_root
        else str(ROOT / "data" / "stage_3_data")
    )
    data_obj.dataset_source_file_name = ""

    method_obj = MODELS[args.dataset]("CNN " + args.dataset.upper(), "")
    method_obj.max_epoch = min(args.epochs, 2) if args.quick else args.epochs
    method_obj.batch_size = args.batch_size
    method_obj.patience = args.patience
    method_obj.dropout = args.dropout
    method_obj.learning_rate = args.learning_rate
    method_obj.device_name = args.device
    method_obj.channels_last = args.channels_last

    result_obj = Result_Saver("result saver", "")
    folder = ROOT / "result" / "stage_3_result" / args.dataset / (
        "quick" if args.quick else "full")
    result_obj.result_destination_folder_path = str(folder)
    result_obj.result_destination_file_name = "experiment_summary.json"

    setting_obj = Setting_CNN_Experiment("CNN train/validation/test", "")
    setting_obj.configurations = configs
    evaluate_obj = Evaluate_Classification("macro classification metrics", "")
    # ---- running section (same flow as instructor template) ----
    print("************ Start Stage 3 ************")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    score, _ = setting_obj.load_run_save_evaluate()
    print("************ Overall Performance ************")
    print(args.dataset.upper(), "test scores:", score)
    print("************ Finish ************")


if __name__ == "__main__":
    main()
