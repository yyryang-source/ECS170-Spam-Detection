"""Instructor-template experiment wiring: load -> method.run -> evaluate -> save."""
from pathlib import Path

from local_code.base_class.setting import setting
from script.stage4.architecture import plot_recurrent_architecture


class Setting_RNN_Experiment(setting):
    def load_run_save_evaluate(self):
        loaded = self.dataset.load()
        self.method.data = loaded
        self.method.result_destination_folder_path = (
            self.result.result_destination_folder_path)
        prediction = self.method.run()
        self.evaluate.data = prediction
        summary = self.evaluate.evaluate()
        output = Path(self.result.result_destination_folder_path)
        args = self.method.args
        for task, info in summary.items():
            vocab_path = output / task / "vocab.json"
            import json
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            plot_recurrent_architecture(
                task=task, kind=info["selected_model"],
                embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                layers=args.layers, dropout=args.dropout, vocab_size=len(vocab),
                destination=output / task / "model_architecture.png")
        self.result.data = {
            "course": "ECS170 Fall 2026", "stage": 4,
            "source": loaded, "device": prediction["device"],
            "quick_smoke_test": bool(args.quick),
            "settings": {
                "task": args.task, "models": list(args.models),
                "embedding_dim": args.embedding_dim,
                "initialization": "GloVe fine-tuned" if args.glove else "random",
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "dropout": args.dropout, "batch_size": args.batch_size,
                "seed": args.seed, "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "classification_epochs": 1 if args.quick else args.epochs_classification,
                "generation_epochs": 1 if args.quick else args.epochs_generation,
            },
            "model_selection_uses_official_test": False,
            "metrics": summary,
        }
        path = self.result.save()
        print("Stage 4 instructor-template summary:", path)
        if args.quick:
            print("QUICK MODE: diagnostic only; never report these scores.")
        return summary, None
