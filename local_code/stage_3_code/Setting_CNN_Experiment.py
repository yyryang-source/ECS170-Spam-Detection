"""Instructor-style SettingModule: load -> train -> save -> evaluate."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.base_class.setting import setting
from local_code.stage_3_code.Architecture_Plot import plot_architecture
from script.cnn.common import (get_device, plot_confusion, plot_loss,
                               plot_train_val, save_csv, save_json)


class Setting_CNN_Experiment(setting):
    configurations = ("baseline", "residual", "kernel5")

    def load_run_save_evaluate(self):
        # Keep the same load -> method.run -> result.save -> evaluate pattern
        # as the instructor's stage_1 Setting_Train_Test_Split class.
        loaded = self.dataset.load()
        output = Path(self.result.result_destination_folder_path)
        output.mkdir(parents=True, exist_ok=True)
        val_rows = []
        histories = {}
        trained = {}

        for config in self.configurations:
            self.method.config = config
            self.method.data = loaded
            prediction = self.method.run()  # train; predict ONLY validation split
            self.evaluate.data = prediction
            val_metrics = self.evaluate.evaluate()
            val_rows.append({
                "Model": config, "Parameters": prediction["parameters"],
                "Val Loss": prediction["val_loss"], **val_metrics
            })
            histories[config] = prediction["history"]
            trained[config] = self.method.network
            save_csv(output / f"{config}_loss_history.csv",
                     prediction["history"])
            print("VALIDATION:", config, val_metrics)

        save_csv(output / "ablation_validation.csv", val_rows)
        winner = max(val_rows, key=lambda r: r["F1"])["Model"]
        winner_model = trained[winner]
        device = get_device(self.method.device_name)
        test_loader = DataLoader(
            loaded["test"], batch_size=self.method.batch_size,
            num_workers=0
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.method.label_smoothing
        )
        test_loss, true_y, pred_y = self.method.score(
            winner_model, test_loader, device, criterion)
        learned_result = {"pred_y": pred_y, "true_y": true_y}
        self.evaluate.data = learned_result
        test_metrics = self.evaluate.evaluate()
        save_csv(output / "final_test_metrics.csv", [{
            "Dataset": self.dataset.dataset_key, "Model": winner,
            "Test Loss": test_loss, **test_metrics
        }])

        plot_loss(histories, output / "ablation_val_loss.png",
                  f"{self.dataset.dataset_key.upper()} Validation Loss")
        plot_train_val(histories[winner], output / "best_loss_curve.png",
                       f"{self.dataset.dataset_key.upper()} {winner} Convergence")
        plot_confusion(true_y, pred_y, loaded["classes"],
                       output / "confusion_matrix.png",
                       f"{self.dataset.dataset_key.upper()} Test Confusion Matrix")
        plot_architecture(
            winner, self.dataset.dataset_key, loaded["classes"],
            loaded["provenance"]["channels"],
            output / "model_architecture.png",
            sum(p.numel() for p in winner_model.parameters()
                if p.requires_grad),
            pool_size=self.method.pool_size
        )
        torch.save({
            "model": winner_model.cpu().state_dict(),
            "config": winner,
            "dataset": self.dataset.dataset_key,
            "classes": loaded["classes"],
            "channels": loaded["provenance"]["channels"],
            "dropout": self.method.dropout,
            "pool_size": self.method.pool_size,
        }, output / "best_model.pt")
        metadata = {
            "Student": "Yourui Yang",
            "Course": "ECS170 Fall 2026",
            "Dataset": self.dataset.dataset_key,
            "Dataset Provenance": loaded["provenance"],
            "Device": str(device),
            "Pooling Grid": self.method.pool_size,
            "Architecture Configurations": list(self.configurations),
            "Epochs": self.method.max_epoch,
            "Batch Size": self.method.batch_size,
            "AdamW LR": self.method.learning_rate,
            "Weight Decay": self.method.weight_decay,
            "Label Smoothing": self.method.label_smoothing,
            "Dropout": self.method.dropout,
            "Early Stopping Patience": self.method.patience,
            "Seed": self.method.seed,
            "Best Model": winner,
            "Best Selection": "Highest validation macro F1",
            "Test Used For Selection": False,
            "Validation": val_rows,
            "Test": test_metrics,
            "Quick Smoke Test": bool(self.dataset.quick),
        }
        self.result.data = metadata
        self.result.save()
        print("\n=== VALIDATION ABLATIONS ===")
        for row in val_rows:
            print(row)
        print("BEST:", winner, "TEST:", test_metrics, "OUT:", output)
        return test_metrics, None
