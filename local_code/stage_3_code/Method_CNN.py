"""Template-compliant CNN MethodModule with PyTorch training and evaluation.

Inherits BOTH the instructor's method base class and torch.nn.Module,
just like the supplied stage-1 Method_MLP. Only validation data are seen
during training/selection; held-out testing occurs in Setting_CNN_Experiment.
"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.base_class.method import method
from script.stage3.models import Stage3CNN
from script.cnn.common import get_device, seed_everything


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 15
    learning_rate = 1e-3
    batch_size = 128
    weight_decay = 1e-4
    label_smoothing = 0.05
    dropout = 0.30
    patience = 4
    seed = 42
    device_name = "auto"
    config = "baseline"
    pool_size = 1
    channels_last = False

    def __init__(self, mName="CNN", mDescription="", dataset_key="mnist"):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.dataset_key = dataset_key
        self.network = None

    def forward(self, x):
        if self.network is None:
            raise RuntimeError("The model must be initialized before forward().")
        return self.network(x)

    @staticmethod
    def score(model, loader, device, criterion):
        model.eval()
        count, total_loss = 0, 0.0
        y_true, pred_y = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                count += len(y)
                total_loss += loss.item() * len(y)
                y_true.extend(y.cpu().tolist())
                pred_y.extend(output.argmax(dim=1).cpu().tolist())
        if count == 0:
            raise ValueError("Empty evaluation dataset")
        return total_loss/count, y_true, pred_y

    def run(self, trainData=None, trainLabel=None, testData=None):
        """Match the template's run() entry point and return validation predictions."""
        if self.data is None:
            raise ValueError("Set method.data using Dataset_Loader.load()")
        if self.max_epoch < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Epoch, batch size and patience must be positive")

        seed_everything(self.seed)
        device = get_device(self.device_name)
        self.network = Stage3CNN(
            in_channels=self.data["provenance"]["channels"],
            num_classes=len(self.data["classes"]),
            config=self.config,
            dropout=self.dropout,
            pool_size=self.pool_size
        ).to(device)
        if self.channels_last:
            self.network.to(memory_format=torch.channels_last)

        training_batches = DataLoader(
            self.data["train"], batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0
        )
        validation_batches = DataLoader(
            self.data["validation"], batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        best_loss, best_weights, stale, history = float("inf"), None, 0, []

        for epoch in range(1, self.max_epoch + 1):
            self.network.train()
            train_loss_sum, count = 0.0, 0
            for x, y in training_batches:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.forward(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                count += len(y)
                train_loss_sum += loss.item()*len(y)
            val_loss, val_true, val_pred = self.score(
                self.network, validation_batches, device, criterion)
            scheduler.step(val_loss)
            history.append({
                "Epoch": epoch, "Train Loss": train_loss_sum/count,
                "Val Loss": val_loss,
                "Learning Rate": optimizer.param_groups[0]["lr"]
            })
            print(f"{self.dataset_key} {self.config}: epoch={epoch:02d} "
                  f"train={train_loss_sum/count:.4f} val={val_loss:.4f}",
                  flush=True)
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.network.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    print(f"{self.config}: early stopping", flush=True)
                    break

        self.network.load_state_dict(best_weights)
        best_val_loss, val_true, val_pred = self.score(
            self.network, validation_batches, device, criterion)
        return {
            "true_y": val_true,
            "pred_y": val_pred,
            "history": history,
            "val_loss": best_val_loss,
            "parameters": sum(p.numel() for p in self.network.parameters()
                              if p.requires_grad),
            "config": self.config
        }
