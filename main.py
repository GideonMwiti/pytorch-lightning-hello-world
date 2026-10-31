"""
Project 16: PyTorch Lightning Hello World - Clean Modular Deep Learning
Dataset: MNIST Handwritten Digits (60,000 Train/Val, 10,000 Test)
Framework: PyTorch Lightning 2.0+ & PyTorch
Author: Gideon Mwiti

Key Features:
- Decoupled research & engineering code using LightningModule and LightningDataModule.
- Deep Regularized Multi-Layer Perceptron with BatchNorm, ReLU, and Dropout.
- Integrated TorchMetrics for multi-class classification tracking.
- Automatic device placement, validation checks, and optimizer management.
- Custom Lightning Callback for epoch-level metrics tracking and history logging.
- 4-Panel Publication-Quality Diagnostic Visual Artifact:
  1) Training and Validation Cross-Entropy Loss Convergence
  2) Training and Validation Accuracy Trajectories (Reaching >98%)
  3) Qualitative Test Predictions Grid (16 Sample Digits with Confidence Bars)
  4) Normalized 10x10 Confusion Matrix across MNIST Digits (0 - 9)
"""

import os
import sys
import types
import random
import numpy as np
import matplotlib.pyplot as plt

# Workaround for Windows environments where WDAC blocks unsigned bz2/lzma DLLs
if 'bz2' not in sys.modules:
    sys.modules['bz2'] = types.SimpleNamespace(open=open)
if 'lzma' not in sys.modules:
    sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


class MNISTDataModule(pl.LightningDataModule):
    """
    Decoupled data handling module managing MNIST downloads, 
    splits, and DataLoaders for train, validation, and test sets.
    """
    def __init__(self, data_dir, batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        if stage in ('fit', None):
            full_mnist = torchvision.datasets.MNIST(
                root=self.data_dir, train=True, download=False, transform=self.transform
            )
            # 55,000 train / 5,000 validation split
            self.train_dataset, self.val_dataset = random_split(
                full_mnist, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
            
        if stage in ('test', None):
            self.test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)


class LitMNISTClassifier(pl.LightningModule):
    """
    PyTorch LightningModule encapsulating model architecture,
    loss computation, optimization, and training/validation steps.
    """
    def __init__(self, hidden_dim1=256, hidden_dim2=128, lr=0.001, dropout=0.25):
        super().__init__()
        self.save_hyperparameters()
        
        # Deep Feedforward Neural Network
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 10)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # TorchMetrics for accurate multi-class evaluation
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.test_acc(logits, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        return [optimizer], [scheduler]


class MetricsHistoryCallback(Callback):
    """Custom callback to collect epoch-level metrics for visual plotting."""
    def __init__(self):
        super().__init__()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        tr_loss = trainer.callback_metrics.get('train_loss')
        tr_acc = trainer.callback_metrics.get('train_acc')
        if tr_loss is not None:
            self.history['train_loss'].append(tr_loss.item())
        if tr_acc is not None:
            self.history['train_acc'].append(tr_acc.item() * 100.0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_loss is not None:
            self.history['val_loss'].append(val_loss.item())
        if val_acc is not None:
            self.history['val_acc'].append(val_acc.item() * 100.0)


def generate_visual_artifact(history, sample_imgs, sample_lbls, sample_preds, sample_confs,
                             targets, preds, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    # Harmonize array lengths
    num_pts = min(len(history['train_loss']), len(history['val_loss']),
                  len(history['train_acc']), len(history['val_acc']))
    epochs = range(1, num_pts + 1)
    tr_loss = history['train_loss'][:num_pts]
    v_loss = history['val_loss'][:num_pts]
    tr_acc = history['train_acc'][:num_pts]
    v_acc = history['val_acc'][:num_pts]

    # -------------------------------------------------------------
    # Panel 1: Cross-Entropy Loss Convergence
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(epochs, tr_loss, 'b-o', label='Training Loss', linewidth=2.0)
    ax1.plot(epochs, v_loss, 'r-s', label='Validation Loss', linewidth=2.2)
    ax1.set_title("PyTorch Lightning Loss Trajectory vs. Epochs\nSmooth Convergence Managed via LightningModule", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Classification Accuracy Trajectories
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(epochs, tr_acc, 'b--', label='Train Accuracy', alpha=0.7)
    ax2.plot(epochs, v_acc, 'g-o', label=f"Val Accuracy (Final: {v_acc[-1]:.2f}%)", linewidth=2.2)
    ax2.set_title("Training and Validation Classification Accuracy (%)\nRapid Saturation to >98% Generalization Accuracy", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_ylim([90, 100])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Qualitative Sample Predictions Grid (4x4)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.axis('off')
    sub_grid = axes[1, 0].inset_axes([0.0, 0.0, 1.0, 1.0])
    sub_grid.axis('off')

    for idx in range(16):
        row = idx // 4
        col = idx % 4
        sub_ax = sub_grid.inset_axes([col * 0.25 + 0.02, 0.75 - row * 0.25 + 0.02, 0.21, 0.21])

        img = sample_imgs[idx].squeeze().cpu().numpy()
        # Un-normalize for display
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        sub_ax.imshow(img, cmap='gray_r')
        sub_ax.axis('off')

        true_lbl = sample_lbls[idx]
        pred_lbl = sample_preds[idx]
        conf = sample_confs[idx] * 100.0

        is_correct = (true_lbl == pred_lbl)
        color = 'darkgreen' if is_correct else 'crimson'
        sub_ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\n({conf:.1f}%)", 
                         fontsize=8.5, fontweight='bold', color=color, pad=2)

    ax3.set_title("Qualitative Test Predictions (4x4 Grid)\nHigh-Confidence Classification across Handwritten Digits", 
                  fontsize=11, fontweight='bold', pad=15)

    # -------------------------------------------------------------
    # Panel 4: Normalized Confusion Matrix (10x10)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    cm = np.zeros((10, 10))
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im4 = ax4.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks(range(10))
    ax4.set_yticks(range(10))
    ax4.set_xticklabels([str(i) for i in range(10)], fontsize=9)
    ax4.set_yticklabels([str(i) for i in range(10)], fontsize=9)
    ax4.set_xlabel("Predicted Digit Class", fontsize=10)
    ax4.set_ylabel("True Ground Truth Digit", fontsize=10)
    ax4.set_title(f"Normalized Confusion Matrix (10 Classes, N = {len(targets)})\nExtremely Dense Diagonal with Minimal Off-Diagonal Confusions", fontsize=11, fontweight='bold')

    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            if val >= 0.01:
                color = "white" if val > 0.5 else "black"
                ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Accuracy Ratio", fontsize=10)

    plt.suptitle("PyTorch Lightning Modular Digit Classifier (MNIST Benchmark)", 
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisual diagnostic report saved successfully at: {os.path.abspath(save_path)}")


def main():
    print("=" * 72)
    print("        PROJECT 16: PYTORCH LIGHTNING HELLO WORLD ON MNIST")
    print("=" * 72)

    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_path = os.path.join(script_dir, "pytorch_lightning_mnist_results.png")

    # Hyperparameters
    batch_size = 256
    epochs = 5
    learning_rate = 0.001

    print("\n[1/4] Initializing PyTorch Lightning DataModule...")
    data_module = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    data_module.setup()
    print("  Dataset: MNIST (55,000 Train, 5,000 Validation, 10,000 Test)")

    print("\n[2/4] Initializing LightningModule Architecture...")
    model = LitMNISTClassifier(hidden_dim1=256, hidden_dim2=128, lr=learning_rate, dropout=0.25)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: 784 -> 256 (BN, ReLU, Drop) -> 128 (BN, ReLU, Drop) -> 10")
    print(f"  Trainable Parameters: {total_params:,}")

    # Callbacks and Trainer
    metrics_cb = MetricsHistoryCallback()
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(script_dir, "checkpoints"),
        filename="mnist-best-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        callbacks=[metrics_cb, checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=50
    )

    print("\n[3/4] Executing pl.Trainer.fit()...")
    trainer.fit(model, datamodule=data_module)

    print("\n[4/4] Executing pl.Trainer.test()...")
    test_results = trainer.test(model, datamodule=data_module)
    test_acc = test_results[0]['test_acc'] * 100.0
    test_loss = test_results[0]['test_loss']

    print("\n" + "=" * 72)
    print(f"PyTorch Lightning Training Complete!")
    print(f"Final Validation Accuracy : {metrics_cb.history['val_acc'][-1]:.2f}%")
    print(f"Final Out-of-Sample Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    print("=" * 72)

    # Collect predictions for visualization
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    
    sample_imgs, sample_lbls = next(iter(test_loader))
    sample_imgs = sample_imgs[:16]
    sample_lbls = sample_lbls[:16].numpy()

    model.eval()
    with torch.no_grad():
        out = model(sample_imgs)
        probs = torch.softmax(out, dim=1)
        confs, preds = torch.max(probs, dim=1)
        sample_confs = confs.numpy()
        sample_preds = preds.numpy()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            p = torch.argmax(out, dim=1)
            all_preds.extend(p.numpy())
            all_targets.extend(y.numpy())

    # Generate visual artifact
    generate_visual_artifact(
        metrics_cb.history, sample_imgs, sample_lbls, sample_preds, sample_confs,
        np.array(all_targets), np.array(all_preds), save_path
    )


if __name__ == "__main__":
    main()

