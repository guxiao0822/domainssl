import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from einops import rearrange
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

from utils.utils import save_checkpoint, AverageMeter, ProgressMeter  # explicit imports


class SimCLRTrainer:
    """
    Trainer for SimCLR self-supervised learning on ECG data.

    Args:
        config (dict): Configuration dict with keys:
            - model.backbone_fn (str)
            - training.epochs (int)
            - training.batch_size (int)
            - training.temp (float)
            - exp (str): experiment name
            - task (str): task name (e.g., 'ecg')
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        model (nn.Module): SimCLR model instance.
    """
    def __init__(
        self,
        config: dict,
        optimizer: torch.optim.Optimizer,
        scheduler,
        model: nn.Module
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.temp = config['training']['temp']
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_score = 0.0
        self.exp_name = config['exp']
        self.task = config['task']
        self.n_views = 2
        self.scaler = GradScaler()

    def info_nce_loss(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute InfoNCE loss logits and labels.
        """
        batch_size = features.size(0) // self.n_views
        labels = torch.cat([torch.arange(batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        features = F.normalize(features, dim=1)
        similarity = features @ features.T
        mask = torch.eye(labels.size(0), device=self.device).bool()
        labels = labels[~mask].view(labels.size(0), -1)
        similarity = similarity[~mask].view(similarity.size(0), -1)
        positives = similarity[labels.bool()].view(labels.size(0), -1)
        negatives = similarity[~labels.bool()].view(similarity.size(0), -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temp
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return logits, targets

    def train_one_epoch(self, train_loader, epoch) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter('Loss', ':6.4f')
        acc_meter = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(len(train_loader), [loss_meter, acc_meter], prefix=f'Train Epoch: {epoch}')
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch['x'].to(self.device).float()
            x = rearrange(x, 'b n c t -> (n b) c t')
            with autocast():
                features = self.model(x)
                logits, targets = self.info_nce_loss(features)
                loss = self.criterion(logits, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # update meters
            total_loss += loss.item()
            loss_meter.update(loss.item(), x.size(0))
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            acc_meter.update(acc, x.size(0))
            if batch_idx % 10 == 0:
                progress.display(batch_idx)

        return total_loss / len(train_loader)

    def validate(self, val_loader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        acc_meter = AverageMeter('Val Acc', ':6.2f')
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device).float()
                x = rearrange(x, 'b n c t -> (n b) c t')
                features = self.model(x)
                logits, targets = self.info_nce_loss(features)
                preds = logits.argmax(dim=1)
                acc = (preds == targets).float().mean().item()
                acc_meter.update(acc, x.size(0))
        print(f'Validation Accuracy: {acc_meter.avg:.4f}')
        return acc_meter.avg

    def train(self, train_loader, val_loader, ds_train_loader, ds_val_loader, online_evaluator):
        """Full training loop with early stopping and online evaluation."""
        os.makedirs(f'exp_log/{self.exp_name}', exist_ok=True)
        patience, no_improve = 5, 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            if epoch % 10 == 0:
                score = online_evaluator.online_train(self.model, ds_train_loader, ds_val_loader)
                if score > self.best_score:
                    save_checkpoint(self.model.encoder, self.optimizer, epoch, train_loss,
                                    f'exp_log/{self.exp_name}/best.pth')
                    self.best_score = score
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            val_acc = self.validate(val_loader)
        # Save last checkpoint
        save_checkpoint(self.model.encoder, self.optimizer, epoch, train_loss,
                        f'exp_log/{self.exp_name}/last.pth')


if __name__ == '__main__':
    # Example instantiation
    from models import resnet1d18
    from utils.utils import get_dummy_loaders, DummyEvaluator

    config = {
        'model': {'backbone_fn': 'resnet1d18'},
        'training': {'epochs': 100, 'batch_size': 64, 'temp': 0.07},
        'exp': 'simclr_ecg',
        'task': 'ecg'
    }
    model = resnet1d18(num_classes=4, input_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    trainer = SimCLRTrainer(config, optimizer, scheduler, model)
    train_loader, val_loader = get_dummy_loaders()
    ds_train_loader, ds_val_loader = get_dummy_loaders()
    evaluator = DummyEvaluator()
    trainer.train(train_loader, val_loader, ds_train_loader, ds_val_loader, evaluator)