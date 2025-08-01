import os
import time
import yaml
import torch
import numpy as np


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    filename: str) -> None:
    """
    Save model and optimizer states to a checkpoint file.

    Args:
        model: Model whose state_dict will be saved.
        optimizer: Optimizer whose state_dict will be saved.
        epoch: Current epoch number.
        loss: Loss value at saving time.
        filename: Path to the output checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to '{filename}'")


def save_config_file(model_checkpoints_folder: str,
                     args: dict) -> None:
    """
    Save training configuration as a YAML file in the checkpoint folder.

    Args:
        model_checkpoints_folder: Directory to save config.yml.
        args: Configuration dictionary to serialize.
    """
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    config_path = os.path.join(model_checkpoints_folder, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False)
    print(f"Config saved to '{config_path}'")


def accuracy(output: torch.Tensor,
             target: torch.Tensor,
             topk: tuple = (1,)) -> list:
    """
    Compute the top-k accuracy for the specified values of k.

    Args:
        output: Model predictions (logits) of shape [batch_size, num_classes].
        target: Ground-truth labels of shape [batch_size].
        topk: Tuple of integers specifying which top-k accuracies to compute.

    Returns:
        List of top-k accuracy percentages.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Obtain top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    Display progress for training or validation loops.
    """
    def __init__(self, num_batches: int, meters: list, prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class MeanTopKRecallMeter:
    """
    Computes and stores the average top-k recall metric.
    """
    def __init__(self, num_classes: int, k: int = 5,
                 name: str = 'MeanTopKRecall', fmt: str = ':.2f') -> None:
        self.num_classes = num_classes
        self.k = k
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.tps = np.zeros(self.num_classes, dtype=np.float32)
        self.nums = np.zeros(self.num_classes, dtype=np.int32)
        self.val = 0.0
        self.avg = 0.0
        self.count = 0

    def add(self, scores: np.ndarray, labels: np.ndarray) -> None:
        # scores shape: [batch_size, num_classes]
        # labels shape: [batch_size]
        # Compute top-k predictions
        topk_preds = np.argsort(scores, axis=1)[:, -self.k:]
        for i, label in enumerate(labels):
            if label in topk_preds[i]:
                self.tps[label] += 1
            self.nums[label] += 1

        # Update current and average
        recalls = (self.tps / np.maximum(self.nums, 1))[self.nums > 0]
        self.val = recalls.mean() * 100 if recalls.size > 0 else 0.0
        self.count += 1
        self.avg = ((self.avg * (self.count - 1)) + self.val) / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
