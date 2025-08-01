import torch
from torch.optim.lr_scheduler import LambdaLR

from datasets.dataloader import SimCLR_Dataloader, Downstream_Dataloader
from networks import *
from trainers import *

import argparse
import yaml
import pprint
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script with SSL")
    parser.add_argument('--config', type=str, default='configs/cinc17/config_ssl_simclr.yaml', help="Path to the configuration file")
    parser.add_argument('--lr', type=float, help="Learning rate override")
    parser.add_argument('--exp', type=str, help="Learning rate override")
    parser.add_argument('--batch_size', type=int, help="Batch size override")
    parser.add_argument('--data_path', type=str, help="Data path override")
    parser.add_argument('--epochs', type=int, help="Number of epochs override")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'posthoc', 'transfer'], help="Operation mode")
    parser.add_argument('--backbone', type=str, help="Backbone override")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def override_config(args, config):
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.exp:
        config['training']['exp'] = args.exp
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.data_path:
        config['data']['data_path'] = args.data_path
    return config

# Define the warm-up and decay schedule
def lr_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

def main():
    # Parse command-line arguments and load configuration
    args = parse_arguments()
    config = load_config(args.config)
    config = override_config(args, config)

    # Print all configurations
    print("Current Configuration:")
    pprint.pprint(config)

    # Create data loaders
    print("Preparing Pretraining:")
    train_loader, val_loader, test_loader = SimCLR_Dataloader(data_path=config['data']['data_path'],
                                                  batch_size=config['training']['batch_size'],
                                                  fold_num=config['data']['group_number'],
                                                  input_length=config['training']['input_length'],)
    print("Preparing Downstream:")
    ds_train_loader, ds_val_loader, ds_test_loader = Downstream_Dataloader(data_path=config['data']['data_path'])

    # Print dataset lengths
    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Test dataset length: {len(test_loader.dataset)}")

    # Load the ResNet1D model
    backbone = SimCLR(**config['model'])

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, config['training']['warmup_steps']*len(train_loader),
                                                                     config['training']['epochs']*len(train_loader)))

    # get the trainer
    trainers = {'simclr': SimCLRTrainer,
                'domainssl': DomainSSLTrainer,}

    trainer = trainers[config['method']](config=config, optimizer=optimizer, scheduler=scheduler, model=backbone)

    online_evaluator = OnlineEvaluator(**{**config['evaluation'], **config['model']})

    if args.mode == 'train':
        trainer.train(train_loader, val_loader, ds_train_loader, ds_val_loader, online_evaluator)


if __name__ == "__main__":
    set_seed(42)
    main()