#!/usr/bin/env python3
"""
Demo with layer-shared (recurrent) ResNet using streaming normalization.

This script trains a ResNet with shared residual blocks per stage using
streaming normalization instead of standard batch normalization.
Supports MNIST, CIFAR-10, and CIFAR-100 datasets.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
import json
import sys

# Add parent directory to path to allow importing snorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snorm.streaming_norm import StreamingNorm2d
from snorm.architectures import (
    PlainMLP, 
    PlainMLPShare, 
    ResidualBlock, 
    ResNetShare,
)

# Default experiment directory
EXPERIMENT_DIR = "snorm-resnet-experiment"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def set_weight_update_detectors(model: nn.Module, value: float = 1.0):
#     """Set all weight_update_detector parameters in streaming modules to a value."""
#     for module in model.modules():
#         if hasattr(module, 'weight_update_detector'):
#             with torch.no_grad():
#                 module.weight_update_detector.fill_(value)


def print_streaming_debug_info(model: nn.Module):
    """Print debug info from streaming operations (mean of forward and backward statistics)."""
    for name, module in model.named_modules():
        if hasattr(module, 'streaming_mean') or hasattr(module, 'streaming_std'):
            # Check if it's a StreamingNorm layer
            if hasattr(module, 'streaming_mean'):
                streaming_mean = module.streaming_mean
                streaming_std = module.streaming_std
                
                # Get forward statistics
                if streaming_mean.long_term_running_avg_f is not None:
                    mean_f_mean = streaming_mean.long_term_running_avg_f.mean().item()
                    mean_f_std = streaming_mean.long_term_running_avg_f.std().item()
                else:
                    mean_f_mean = mean_f_std = 0.0
                
                if streaming_std.long_term_running_avg_f is not None:
                    std_f_mean = streaming_std.long_term_running_avg_f.mean().item()
                    std_f_std = streaming_std.long_term_running_avg_f.std().item()
                else:
                    std_f_mean = std_f_std = 0.0
                
                # Get backward statistics
                if streaming_mean.long_term_running_avg_b is not None:
                    mean_b_mean = streaming_mean.long_term_running_avg_b.mean().item()
                    mean_b_std = streaming_mean.long_term_running_avg_b.std().item()
                else:
                    mean_b_mean = mean_b_std = 0.0
                
                if streaming_std.long_term_running_avg_b is not None:
                    std_b_mean = streaming_std.long_term_running_avg_b.mean().item()
                    std_b_std = streaming_std.long_term_running_avg_b.std().item()
                else:
                    std_b_mean = std_b_std = 0.0
                
                print(f"  {name}:")
                print(f"    Forward - mean: {mean_f_mean:.6f}±{mean_f_std:.6f}, std: {std_f_mean:.6f}±{std_f_std:.6f}")
                print(f"    Backward - mean: {mean_b_mean:.6f}±{mean_b_std:.6f}, std: {std_b_mean:.6f}±{std_b_std:.6f}")


# ResidualBlock and ResNetShare are now imported from esllm2.architectures


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, print_freq=100, debug_streaming=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Note: weight_update_detector automatically becomes nonzero after optimizer.step()
        # because it gets a constant gradient of 1.0 in backward pass
        # After optimizer.step() with lr=0.1, it becomes: 0 - 0.1 * 1.0 = -0.1 (nonzero)
        # This negative value (nonzero) will trigger the update check in the next forward pass
        # The forward pass will reset it to 0 after processing
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % print_freq == 0:
            print(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            if debug_streaming:
                print(f'  Streaming debug info:')
                print_streaming_debug_info(model)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(
        description='Demo with layer-shared ResNet using streaming normalization'
    )
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use: mnist, cifar10, or cifar100')
    parser.add_argument('--data_cache_dir', type=str, default='./data',
                       help='Directory to cache/download datasets (default: ./data)')
    parser.add_argument('--experiment_dir', type=str, default=EXPERIMENT_DIR,
                       help=f'Directory to save run results (default: {EXPERIMENT_DIR}). '
                            'Each run creates a subdirectory with timestamp: experiment_dir/run_YYYYMMDD_HHMMSS')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cosine'], help='LR schedule')
    parser.add_argument('--num_layers', type=int, nargs='+', default=[3, 5, 4, 7], 
                       help='Number of layers per stage (for resnet_share) or number of layers (for MLP architectures)')
    parser.add_argument('--feat_num', type=int, nargs='+', default=[32, 64, 128, 256], help='Features per stage')
    parser.add_argument('--block_type', type=str, default='simple', choices=['simple', 'tiny_resnet2', 'basic'], 
                       help='Block type: simple (one 3x3 conv), tiny_resnet2 (depthwise+1x1), basic (two 3x3 convs)')
    parser.add_argument('--norm_type', type=str, default=None, 
                       choices=[None, 'none', 'snorm', 'batchnorm', 'layernorm', 'groupnorm'],
                       help='Normalization type: none (no norm), snorm (streaming norm), batchnorm, layernorm, groupnorm')
    parser.add_argument('--no_streaming_norm', action='store_true', help='Disable streaming normalization (deprecated, use --norm_type instead)')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections')
    parser.add_argument('--decay_factor_f', type=float, default=0.7, help='Decay factor for forward streaming (default: 0.7)')
    parser.add_argument('--decay_factor_b', type=float, default=0.7, help='Decay factor for backward streaming (default: 0.7)')
    parser.add_argument('--snorm_alpha', type=float, nargs=3, default=[0.5, 0.5, 0.0],
                       help='Weight coefficients for forward output [long_term, short_term, current] (default: 0.5 0.5 0.0)')
    parser.add_argument('--snorm_beta', type=float, nargs=3, default=[0.6, 0.0, 0.4],
                       help='Weight coefficients for backward output [long_term, short_term, current] (default: 0.6 0.0 0.4)')
    parser.add_argument('--architecture', type=str, default=None, 
                       choices=['resnet_share', 'plain_mlp', 'plain_mlp_share'],
                       help='Architecture type: resnet_share, plain_mlp, or plain_mlp_share. '
                            'If not specified, will auto-detect based on other arguments.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for MLP architectures')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'silu', 'gelu', 'tanh'],
                       help='Activation function for MLP architectures (default: relu)')
    parser.add_argument('--print_freq', type=int, default=100, help='Print statistics every N batches (default: 100)')
    parser.add_argument('--debug_streaming', action='store_true', help='Print debug info from streaming operations')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Logging
    logging_dir = f"{args.experiment_dir}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(logging_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logging_dir)
    
    # Save config
    config = vars(args)
    with open(f"{logging_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Determine dataset parameters and create transforms
    if args.dataset == "mnist":
        # MNIST transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_cache_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_cache_dir, train=False, download=True, transform=transform_test
        )
        input_size = (28, 28)
        num_channels = 1
        num_classes = 10
        dataset_name = "MNIST"
        
    elif args.dataset == "cifar10":
        # CIFAR-10 transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_cache_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_cache_dir, train=False, download=True, transform=transform_test
        )
        input_size = (32, 32)
        num_channels = 3
        num_classes = 10
        dataset_name = "CIFAR-10"
        
    elif args.dataset == "cifar100":
        # CIFAR-100 transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_cache_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.data_cache_dir, train=False, download=True, transform=transform_test
        )
        input_size = (32, 32)
        num_channels = 3
        num_classes = 100
        dataset_name = "CIFAR-100"
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Dataset cache directory: {args.data_cache_dir}")
    print(f"  Train set: {len(train_dataset):,} samples")
    print(f"  Test set: {len(test_dataset):,} samples")
    print(f"  Total: {len(train_dataset) + len(test_dataset):,} samples")
    print(f"  Input size: {input_size} ({'28×28×1' if args.dataset == 'mnist' else '32×32×3'})")
    print(f"  Number of classes: {num_classes}")
    
    # Normalize norm_type argument (handle None vs "none")
    if args.norm_type is None or args.norm_type == "none":
        norm_type = None
    else:
        norm_type = args.norm_type
    
    # Handle deprecated --no_streaming_norm flag
    if args.no_streaming_norm and norm_type is None:
        norm_type = None
    elif args.no_streaming_norm and norm_type == "snorm":
        norm_type = None  # Override snorm if --no_streaming_norm is set
    
    if norm_type:
        print(f"  Normalization: {norm_type}")
    else:
        print(f"  Normalization: none")
    
    # Determine input size for MLP architectures
    if args.dataset == "mnist":
        input_size_flat = 28 * 28
    else:  # CIFAR-10 or CIFAR-100
        input_size_flat = 32 * 32 * 3
    
    # Auto-detect architecture if not specified
    if args.architecture is None:
        # If num_layers is a single value or short list (1-2 elements), likely MLP
        # Also check if hidden_size is explicitly set (not just default)
        if len(args.num_layers) <= 2:
            args.architecture = 'plain_mlp_share'
            print(f"Auto-detected architecture: {args.architecture} (based on num_layers={args.num_layers})")
        else:
            args.architecture = 'resnet_share'
            print(f"Auto-detected architecture: {args.architecture} (based on num_layers={args.num_layers})")
    
    # Create model based on architecture
    if args.architecture == "resnet_share":
        # Validate num_layers and feat_num for ResNet
        if len(args.num_layers) != len(args.feat_num):
            raise ValueError(f"For resnet_share, num_layers and feat_num must have the same length. "
                           f"Got num_layers={args.num_layers} (length {len(args.num_layers)}) and "
                           f"feat_num={args.feat_num} (length {len(args.feat_num)})")
        model = ResNetShare(
            input_size=input_size,
            num_channels=num_channels,
            num_classes=num_classes,
            num_layers=args.num_layers,
            feat_num=args.feat_num,
            norm_type=norm_type,
            block_type=args.block_type,
            use_residual=not args.no_residual,
            snorm_decay_factor_f=args.decay_factor_f,
            snorm_decay_factor_b=args.decay_factor_b,
            snorm_alpha=args.snorm_alpha,
            snorm_beta=args.snorm_beta,
        ).to(device)
    elif args.architecture == "plain_mlp":
        # For MLP, use first element of num_layers if it's a list, otherwise use the value directly
        mlp_num_layers = args.num_layers[0] if len(args.num_layers) > 0 else 3
        model = PlainMLP(
            input_size=input_size_flat,
            hidden_size=args.hidden_size,
            num_layers=mlp_num_layers,
            num_classes=num_classes,
            activation=args.activation,
            norm_type=norm_type,
            snorm_decay_factor_f=args.decay_factor_f,
            snorm_decay_factor_b=args.decay_factor_b,
            snorm_alpha=args.snorm_alpha,
            snorm_beta=args.snorm_beta,
        ).to(device)
    elif args.architecture == "plain_mlp_share":
        # For MLP, use first element of num_layers if it's a list, otherwise use the value directly
        mlp_num_layers = args.num_layers[0] if len(args.num_layers) > 0 else 3
        model = PlainMLPShare(
            input_size=input_size_flat,
            hidden_size=args.hidden_size,
            num_layers=mlp_num_layers,
            num_classes=num_classes,
            activation=args.activation,
            norm_type=norm_type,
            snorm_decay_factor_f=args.decay_factor_f,
            snorm_decay_factor_b=args.decay_factor_b,
            snorm_alpha=args.snorm_alpha,
            snorm_beta=args.snorm_beta,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model created: {total_params:,} total params, {trainable_params:,} trainable')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.lr_schedule == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    print('\nStarting training...')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                                           print_freq=args.print_freq, debug_streaming=args.debug_streaming)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'New best accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed. Best test accuracy: {best_acc:.2f}%')
    print(f'Results saved to: {logging_dir}')
    writer.close()


if __name__ == '__main__':
    main()

