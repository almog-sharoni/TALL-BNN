#!/usr/bin/env python3
"""
Training script for BNN on MNIST dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
from tqdm import tqdm

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow


def get_mnist_loaders(batch_size=128, data_dir='./data'):
    """Get MNIST train and test data loaders"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # Enhanced data augmentation for better generalization
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),

        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization  

    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=test_transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()  # Sets model to training mode (using floating point operations)
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Convert binary outputs to logits for cross-entropy
        # BNN outputs are {-1, +1}, convert to proper logits
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            # If outputs are already float (from training mode), use directly
            loss = criterion(output, target)
        else:
            # Convert binary to float for loss computation
            output_float = output.float()
            loss = criterion(output_float, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device, use_tall=False, use_training_mode=False):
    """Evaluate the model on test set"""
    if use_training_mode:
        # Keep model in training mode but don't compute gradients
        model.train()  # Keep in training mode to match training behavior
    else:
        # Use evaluation mode (fully binarized)
        model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'backbone'):
                # Use TALL voting for inference
                output = model(data)  # Already returns class indices
                pred = output
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train BNN on MNIST')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--model', type=str, default='deep', choices=['deep', 'shallow'],
                        help='model architecture (default: deep)')
    parser.add_argument('--use-tall', action='store_true', default=False,
                        help='use TALL voting for inference')
    parser.add_argument('--tall-iter', type=int, default=30,
                        help='number of TALL iterations (default: 30)')
    parser.add_argument('--tall-flip-p', type=float, default=0.3,
                        help='TALL bit flip probability (default: 0.3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current model')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory to store MNIST data')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    
    # Create model
    print(f"Creating {args.model} BNN model...")
    if args.model == 'deep':
        backbone = build_cam4_deep(num_classes=10)
    else:
        backbone = build_cam4_shallow(num_classes=10)
    
    model = backbone.to(device)
    
    # Create TALL wrapper if requested
    if args.use_tall:
        print(f"Wrapping with TALL (iter={args.tall_iter}, flip_p={args.tall_flip_p})")
        tall_model = TALLClassifier(
            backbone, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer with better hyperparameters
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9,  # Increased momentum
        weight_decay=1e-4,
        nesterov=True  # Add Nesterov momentum
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        
        # Evaluate using the same model mode as in training
        start_time = time.time()
        if args.use_tall:
            test_acc = evaluate(tall_model, test_loader, device, use_tall=True, use_training_mode=True)
        else:
            test_acc = evaluate(model, test_loader, device, use_tall=False, use_training_mode=True)
        eval_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc (Training Mode): {test_acc:.2f}%  # Using same model mode as training")
        print(f"LR: {current_lr:.6f}")
        print(f"Time - Train: {train_time:.1f}s, Eval: {eval_time:.1f}s")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_model:
                model_name = f"bnn_{args.model}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'args': args
                }, model_name)
                print(f"New best model saved: {model_name} (Acc: {best_acc:.2f}%)")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Final evaluation with fully binarized model
    print("\nFinal evaluation with fully binarized model...")
    binary_acc = evaluate(model, test_loader, device, use_tall=False, use_training_mode=False)
    print(f"Fully Binarized Test Accuracy: {binary_acc:.2f}%")
    
    # Final evaluation with TALL if not used during training
    if not args.use_tall:
        print("\nEvaluating with TALL voting...")
        tall_model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
        tall_acc = evaluate(tall_model, test_loader, device, use_tall=True, use_training_mode=False)
        print(f"TALL Test Accuracy: {tall_acc:.2f}%")


if __name__ == '__main__':
    main()
