#!/usr/bin/env python3
"""
Training script for BNN on FER2013 (Facial Expression Recognition) dataset

FER2013 contains grayscale 48x48 pixel face images with 7 emotion classes:
0 - Angry, 1 - Disgust, 2 - Fear, 3 - Happy, 4 - Sad, 5 - Surprise, 6 - Neutral
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow, BinarizeLinear, BinaryActivation, binarize


import os
from torchvision.datasets import ImageFolder

def check_fer2013_folder_structure(data_dir):
    """Check if FER2013 folder structure exists"""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Expected train and test folders not found in {data_dir}")
        print("Expected structure:")
        print("data/FER2013/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   ├── happy/")
        print("│   ├── sad/")
        print("│   ├── surprise/")
        print("│   └── neutral/")
        print("└── test/")
        print("    ├── angry/")
        print("    ├── disgust/")
        print("    ├── fear/")
        print("    ├── happy/")
        print("    ├── sad/")
        print("    ├── surprise/")
        print("    └── neutral/")
        return False
    
    print(f"Found FER2013 dataset structure at: {data_dir}")
    return True


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset loader for folder structure
    
    The FER2013 dataset contains grayscale images of faces with 7 emotions:
    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    
    Expected folder structure:
    data_dir/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Path to the FER2013 dataset directory
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Mapping from folder names to emotion indices
        self.emotion_to_idx = {
            'angry': 0,
            'disgust': 1, 
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }
        
        # Reverse mapping for display
        self.emotion_names = {v: k for k, v in self.emotion_to_idx.items()}
        
        # Load image paths and labels
        self.samples = []
        split_dir = os.path.join(data_dir, split)
        
        for emotion_name, emotion_idx in self.emotion_to_idx.items():
            emotion_dir = os.path.join(split_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_file)
                        self.samples.append((img_path, emotion_idx))
        
        print(f"Loaded {len(self.samples)} {split} samples")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = Counter([label for _, label in self.samples])
        print("Class distribution:")
        for emotion_idx in sorted(class_counts.keys()):
            emotion_name = self.emotion_names[emotion_idx]
            count = class_counts[emotion_idx]
            print(f"  {emotion_idx} ({emotion_name}): {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, emotion
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        class_counts = Counter([label for _, label in self.samples])
        total_samples = len(self.samples)
        num_classes = len(self.emotion_to_idx)
        
        weights = []
        for i in range(num_classes):
            if i in class_counts:
                weight = total_samples / (num_classes * class_counts[i])
                weights.append(weight)
            else:
                weights.append(0.0)
        
        return torch.FloatTensor(weights)

class BinarizeTransform:
    """Custom transform to binarize images using the BNN binarize function"""
    def __init__(self, normalize_first=True):
        self.normalize_first = normalize_first
    
    def __call__(self, tensor):
        """Apply binarization to the tensor"""
        # Import binarize function from BNN_model
        from BNN_model import binarize
        
        if self.normalize_first:
            # Normalize to [-1, 1] range before binarizing
            # Assuming tensor is in [0, 1] range from ToTensor()
            tensor = tensor * 2.0 - 1.0
            threshold = tensor.median()
            tensor = tensor - threshold

        
        return binarize(tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(normalize_first={self.normalize_first})'


def get_fer_loaders(data_dir, batch_size=128, data_augmentation=True, balanced_sampling=False):
    """Get FER2013 train, validation and test data loaders"""
    
    # Define transforms
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            BinarizeTransform(),
            # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            BinarizeTransform(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        BinarizeTransform(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets
    train_dataset = FER2013Dataset(data_dir, split='train', transform=train_transform)
    test_dataset = FER2013Dataset(data_dir, split='test', transform=test_transform)
    
    # Use test set as validation set (or split train set if needed)
    val_dataset = test_dataset
    
    # Create samplers for balanced training if requested
    train_sampler = None
    if balanced_sampling:
        # Calculate sample weights for balanced sampling
        class_counts = Counter([label for _, label in train_dataset.samples])
        sample_weights = []
        
        for _, emotion in train_dataset.samples:
            weight = 1.0 / class_counts[emotion]
            sample_weights.append(weight)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, # TODO: Use sampler instead
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, use_class_weights=False, class_weights=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Convert binary outputs to logits for cross-entropy
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            loss = criterion(output, target)
        else:
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


def evaluate(model, test_loader, device, use_tall=False, use_training_mode=False, class_names=None):
    """Evaluate the model on test set with detailed metrics"""
    if use_training_mode:
        model.train()
    else:
        model.eval()
    
    correct = 0
    total = 0
    class_correct = np.zeros(7)  # 7 emotion classes
    class_total = np.zeros(7)
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'backbone'):
                output = model(data)
                pred = output
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Per-class accuracy
            for i in range(7):
                class_mask = (target == i)
                if class_mask.sum() > 0:
                    class_correct[i] += pred[class_mask].eq(target[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Print detailed results
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\nPer-class accuracy:")
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    for i in range(7):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"  {i} ({emotion_names[i]}): {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"  {i} ({emotion_names[i]}): N/A (no samples)")
    
    return accuracy, class_correct / np.maximum(class_total, 1)


def build_fer_model(model_type='deep', num_classes=7):
    """Build BNN model for FER (48x48 input images)"""
    input_features = 48 * 48  # FER2013 images are 48x48
    
    if model_type == 'deep':
        return BinaryMLP(
            in_features=input_features,
            hidden_sizes=(4096, 4096, 128),
            num_classes=num_classes
        )
    elif model_type == 'shallow':
        return BinaryMLP(
            in_features=input_features,
            hidden_sizes=(512,),
            num_classes=num_classes
        )
    elif model_type == 'medium':
        return BinaryMLP(
            in_features=input_features,
            hidden_sizes=(2048, 256),
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_training_history(history, filename):
    """Save training history to JSON file"""
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history, save_path=None):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    # Best accuracy over time
    best_acc = []
    best_so_far = 0
    for acc in history['val_acc']:
        if acc > best_so_far:
            best_so_far = acc
        best_acc.append(best_so_far)
    
    ax4.plot(epochs, best_acc, 'purple', label='Best Validation Accuracy')
    ax4.set_title('Best Validation Accuracy Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def visualize_train_transforms(data_dir, batch_size=8, data_augmentation=True):
    """Visualize a batch of images from the train loader with transforms applied."""
    import matplotlib.pyplot as plt
    train_loader, _, _ = get_fer_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        data_augmentation=data_augmentation,
        balanced_sampling=False
    )
    images, labels = next(iter(train_loader))
    # Unnormalize images for display
    images = images * 0.5 + 0.5  # from [-1,1] to [0,1]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(8, 4))
    plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.title('Sample Train Images with Transforms')
    plt.axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train BNN on FER2013 dataset')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
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
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='directory to save outputs (default: ./outputs)')
    parser.add_argument('--data-augmentation', action='store_true', default=True,
                        help='use data augmentation (default: True)')
    parser.add_argument('--balanced-sampling', action='store_true', default=True,
                        help='use balanced sampling for training')
    parser.add_argument('--class-weights', action='store_true', default=False,
                        help='use class weights in loss function')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading FER2013 dataset...")
    # Check if FER2013 folder structure exists
    data_dir = os.path.join("../data", "FER2013")
    if not check_fer2013_folder_structure(data_dir):
        print("Please ensure FER2013 dataset is properly downloaded with train/test folder structure.")
        return
    
    train_loader, val_loader, test_loader = get_fer_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        data_augmentation=args.data_augmentation,
        balanced_sampling=args.balanced_sampling
    )
    
    # Create model
    print(f"Creating {args.model} BNN model for FER2013...")
    backbone = build_fer_model(model_type=args.model, num_classes=7)
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
    
    # Setup optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=3e-2, weight_decay=1e-5)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Get class weights if requested
    class_weights = None
    if args.class_weights:
        # Calculate class weights from training data
        train_dataset_temp = FER2013Dataset(args.csv_file, usage_filter='Training')
        class_weights = train_dataset_temp.get_class_weights()
        print(f"Using class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_class_weights=args.class_weights, class_weights=class_weights
        )
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        if args.use_tall:
            val_acc, _ = evaluate(tall_model, val_loader, device, use_tall=True, use_training_mode=False)
        else:
            val_acc, _ = evaluate(model, val_loader, device, use_tall=False, use_training_mode=False)
        
        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader)
        
        eval_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        print(f"Time - Train: {train_time:.1f}s, Eval: {eval_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Test on test set when we have a new best validation model
            if args.use_tall:
                test_acc, _ = evaluate(tall_model, test_loader, device, use_tall=True, use_training_mode=False)
            else:
                test_acc, _ = evaluate(model, test_loader, device, use_tall=False, use_training_mode=False)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            if args.save_model:
                model_name = os.path.join(args.output_dir, f"fer_{args.model}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                    'args': args,
                    'history': history
                }, model_name)
                print(f"New best model saved: {model_name} (Val: {best_val_acc:.2f}%, Test: {test_acc:.2f}%)")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"fer_{args.model}_history.json")
    save_training_history(history, history_path)
    
    # Plot training curves
    curves_path = os.path.join(args.output_dir, f"fer_{args.model}_training_curves.png")
    plot_training_curves(history, curves_path)
    
    # Final evaluation with fully binarized model
    print("\nFinal evaluation with fully binarized model...")
    binary_acc, _ = evaluate(model, test_loader, device, use_tall=False, use_training_mode=False)
    print(f"Fully Binarized Test Accuracy: {binary_acc:.2f}%")
    
    # Final evaluation with TALL if not used during training
    if not args.use_tall:
        print("\nEvaluating with TALL voting...")
        tall_model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
        tall_acc, _ = evaluate(tall_model, test_loader, device, use_tall=True, use_training_mode=False)
        print(f"TALL Test Accuracy: {tall_acc:.2f}%")


if __name__ == '__main__':
    main()
