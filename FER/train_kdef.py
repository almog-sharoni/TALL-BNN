#!/usr/bin/env python3
"""
Training script for BNN on KDEF (Karolinska Directed Emotional Faces) dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import argparse
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import json

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow, BinarizeLinear, BinaryActivation, binarize


class KDEFDataset(Dataset):
    """
    KDEF Dataset loader with proper stratified subject-independent splitting
    
    KDEF contains images with naming convention:
    [MODEL][EMOTION][ANGLE].JPG
    where:
    - MODEL: AM01-AM35 (male), AF01-AF35 (female), BM01-BM35, BF01-BF35
    - EMOTION: AN (angry), DI (disgusted), AF (afraid), HA (happy), 
               NE (neutral), SA (sad), SU (surprised)
    - ANGLE: S (straight), FL (45° left), FR (45° right), HL (90° left), HR (90° right)
    """
    
    def __init__(self, image_paths, labels, subjects, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of emotion labels
            subjects (list): List of subject IDs
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.subjects = subjects
        self.transform = transform
        
        # Emotion mapping - consistent with KDEF naming convention
        self.emotion_to_idx = {
            'AN': 0,  # angry
            'DI': 1,  # disgusted  
            'AF': 2,  # afraid
            'HA': 3,  # happy
            'NE': 4,  # neutral
            'SA': 5,  # sad
            'SU': 6   # surprised
        }
        
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        self.emotion_names = ['angry', 'disgusted', 'afraid', 'happy', 'neutral', 'sad', 'surprised']
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to grayscale
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @classmethod
    def parse_filename(cls, filename):
        """
        Parse KDEF filename to extract subject, emotion, and angle
        Format: SUBJECTEMOTION[ANGLE].JPG
        Example: AF01AFFL.JPG -> Subject: AF01, Emotion: AF (Afraid), Angle: FL
        """
        basename = os.path.splitext(filename)[0]
        if len(basename) < 6:
            return None, None, None
        
        subject = basename[:4]  # First 4 chars: subject ID
        emotion = basename[4:6]  # Next 2 chars: emotion code
        angle = basename[6:] if len(basename) > 6 else "S"  # Remaining: angle/pose
        
        return subject, emotion, angle
    
    @classmethod
    def load_dataset_info(cls, data_dir, angles=['S']):
        """
        Load all KDEF images and extract metadata from the folder structure
        Returns lists of image paths, labels, subjects, and metadata
        """
        # Look for KDEF folder in the provided directory
        possible_paths = [
            os.path.join(data_dir, 'KDEF'),  # Direct KDEF folder under data
            os.path.join(data_dir, 'KDEF_and_AKDEF', 'KDEF'),  # Legacy path
            data_dir  # If data_dir itself is the KDEF folder
        ]
        
        kdef_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                # Check if this directory contains subject folders
                contents = os.listdir(path)
                subject_folders = [d for d in contents if os.path.isdir(os.path.join(path, d)) 
                                 and (d.startswith('AM') or d.startswith('AF') or d.startswith('BM') or d.startswith('BF'))]
                if subject_folders:
                    kdef_dir = path
                    break
        
        if kdef_dir is None:
            raise FileNotFoundError(f"KDEF directory not found. Checked paths: {possible_paths}")
        
        print(f"Found KDEF directory: {kdef_dir}")
        
        image_paths = []
        labels = []
        subjects = []
        metadata = []
        
        emotion_mapping = cls({}, [], [], None).emotion_to_idx
        emotion_names = cls({}, [], [], None).emotion_names
        
        print("Scanning KDEF dataset...")
        
        # Iterate through all subject directories
        for subject_dir in sorted(os.listdir(kdef_dir)):
            subject_path = os.path.join(kdef_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            
            # Skip if not a valid subject directory
            if not (subject_dir.startswith('AM') or subject_dir.startswith('AF') or 
                   subject_dir.startswith('BM') or subject_dir.startswith('BF')):
                continue
            
            # Process each image in the subject directory
            for filename in os.listdir(subject_path):
                if not filename.upper().endswith('.JPG'):
                    continue
                
                subject, emotion, angle = cls.parse_filename(filename)
                if subject is None or emotion not in emotion_mapping:
                    continue
                
                # Filter by desired angles
                if angle not in angles:
                    continue
                
                image_path = os.path.join(subject_path, filename)
                label = emotion_mapping[emotion]
                
                image_paths.append(image_path)
                labels.append(label)
                subjects.append(subject)
                metadata.append({
                    'filename': filename,
                    'subject': subject,
                    'emotion': emotion,
                    'emotion_name': emotion_names[label],
                    'angle': angle,
                    'label': label
                })
        
        print(f"Found {len(image_paths)} images from {len(set(subjects))} subjects")
        print(f"Emotion distribution:")
        emotion_counts = Counter(labels)
        for emotion_idx, count in sorted(emotion_counts.items()):
            print(f"  {emotion_names[emotion_idx]}: {count}")
        
        # Print subject distribution
        print(f"Subject distribution:")
        subject_counts = Counter(subjects)
        male_subjects = sum(1 for s in set(subjects) if s.startswith('AM') or s.startswith('BM'))
        female_subjects = sum(1 for s in set(subjects) if s.startswith('AF') or s.startswith('BF'))
        print(f"  Male subjects: {male_subjects}")
        print(f"  Female subjects: {female_subjects}")
        print(f"  Total unique subjects: {len(set(subjects))}")
        
        return image_paths, labels, subjects, metadata


def create_stratified_subject_splits(subjects, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Create train/val/test splits ensuring:
    1. Subject independence (no subject appears in multiple splits)
    2. Stratification by emotion (similar emotion distribution across splits)
    
    Args:
        subjects: List of subject IDs for each sample
        labels: List of emotion labels for each sample
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_indices, val_indices, test_indices
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create subject-level data
    unique_subjects = sorted(list(set(subjects)))
    subject_to_emotions = defaultdict(list)
    
    # Group labels by subject
    for i, (subject, label) in enumerate(zip(subjects, labels)):
        subject_to_emotions[subject].append(label)
    
    # Calculate dominant emotion for each subject (for stratification)
    subject_dominant_emotions = []
    for subject in unique_subjects:
        emotions = subject_to_emotions[subject]
        if emotions:
            dominant_emotion = Counter(emotions).most_common(1)[0][0]
        else:
            dominant_emotion = 0  # Default to first emotion if no data
        subject_dominant_emotions.append(dominant_emotion)
    
    # Convert to arrays for splitting
    subjects_array = np.array(unique_subjects)
    
    # First split: separate train from (val + test)
    train_subjects, temp_subjects, train_emotions, temp_emotions = train_test_split(
        subjects_array, 
        subject_dominant_emotions,
        test_size=val_ratio + test_ratio,
        stratify=subject_dominant_emotions,
        random_state=random_state
    )
    
    # Second split: separate val from test
    if val_ratio > 0 and test_ratio > 0:
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_subjects, test_subjects, _, _ = train_test_split(
            temp_subjects,
            temp_emotions,
            test_size=1 - val_test_ratio,
            stratify=temp_emotions,
            random_state=random_state
        )
    elif val_ratio > 0:
        val_subjects = temp_subjects
        test_subjects = np.array([])
    else:
        val_subjects = np.array([])
        test_subjects = temp_subjects
    
    # Convert back to sample indices
    train_indices = [i for i, subject in enumerate(subjects) if subject in train_subjects]
    val_indices = [i for i, subject in enumerate(subjects) if subject in val_subjects]
    test_indices = [i for i, subject in enumerate(subjects) if subject in test_subjects]
    
    # Verify no subject overlap
    train_subject_set = set(subjects[i] for i in train_indices)
    val_subject_set = set(subjects[i] for i in val_indices)
    test_subject_set = set(subjects[i] for i in test_indices)
    
    assert len(train_subject_set & val_subject_set) == 0, "Train and val sets share subjects"
    assert len(train_subject_set & test_subject_set) == 0, "Train and test sets share subjects"
    assert len(val_subject_set & test_subject_set) == 0, "Val and test sets share subjects"
    
    print(f"Subject-independent split:")
    print(f"  Train: {len(train_subject_set)} subjects, {len(train_indices)} images")
    print(f"  Val: {len(val_subject_set)} subjects, {len(val_indices)} images")
    print(f"  Test: {len(test_subject_set)} subjects, {len(test_indices)} images")
    
    # Print emotion distribution for each split
    emotion_names = ['angry', 'disgusted', 'afraid', 'happy', 'neutral', 'sad', 'surprised']
    for split_name, indices in [("Train", train_indices), ("Val", val_indices), ("Test", test_indices)]:
        if len(indices) > 0:
            split_labels = [labels[i] for i in indices]
            emotion_counts = Counter(split_labels)
            total = len(split_labels)
            print(f"\n{split_name} emotion distribution:")
            for emotion_idx in range(7):
                count = emotion_counts.get(emotion_idx, 0)
                percentage = 100 * count / total if total > 0 else 0
                emotion_name = emotion_names[emotion_idx]
                print(f"  {emotion_name}: {count} ({percentage:.1f}%)")
    
    return train_indices, val_indices, test_indices


def get_kdef_transforms(image_size=224, augment=True):
    """Get transforms for KDEF dataset"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.RandomCrop(image_size),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=10),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Only brightness and contrast for grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
        ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
    ])
    
    return train_transform, val_test_transform


def get_kdef_loaders(data_dir, batch_size=32, test_size=0.2, random_state=42, 
                     image_size=224, augment=True, angles=['S', 'FL', 'FR', 'HL', 'HR'], 
                     num_workers=4, subject_independent=False):
    """Get KDEF train and test data loaders
    
    Args:
        data_dir: Path to KDEF dataset
        batch_size: Batch size for data loaders
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        image_size: Size to resize images to
        augment: Whether to apply data augmentation to training set
        angles: List of camera angles to include
        num_workers: Number of worker processes for data loading
        subject_independent: If True, ensures subjects in train set don't appear in test set
    """
    
    # Load dataset information
    image_paths, labels, subjects, metadata = KDEFDataset.load_dataset_info(data_dir, angles=angles)
    
    if len(image_paths) == 0:
        raise ValueError(f"No KDEF images found in {data_dir}. Please check the path and file format.")
    
    if subject_independent:
        # Subject-independent stratified split
        unique_subjects = sorted(list(set(subjects)))
        subject_to_emotions = defaultdict(list)
        
        # Group labels by subject
        for i, (subject, label) in enumerate(zip(subjects, labels)):
            subject_to_emotions[subject].append(label)
        
        # Calculate dominant emotion for each subject (for stratification)
        subject_dominant_emotions = []
        for subject in unique_subjects:
            emotions = subject_to_emotions[subject]
            if emotions:
                dominant_emotion = Counter(emotions).most_common(1)[0][0]
            else:
                dominant_emotion = 0
            subject_dominant_emotions.append(dominant_emotion)
        
        # Split subjects into train/test
        train_subjects, test_subjects, _, _ = train_test_split(
            unique_subjects, 
            subject_dominant_emotions,
            test_size=test_size,
            stratify=subject_dominant_emotions,
            random_state=random_state
        )
        
        # Convert back to sample indices
        train_indices = [i for i, subject in enumerate(subjects) if subject in train_subjects]
        test_indices = [i for i, subject in enumerate(subjects) if subject in test_subjects]
        
        # Verify no subject overlap
        train_subject_set = set(subjects[i] for i in train_indices)
        test_subject_set = set(subjects[i] for i in test_indices)
        assert len(train_subject_set & test_subject_set) == 0, "Train and test sets share subjects"
        
        print(f"Subject-independent split:")
        print(f"  Train: {len(train_subject_set)} subjects, {len(train_indices)} images")
        print(f"  Test: {len(test_subject_set)} subjects, {len(test_indices)} images")
    else:
        # Standard stratified split (not subject-independent)
        train_indices, test_indices = train_test_split(
            range(len(image_paths)), 
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        # Calculate subject overlap for informational purposes
        train_subject_set = set(subjects[i] for i in train_indices)
        test_subject_set = set(subjects[i] for i in test_indices)
        overlap_subjects = train_subject_set & test_subject_set
        
        print(f"Standard stratified split (not subject-independent):")
        print(f"  Train: {len(train_indices)} images")
        print(f"  Test: {len(test_indices)} images")
        print(f"  Subject overlap: {len(overlap_subjects)} subjects appear in both train and test sets")
    
    # Get transforms
    train_transform, test_transform = get_kdef_transforms(image_size, augment)
    
    # Create datasets
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_subjects_list = [subjects[i] for i in train_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_subjects_list = [subjects[i] for i in test_indices]
    
    train_dataset = KDEFDataset(train_paths, train_labels, train_subjects_list, transform=train_transform)
    test_dataset = KDEFDataset(test_paths, test_labels, test_subjects_list, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


# class ImageBinaryMLP(nn.Module):
#     """
#     BNN MLP adapted for image inputs with automatic flattening
#     """
#     def __init__(self, image_size=224, hidden_sizes=(4096, 512, 128), num_classes=7):
#         super().__init__()
        
#         # Calculate input features for grayscale images: 1 * height * width
#         in_features = 1 * image_size * image_size
        
#         layers = []
#         prev_size = in_features
        
#         # Add flatten layer for images
#         self.flatten = nn.Flatten()
        
#         # Build hidden layers
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 BinarizeLinear(prev_size, hidden_size),
#                 nn.BatchNorm1d(hidden_size),
#                 BinaryActivation()
#             ])
#             prev_size = hidden_size
        
#         # Store hidden layers and final layer separately for TALL compatibility
#         self.hidden = nn.Sequential(*layers)
#         self.fc_out = BinarizeLinear(prev_size, num_classes)
    
#     def forward(self, x):
#         # Flatten image input
#         x = self.flatten(x)
#         x = self.hidden(x)
#         if self.training:
#             return self.fc_out(x)  # FP32 logits during training
#         else:
#             return binarize(self.fc_out(x))  # Binary outputs during inference
    
#     def features(self, x):
#         """Expose last-hidden activations for TALL compatibility"""
#         x = self.flatten(x)
#         return self.hidden(x)


def build_kdef_model(model_type='deep', image_size=224, num_classes=7):
    """Build BNN model adapted for KDEF dataset - using same architecture as MNIST"""
    in_features = image_size * image_size  # Flattened input size
    
    if model_type == 'deep':
        model = build_cam4_deep(num_classes=num_classes, in_features=in_features)
    elif model_type == 'shallow':
        model = build_cam4_shallow(num_classes=num_classes, in_features=in_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


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
    parser = argparse.ArgumentParser(description='Train BNN on KDEF dataset with subject-independent stratified splits')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory containing KDEF data (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--model', type=str, default='deep', choices=['deep', 'shallow'],
                        help='model architecture (default: deep)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='input image size (default: 224)')
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
    parser.add_argument('--no-augment', action='store_true', default=False,
                        help='disable data augmentation')
    parser.add_argument('--angles', nargs='+', default=['S', 'FL', 'FR', 'HL', 'HR'],
                        choices=['S', 'FL', 'FR', 'HL', 'HR'],
                        help='which camera angles to include (default: all angles)')
    parser.add_argument('--subject-independent', action='store_true', default=False,
                        help='use subject-independent stratified splits (default: False)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading KDEF dataset...")
    try:
        train_loader, test_loader = get_kdef_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            test_size=0.2,
            image_size=args.image_size,
            augment=not args.no_augment,
            angles=args.angles,
            num_workers=args.num_workers,
            subject_independent=args.subject_independent
        )
    except Exception as e:
        print(f"Error loading KDEF dataset: {e}")
        print("Please ensure the KDEF dataset is properly extracted and the path is correct.")
        return
    
    # Create model
    print(f"Creating {args.model} BNN model for KDEF (7 emotions)...")
    model = build_kdef_model(
        model_type=args.model,
        image_size=args.image_size,
        num_classes=7
    ).to(device)
    
    # Create TALL wrapper if requested
    if args.use_tall:
        print(f"Wrapping with TALL (iter={args.tall_iter}, flip_p={args.tall_flip_p})")
        tall_model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
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
    
    # Emotion names for reporting
    emotion_names = ['angry', 'disgusted', 'afraid', 'happy', 'neutral', 'sad', 'surprised']
    
    # Training loop
    print("Starting training...")
    best_acc = 0.0

    # Clear CUDA cache to avoid memory issues
    torch.cuda.empty_cache()
    
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
                model_name = f"kdef_bnn_{args.model}_best.pth"
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
