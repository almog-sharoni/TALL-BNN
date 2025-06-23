#!/usr/bin/env python3
"""
Training script for BNN on Google Speech Commands dataset using MFCC features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
import random
import numpy as np
import hashlib
import argparse
import time
from tqdm import tqdm

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow


# Define the keywords for the task
KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
BACKGROUND_NOISE = '_background_noise_'
SILENCE_LABEL = '_silence_'
UNKNOWN_LABEL = '_unknown_'

# Create label mapping
LABELS = [SILENCE_LABEL] + KEYWORDS + [UNKNOWN_LABEL]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}


def load_audio_file(file_path, target_sample_rate=16000):
    """Load audio file with fallback methods"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        return waveform, target_sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return silence as fallback
        return torch.zeros(1, target_sample_rate), target_sample_rate


class AudioAugmentations:
    def __init__(self, sample_rate=16000, training=True):
        self.sample_rate = sample_rate
        self.training = training
    
    def __call__(self, waveform):
        if not self.training:
            return waveform
            
        # Random time shifting (small amounts)
        if random.random() < 0.4:
            shift_amount = random.randint(-int(0.1 * self.sample_rate), int(0.1 * self.sample_rate))
            if shift_amount != 0:
                if shift_amount > 0:
                    waveform = F.pad(waveform, (shift_amount, 0))[:, :waveform.shape[1]]
                else:
                    waveform = F.pad(waveform, (0, -shift_amount))[:, -shift_amount:]
        
        # Random amplitude scaling
        if random.random() < 0.6:
            scale = random.uniform(0.8, 1.2)
            waveform = waveform * scale
        
        # Add background noise
        if random.random() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Random speed perturbation (time stretching)
        if random.random() < 0.2:
            speed_factor = random.uniform(0.95, 1.05)
            if speed_factor != 1.0:
                current_len = waveform.shape[1]
                new_len = int(current_len / speed_factor)
                if new_len > 0:
                    indices = torch.linspace(0, current_len - 1, new_len).long()
                    waveform = waveform[:, indices]
                    # Pad or truncate back to original length
                    if waveform.shape[1] < current_len:
                        pad_len = current_len - waveform.shape[1]
                        waveform = F.pad(waveform, (0, pad_len))
                    elif waveform.shape[1] > current_len:
                        waveform = waveform[:, :current_len]
        
        return waveform


class SpeechCommandsDataset(Dataset):
    def __init__(self, data_path, subset='training', transform=None, sample_rate=16000):
        self.data_path = data_path
        self.subset = subset
        self.transform = transform
        self.sample_rate = sample_rate
        
        # Load validation and test lists
        validation_list = self._load_list_file(os.path.join(data_path, 'validation_list.txt'))
        test_list = self._load_list_file(os.path.join(data_path, 'testing_list.txt'))
        
        self.data = []
        self._build_dataset(validation_list, test_list)
        
    def _load_list_file(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _build_dataset(self, validation_list, test_list):
        # First collect all audio files
        all_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.data_path)
                    
                    # Skip background noise files
                    if BACKGROUND_NOISE in rel_path:
                        continue
                    
                    # Extract label from directory name
                    label = os.path.basename(root)
                    
                    # Map to our label system
                    if label in KEYWORDS:
                        target_label = label
                    else:
                        target_label = UNKNOWN_LABEL
                    
                    all_files.append((file_path, LABEL_TO_IDX[target_label], rel_path))
        
        # If validation/test lists don't exist, create splits programmatically
        if len(validation_list) == 0 and len(test_list) == 0:
            # Create deterministic splits based on file hash for reproducibility
            for file_path, label_idx, rel_path in all_files:
                # Use file path hash to determine split
                file_hash = int(hashlib.md5(rel_path.encode()).hexdigest(), 16)
                split_val = file_hash % 100
                
                if split_val < 10:  # 10% for validation
                    if self.subset == 'validation':
                        self.data.append((file_path, label_idx))
                elif split_val < 20:  # 10% for test
                    if self.subset == 'testing':
                        self.data.append((file_path, label_idx))
                else:  # 80% for training
                    if self.subset == 'training':
                        self.data.append((file_path, label_idx))
        else:
            # Use provided validation/test lists
            for file_path, label_idx, rel_path in all_files:
                # Determine subset
                if rel_path in validation_list:
                    if self.subset != 'validation':
                        continue
                elif rel_path in test_list:
                    if self.subset != 'testing':
                        continue
                else:
                    if self.subset != 'training':
                        continue
                
                self.data.append((file_path, label_idx))
        
        # Add silence samples (randomly during training)
        if self.subset == 'training':
            num_silence = len(self.data) // 10  # 10% silence samples
            for _ in range(num_silence):
                self.data.append(('_silence_', LABEL_TO_IDX[SILENCE_LABEL]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data[idx][0] == '_silence_':
            # Generate silence
            waveform = torch.zeros(1, self.sample_rate)
        else:
            # Load audio file using robust loading function
            waveform, sample_rate = load_audio_file(self.data[idx][0], self.sample_rate)
        
        # Pad or truncate to 1 second
        target_length = self.sample_rate
        if waveform.shape[1] < target_length:
            # Pad with zeros
            pad_length = target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_length))
        elif waveform.shape[1] > target_length:
            # Random crop during training, center crop during testing
            if self.subset == 'training':
                start = random.randint(0, waveform.shape[1] - target_length)
            else:
                start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        
        label = self.data[idx][1]
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label


def get_transforms(training=True, sample_rate=16000, n_mfcc=13, n_fft=1024, hop_length=512, n_mels=128):
    """Get MFCC transforms for speech commands"""
    
    # MFCC transform - better for speech recognition
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'f_min': 0,
            'f_max': sample_rate // 2,
            'power': 2.0,
            'normalized': True
        }
    )
    
    # Delta and delta-delta features
    compute_deltas = torchaudio.transforms.ComputeDeltas()
    
    # Audio augmentations
    audio_aug = AudioAugmentations(sample_rate, training=training)
    
    # Spec augmentations for training
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=5)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
    
    def transform(waveform):
        # Apply audio augmentations
        if training:
            waveform = audio_aug(waveform)
        
        # Convert to MFCC features
        mfcc_features = mfcc_transform(waveform)
        
        # Add delta and delta-delta features
        delta_features = compute_deltas(mfcc_features)
        delta_delta_features = compute_deltas(delta_features)
        
        # Concatenate MFCC, delta, and delta-delta features
        features = torch.cat([mfcc_features, delta_features, delta_delta_features], dim=1)
        
        # Apply spectrogram augmentations during training
        if training:
            if random.random() < 0.3:
                features = freq_mask(features)
            if random.random() < 0.3:
                features = time_mask(features)
        
        # Normalization using mean and std
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Flatten for BNN (features are [n_mfcc*3, time_frames])
        features = features.flatten()
        
        return features
    
    return transform


def calculate_mfcc_features(sample_rate=16000, n_mfcc=13, n_fft=1024, hop_length=512):
    """Calculate the number of MFCC features for given parameters"""
    # For 1 second audio at given sample rate
    # Time frames calculation for MFCC
    time_frames = (sample_rate - n_fft) // hop_length + 1
    # MFCC features: n_mfcc * 3 (original + delta + delta-delta)
    total_features = n_mfcc * 3 * time_frames
    return total_features, time_frames


# Create model adapted for speech commands MFCC features
class SpeechBinaryMLP(BinaryMLP):
    """BinaryMLP adapted for speech commands with MFCC features"""
    def __init__(self,
                 sample_rate=16000,
                 n_mfcc=13,
                 n_fft=1024,
                 hop_length=512,
                 hidden_sizes=(4096, 4096, 128),
                 num_classes=12,
                 thresholds=None):
        
        # Calculate input features based on MFCC parameters
        in_features, time_frames = calculate_mfcc_features(sample_rate, n_mfcc, n_fft, hop_length)
        print(f"Expected input features: {in_features} (time_frames: {time_frames})")
        
        super().__init__(
            in_features=in_features,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            thresholds=thresholds
        )


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # For training, use standard cross-entropy loss with logits
        loss = criterion(output, target)
        
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


def evaluate(model, test_loader, device, use_tall=False):
    """Evaluate the model on test set"""
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
    parser = argparse.ArgumentParser(description='Train BNN on Google Speech Commands')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    parser.add_argument('--data-dir', type=str, default='./data/speech_commands',
                        help='directory with speech commands data')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='sample rate for audio (default: 16000)')
    parser.add_argument('--n-mfcc', type=int, default=13,
                        help='number of MFCC coefficients (default: 13)')
    parser.add_argument('--n-fft', type=int, default=1024,
                        help='number of FFT points (default: 1024)')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='hop length for STFT (default: 512)')
    parser.add_argument('--n-mels', type=int, default=128,
                        help='number of mel filter banks (default: 128)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Check if dataset path exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset path does not exist: {args.data_dir}")
        print("Please ensure the Google Speech Commands dataset is downloaded.")
        exit(1)
    
    # Load data
    print("Loading Google Speech Commands dataset...")
    train_transform = get_transforms(
        training=True, 
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    test_transform = get_transforms(
        training=False,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    train_dataset = SpeechCommandsDataset(
        args.data_dir, subset='training', transform=train_transform, sample_rate=args.sample_rate
    )
    val_dataset = SpeechCommandsDataset(
        args.data_dir, subset='validation', transform=test_transform, sample_rate=args.sample_rate
    )
    test_dataset = SpeechCommandsDataset(
        args.data_dir, subset='testing', transform=test_transform, sample_rate=args.sample_rate
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )
    
    # Create model
    print(f"Creating {args.model} BNN model for speech commands...")
    if args.model == 'deep':
        model = SpeechBinaryMLP(
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            hidden_sizes=(4096, 4096, 128),
            num_classes=len(LABELS)
        )
    else:
        model = SpeechBinaryMLP(
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            hidden_sizes=(128,),
            num_classes=len(LABELS)
        )
    
    model = model.to(device)
    
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
    
    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
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
        
        # Evaluate on validation set
        start_time = time.time()
        if args.use_tall:
            val_acc = evaluate(tall_model, val_loader, device, use_tall=True)
        else:
            val_acc = evaluate(model, val_loader, device, use_tall=False)
        eval_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        print(f"Time - Train: {train_time:.1f}s, Eval: {eval_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_model:
                model_name = f"speech_bnn_{args.model}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'args': args
                }, model_name)
                print(f"New best model saved: {model_name} (Val Acc: {best_acc:.2f}%)")
    
    # Final test evaluation
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    print("\nFinal test evaluation...")
    if args.use_tall:
        test_acc = evaluate(tall_model, test_loader, device, use_tall=True)
    else:
        test_acc = evaluate(model, test_loader, device, use_tall=False)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Final evaluation with TALL if not used during training
    if not args.use_tall:
        print("\nEvaluating with TALL voting...")
        tall_model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
        tall_acc = evaluate(tall_model, test_loader, device, use_tall=True)
        print(f"TALL Test Accuracy: {tall_acc:.2f}%")


if __name__ == '__main__':
    main()
