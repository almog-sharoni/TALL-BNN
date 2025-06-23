#!/usr/bin/env python3
"""
Training script for BNN on FASTA dataset (COVID vs Non-COVID classification)

Enhanced with comprehensive bio-detection methodology:
- Shallow MLP for HDCAM feasibility
- TALL for accuracy improvement
- Clinical metrics: Sensitivity, Precision, Specificity
- TOP-1 and TOP-2 evaluation protocols

This script trains a Binary Neural Network to classify DNA/RNA sequences 
as COVID or Non-COVID based on FASTA format sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import time
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow



class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in biomedical detection
    
    Original paper: "Focal Loss for Dense Object Detection" by Lin et al.
    Reduces loss contribution from easy examples and focuses on hard examples.
    
    Args:
        alpha: Weighting factor for rare class (COVID), typically 0.25-0.75
        gamma: Focusing parameter, typically 2.0
    """
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class F1Loss(nn.Module):
    """
    F1 Loss that directly optimizes F1 score
    
    Particularly useful for biomedical applications where both precision and recall
    are important (avoiding both false positives and false negatives).
    """
    def __init__(self, smooth=1e-8):
        super(F1Loss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        # Convert to probabilities
        y_pred = F.softmax(y_pred, dim=1)
        
        # Get positive class probabilities
        y_pred_pos = y_pred[:, 1]
        y_true_pos = y_true.float()
        
        # Calculate TP, FP, FN
        tp = torch.sum(y_pred_pos * y_true_pos)
        fp = torch.sum(y_pred_pos * (1 - y_true_pos))
        fn = torch.sum((1 - y_pred_pos) * y_true_pos)
        
        # Calculate precision and recall
        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
        
        # Return 1 - F1 as loss (we want to minimize loss)
        return 1 - f1


class SensitivitySpecificityLoss(nn.Module):
    """
    Balanced loss that optimizes both sensitivity and specificity
    
    For biomedical applications where both false positives and false negatives
    have clinical implications. Allows weighting between sensitivity and specificity.
    
    Args:
        sensitivity_weight: Weight for sensitivity (COVID detection rate)
        specificity_weight: Weight for specificity (Non-COVID identification rate)
    """
    def __init__(self, sensitivity_weight=0.8, specificity_weight=0.2, smooth=1e-8):
        super(SensitivitySpecificityLoss, self).__init__()
        self.sensitivity_weight = sensitivity_weight
        self.specificity_weight = specificity_weight
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        # Convert to probabilities
        y_pred = F.softmax(y_pred, dim=1)
        
        # Get predictions and ground truth
        y_pred_pos = y_pred[:, 0]  # COVID class probability (now class 0)
        y_true_pos = 1 - y_true.float()  # COVID targets (convert 0->1, 1->0)
        y_true_neg = y_true.float()      # Non-COVID targets
        y_pred_neg = 1 - y_pred_pos
        
        # Calculate TP, TN, FP, FN
        tp = torch.sum(y_pred_pos * y_true_pos)
        tn = torch.sum(y_pred_neg * y_true_neg)
        fp = torch.sum(y_pred_pos * y_true_neg)
        fn = torch.sum(y_pred_neg * y_true_pos)
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn + self.smooth)  # True Positive Rate
        specificity = tn / (tn + fp + self.smooth)  # True Negative Rate
        
        # Combined loss (minimize 1 - weighted_score)
        weighted_score = (self.sensitivity_weight * sensitivity + 
                         self.specificity_weight * specificity)
        
        return 1 - weighted_score


class MCCLoss(nn.Module):
    """
    Differentiable Matthews Correlation Coefficient Loss
    
    Excellent for biomedical applications as it considers all confusion matrix elements
    and provides a balanced measure that accounts for both FP and FN.
    
    MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    This version uses soft predictions to maintain gradient flow.
    """
    def __init__(self, smooth=1e-7):
        super(MCCLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get soft predictions (maintain gradients)
        pred_pos = probs[:, 0]  # Probability of COVID (now class 0)
        pred_neg = probs[:, 1]  # Probability of Non-COVID (now class 1)
        
        # Convert targets to float tensors
        targets_pos = 1.0 - targets.float()  # COVID targets (convert 0->1)
        targets_neg = targets.float()        # Non-COVID targets
        
        # Calculate soft confusion matrix components
        tp = torch.sum(pred_pos * targets_pos)  # COVID predicted as COVID
        tn = torch.sum(pred_neg * targets_neg)  # Non-COVID predicted as Non-COVID
        fp = torch.sum(pred_pos * targets_neg)  # Non-COVID predicted as COVID
        fn = torch.sum(pred_neg * targets_pos)  # COVID predicted as Non-COVID
        
        # Calculate MCC using soft values
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + self.smooth
        
        mcc = numerator / denominator
        
        # Return negative MCC as loss (maximize MCC = minimize -MCC)
        return 1 - mcc


class BalancedAccuracyLoss(nn.Module):
    """
    Differentiable Balanced Accuracy Loss
    
    Balanced accuracy is the average of sensitivity and specificity,
    providing equal weight to both classes regardless of their frequency.
    
    This version uses soft predictions to maintain gradient flow.
    """
    def __init__(self, smooth=1e-7):
        super(BalancedAccuracyLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get soft predictions (maintain gradients)
        pred_pos = probs[:, 0]  # Probability of COVID (now class 0)
        pred_neg = probs[:, 1]  # Probability of Non-COVID (now class 1)
        
        # Convert targets to float tensors
        targets_pos = 1.0 - targets.float()  # COVID targets (convert 0->1)
        targets_neg = targets.float()        # Non-COVID targets
        
        # Calculate soft confusion matrix components
        tp = torch.sum(pred_pos * targets_pos)  # COVID predicted as COVID
        tn = torch.sum(pred_neg * targets_neg)  # Non-COVID predicted as Non-COVID
        fp = torch.sum(pred_pos * targets_neg)  # Non-COVID predicted as COVID
        fn = torch.sum(pred_neg * targets_pos)  # COVID predicted as Non-COVID
        
        # Calculate sensitivity and specificity using soft values
        sensitivity = tp / (tp + fn + self.smooth)  # True Positive Rate
        specificity = tn / (tn + fp + self.smooth)  # True Negative Rate
        
        balanced_acc = (sensitivity + specificity) / 2
        
        # Return negative balanced accuracy as loss
        return 1 - balanced_acc


def get_bio_loss_function(loss_type='mcc', class_weights=None, **kwargs):
    """
    Get appropriate loss function for biomedical detection
    
    Args:
        loss_type: 'mcc', 'balanced_acc', 'focal', 'f1', 'sens_spec', 'weighted_ce', or 'ce'
        class_weights: Weights for each class [non_covid_weight, covid_weight]
        **kwargs: Additional parameters for specific loss functions
    
    Returns:
        Loss function appropriate for bio-detection
    """
    print(f"🔬 Bio-Detection Loss: Using {loss_type} loss function")
    
    if loss_type == 'mcc':
        smooth = kwargs.get('smooth', 1e-7)
        print(f"   • MCC Loss (smooth={smooth}) - Matthews Correlation Coefficient")
        print(f"   • Optimal for minimizing both FP and FN")
        print(f"   • Considers all confusion matrix elements")
        return MCCLoss(smooth=smooth)
    
    elif loss_type == 'balanced_acc':
        smooth = kwargs.get('smooth', 1e-7)
        print(f"   • Balanced Accuracy Loss (smooth={smooth})")
        print(f"   • Equal weight to sensitivity and specificity")
        print(f"   • Robust to class imbalance")
        return BalancedAccuracyLoss(smooth=smooth)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.7)  # Higher weight for COVID class
        gamma = kwargs.get('gamma', 2.0)
        print(f"   • Focal Loss (α={alpha}, γ={gamma}) - Addresses class imbalance")
        print(f"   • Focuses learning on hard examples")
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'f1':
        print(f"   • F1 Loss - Directly optimizes F1 score")
        print(f"   • Balances precision and recall")
        return F1Loss()
    
    elif loss_type == 'sens_spec':
        sens_weight = kwargs.get('sensitivity_weight', 0.8)
        spec_weight = kwargs.get('specificity_weight', 0.2)
        print(f"   • Sensitivity-Specificity Loss (sens={sens_weight}, spec={spec_weight})")
        print(f"   • Prioritizes COVID detection while maintaining specificity")
        return SensitivitySpecificityLoss(sens_weight, spec_weight)
    
    elif loss_type == 'weighted_ce':
        if class_weights is None:
            # Default: Higher weight for COVID class (minority)
            class_weights = torch.tensor([1.0, 3.0])  # [Non-COVID, COVID]
        print(f"   • Weighted Cross-Entropy (weights={class_weights.tolist()})")
        print(f"   • Higher penalty for misclassifying COVID cases")
        return nn.CrossEntropyLoss(weight=class_weights)
    
    else:  # 'ce' or default
        print(f"   • Standard Cross-Entropy Loss")
        return nn.CrossEntropyLoss()


class FASTADataset(Dataset):
    """
    FASTA Dataset loader for COVID vs Non-COVID classification
    
    The dataset contains FASTA files with DNA/RNA sequences:
    - COVID: 9 different COVID variant sequences
    - Not_COVID: 8 different non-COVID viral sequences
    
    Each sequence is encoded using K-mer representation:
    - K-mers of length k are extracted from sequences
    - Each unique K-mer gets a unique index
    - Sequences are represented as frequency vectors of K-mers
    """
    
    def __init__(self, data_dir, k=3, max_seq_length=2048, max_kmers=1000, split='train', train_ratio=0.8, random_state=42):
        self.data_dir = data_dir
        self.k = k  # K-mer length
        self.max_seq_length = max_seq_length
        self.max_kmers = max_kmers  # Maximum number of K-mers to consider
        self.nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}  # U->T for RNA
        
        # Load and process all sequences first to build K-mer vocabulary
        all_sequences = []
        all_labels = []
        
        # Load COVID sequences (label = 0) - Positive class
        covid_dir = os.path.join(data_dir, 'COVID')
        if os.path.exists(covid_dir):
            for fasta_file in os.listdir(covid_dir):
                if fasta_file.endswith('.fasta'):
                    file_path = os.path.join(covid_dir, fasta_file)
                    sequences = self._parse_fasta(file_path)
                    all_sequences.extend(sequences)
                    all_labels.extend([0] * len(sequences))  # COVID = 0 (positive class)
        
        # Load Non-COVID sequences (label = 1) - Negative class
        non_covid_dir = os.path.join(data_dir, 'Not_COVID')
        if os.path.exists(non_covid_dir):
            for fasta_file in os.listdir(non_covid_dir):
                if fasta_file.endswith('.fasta'):
                    file_path = os.path.join(non_covid_dir, fasta_file)
                    sequences = self._parse_fasta(file_path)
                    all_sequences.extend(sequences)
                    all_labels.extend([1] * len(sequences))  # Non-COVID = 1 (negative class)
        
        print(f"Loaded {len(all_sequences)} sequences")
        print(f"COVID sequences: {len(all_labels) - sum(all_labels)}")  # COVID = 0, so count zeros
        print(f"Non-COVID sequences: {sum(all_labels)}")  # Non-COVID = 1, so count ones
        
        # Build K-mer vocabulary from all sequences
        print(f"Building K-mer vocabulary with k={k}...")
        self._build_kmer_vocabulary(all_sequences)
        
        # Split data into train/test
        if len(all_sequences) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                all_sequences, all_labels, 
                train_size=train_ratio, 
                random_state=random_state,
                stratify=all_labels
            )
            
            if split == 'train':
                self.sequences = X_train
                self.labels = y_train
            else:
                self.sequences = X_test
                self.labels = y_test
        else:
            self.sequences = []
            self.labels = []
        
        print(f"{split.upper()} set: {len(self.sequences)} sequences")
        if len(self.labels) > 0:
            print(f"{split.upper()} COVID: {len(self.labels) - sum(self.labels)}, Non-COVID: {sum(self.labels)}")
    
    def _parse_fasta(self, file_path):
        """Parse FASTA file and return list of sequences"""
        sequences = []
        current_seq = ""
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq.upper())
                        current_seq = ""
                else:
                    current_seq += line
            
            # Don't forget the last sequence
            if current_seq:
                sequences.append(current_seq.upper())
        
        return sequences
    
    def _build_kmer_vocabulary(self, sequences):
        """Build K-mer vocabulary from all sequences"""
        kmer_counts = Counter()
        
        # Extract all K-mers from all sequences
        for sequence in sequences:
            kmers = self._extract_kmers(sequence)
            kmer_counts.update(kmers)
        
        # Select top K-mers based on frequency
        most_common_kmers = kmer_counts.most_common(self.max_kmers)
        
        # Create K-mer to index mapping
        self.kmer_to_idx = {}
        for i, (kmer, count) in enumerate(most_common_kmers):
            self.kmer_to_idx[kmer] = i
        
        self.vocab_size = len(self.kmer_to_idx)
        print(f"Built K-mer vocabulary with {self.vocab_size} unique {self.k}-mers")
        print(f"Most common K-mers: {[kmer for kmer, count in most_common_kmers[:10]]}")
    
    def _extract_kmers(self, sequence):
        """Extract all K-mers from a sequence"""
        # Clean sequence (remove unknown nucleotides for K-mer extraction)
        clean_sequence = ''.join([c for c in sequence if c in 'ACGT'])
        
        kmers = []
        for i in range(len(clean_sequence) - self.k + 1):
            kmer = clean_sequence[i:i+self.k]
            if len(kmer) == self.k:  # Ensure we have a complete K-mer
                kmers.append(kmer)
        
        return kmers
    
    def _encode_sequence_kmers(self, sequence):
        """
        Encode DNA/RNA sequence using binary K-mer representation
        
        Returns a binary vector where each element is 1 if the corresponding
        K-mer (from the top max_kmers most frequent) is present in the sequence,
        and 0 otherwise.
        """
        # Extract K-mers from sequence
        kmers = self._extract_kmers(sequence)
        
        # Create binary vector
        kmer_vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        
        # Create set of unique K-mers in this sequence for fast lookup
        unique_kmers = set(kmers)
        
        # Set to 1 if K-mer is present (binary encoding)
        for kmer in unique_kmers:
            if kmer in self.kmer_to_idx:
                idx = self.kmer_to_idx[kmer]
                kmer_vector[idx] = 1.0
        
        return kmer_vector
    
    def _encode_sequence(self, sequence):
        """
        Encode DNA/RNA sequence to numerical representation
        
        Two encoding strategies:
        1. One-hot encoding: Each nucleotide becomes a 5-dimensional vector
        2. Simple integer encoding: Each nucleotide becomes an integer (used here for efficiency)
        """
        # Truncate or pad sequence to max_seq_length
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + 'N' * (self.max_seq_length - len(sequence))  # Pad with N
        
        # Convert to integer encoding
        encoded = []
        for nucleotide in sequence:
            encoded.append(self.nucleotide_to_idx.get(nucleotide, 4))  # Unknown -> N (4)
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def _one_hot_encode(self, encoded_seq):
        """Convert integer encoded sequence to one-hot encoding"""
        one_hot = torch.zeros(self.max_seq_length, self.vocab_size)
        for i, nucleotide_idx in enumerate(encoded_seq):
            one_hot[i, nucleotide_idx] = 1.0
        
        return one_hot.flatten()  # Flatten to 1D for MLP input
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode sequence using K-mer representation
        kmer_vector = self._encode_sequence_kmers(sequence)
        
        return kmer_vector, torch.tensor(label, dtype=torch.long)


def get_fasta_loaders(data_dir, batch_size=32, k=3, max_seq_length=2048, max_kmers=1000, train_ratio=0.8, use_balanced_sampling=True):
    """
    Get FASTA train and test data loaders
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        k: K-mer length
        max_seq_length: Maximum sequence length (legacy parameter, kept for compatibility)
        max_kmers: Maximum number of K-mers to include in vocabulary
        train_ratio: Ratio of data for training
        use_balanced_sampling: Whether to use balanced class sampling for training
    """
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    covid_dir = os.path.join(data_dir, 'COVID')
    non_covid_dir = os.path.join(data_dir, 'Not_COVID')
    
    if not os.path.exists(covid_dir) or not os.path.exists(non_covid_dir):
        raise ValueError(f"Expected COVID and Not_COVID subdirectories in {data_dir}")
    
    train_dataset = FASTADataset(
        data_dir=data_dir,
        k=k,
        max_seq_length=max_seq_length,
        max_kmers=max_kmers,
        split='train',
        train_ratio=train_ratio
    )
    
    test_dataset = FASTADataset(
        data_dir=data_dir,
        k=k,
        max_seq_length=max_seq_length,
        max_kmers=max_kmers,
        split='test',
        train_ratio=train_ratio
    )
    
    # Create balanced sampler for training if requested
    train_sampler = None
    if use_balanced_sampling and len(train_dataset) > 0:
        # Calculate class weights for balanced sampling
        labels = train_dataset.labels
        covid_count = sum(labels)
        non_covid_count = len(labels) - covid_count
        
        print(f"🔄 Balanced Sampling Enabled:")
        print(f"   Training samples - COVID: {covid_count}, Non-COVID: {non_covid_count}")
        
        # Create weights for each sample (inverse of class frequency)
        if covid_count > 0 and non_covid_count > 0:
            covid_weight = 1.0 / covid_count
            non_covid_weight = 1.0 / non_covid_count
            
            sample_weights = [covid_weight if label == 1 else non_covid_weight for label in labels]
            sample_weights = torch.DoubleTensor(sample_weights)
            
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            print(f"   Balanced weights - COVID: {covid_weight:.8f}, Non-COVID: {non_covid_weight:.8f}")
            print(f"   This ensures equal representation during training")
        else:
            print(f"   Warning: Cannot create balanced sampler (one class has 0 samples)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


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
        
        # Convert binary outputs to logits for cross-entropy
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            loss = criterion(output, target)
        else:
            output_float = output.float()
            loss = criterion(output_float, target)
        
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            for n, p in model.named_parameters():
                # keep BatchNorm params unclipped
                if "bn" in n.lower() or "batchnorm" in n.lower():
                    continue
                p.clamp_(-1.5, 1.5)

        
        running_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        # Update correct and total counts
        correct += (pred == target).sum().item()
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
    """
    Enhanced evaluation with bio-detection metrics
    
    Args:
        model: BNN model to evaluate
        test_loader: Test data loader
        device: Computation device
        use_tall: Whether to use TALL classifier
        use_bio_metrics: Use comprehensive bio-detection metrics
    """

    model.eval()
    
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'forward'):
                # TALL classifier returns class predictions directly
                pred = model(data)
                # Create dummy logits for bio metrics
                output = F.one_hot(pred, num_classes=2).float()
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    
    # Calculate confusion matrix metrics
    cm = confusion_matrix(all_targets, all_preds)
    
    # For binary classification with COVID=0, Non-COVID=1:
    #                           [TP, FN]  <- COVID (class 0)  
    #                           [FP, TN]  <- Non-COVID (class 1)
    if cm.shape == (2, 2):
        TP, FN = cm[0]  # COVID cases: correctly predicted (TP), missed (FN)
        FP, TN = cm[1]  # Non-COVID cases: false alarms (FP), correctly rejected (TN)
        
        # Calculate additional metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Create bio_metrics dictionary
        bio_metrics = {
            'TP': int(TP),
            'FP': int(FP), 
            'TN': int(TN),
            'FN': int(FN),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1_score),
            'accuracy': float(accuracy / 100.0)  # Convert to 0-1 range
        }
        
        print(f"\nDetailed Bio-Detection Metrics:")
        print(f"True Positives (TP):  {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"True Negatives (TN):  {TN}")
        print(f"False Negatives (FN): {FN}")
        print(f"Precision:            {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        print(f"F1-Score:             {f1_score:.4f}")
    else:
        bio_metrics = None
    
    # Print standard classification report
    print("\nStandard Classification Report:")
    print(classification_report(all_targets, all_preds, 
                              target_names=['COVID', 'Non-COVID']))
    
    print("\nStandard Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    # Return accuracy and bio_metrics
    return accuracy, bio_metrics


def main():
    parser = argparse.ArgumentParser(description='Train BNN on FASTA dataset')
    parser.add_argument('--data_dir', default='./data/datasetCOVBNN', 
                       help='Path to FASTA dataset directory')
    parser.add_argument('--model_type', choices=['deep', 'shallow'], default='shallow',
                       help='Model architecture: deep or shallow BNN')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=60,
                       help='Maximum sequence length (sequences will be truncated/padded)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data used for training')
    parser.add_argument('--save_model', default='fasta_bnn_best.pth',
                       help='Path to save the best model')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # K-mer encoding arguments
    parser.add_argument('--k', type=int, default=5,
                       help='K-mer size for sequence encoding')
    parser.add_argument('--max_kmers', type=int, default=1000,
                       help='Maximum vocabulary size for K-mers')
    
    # TALL-specific arguments
    parser.add_argument('--use_tall', action='store_true',
                       help='Use TALL (Time-Augmented Last Layer) for evaluation')
    parser.add_argument('--tall_num_iter', type=int, default=30,
                       help='Number of iterations for TALL voting')
    parser.add_argument('--tall_flip_p', type=float, default=0.30,
                       help='Bit flip probability for TALL')
    
    # Bio-detection specific arguments
    parser.add_argument('--use_bio_metrics', action='store_true', default=False,
                       help='Use comprehensive bio-detection metrics')
    parser.add_argument('--force_shallow', action='store_true',
                       help='Force shallow architecture for HDCAM compatibility')
    parser.add_argument('--clinical_evaluation', action='store_true',
                       help='Perform clinical-grade evaluation with detailed metrics')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save bio-detection results visualization')
    
    # Bio-focused loss function arguments
    parser.add_argument('--loss_type', choices=['focal', 'f1', 'sens_spec', 'weighted_ce', 'ce', 'mcc', 'balanced_acc'], 
                       default='sens_spec', help='Loss function type for bio-detection')
    parser.add_argument('--focal_alpha', type=float, default=0.7,
                       help='Alpha parameter for Focal Loss (weight for COVID class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss (focusing parameter)')
    parser.add_argument('--sensitivity_weight', type=float, default=0.8,
                       help='Weight for sensitivity in sens_spec loss')
    parser.add_argument('--specificity_weight', type=float, default=0.2,
                       help='Weight for specificity in sens_spec loss')
    parser.add_argument('--covid_class_weight', type=float, default=1.0,
                       help='Weight for COVID class in weighted cross-entropy')
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True,
                       help='Use balanced class sampling during training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"K-mer length: {args.k}")
    print(f"Max K-mers in vocabulary: {args.max_kmers}")
    
    # Get data loaders
    print(f"Loading data from {args.data_dir}...")
    train_loader, test_loader = get_fasta_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        k=args.k,
        max_seq_length=args.max_seq_length,
        max_kmers=args.max_kmers,
        train_ratio=args.train_ratio,
        use_balanced_sampling=args.use_balanced_sampling
    )
    
    if len(train_loader.dataset) == 0:
        print("Error: No training data found!")
        return
    
    # Calculate input dimension (K-mer vocabulary size)
    input_dim = train_loader.dataset.vocab_size  # Number of unique K-mers
    num_classes = 2  # COVID vs Non-COVID
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    if args.model_type == 'deep':
        model = build_cam4_deep(num_classes=num_classes, in_features=input_dim)
        print("Using deep BNN architecture (4096→4096→128)")
    else:
        model = build_cam4_shallow(num_classes=num_classes, in_features=input_dim)
        print("🔬 Using shallow BNN architecture (128) - HDCAM optimized")
        print("   • Hardware-feasible for crossbar arrays")
        print("   • Reduced memory footprint")
        print("   • Optimized for bio-detection applications")
        model = build_cam4_shallow(num_classes=num_classes, in_features=input_dim)
        print("Using shallow BNN architecture")
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    if args.covid_class_weight > 1.0:
        class_weights = torch.tensor([1.0, args.covid_class_weight]).to(device)
    else:
        class_weights = None
        
    criterion = get_bio_loss_function(
        loss_type=args.loss_type,
        class_weights=class_weights,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        sensitivity_weight=args.sensitivity_weight,
        specificity_weight=args.specificity_weight
    )
    
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)
    
    print(f"🎯 Optimization target: {args.loss_type} loss")
    if args.loss_type == 'focal':
        print(f"   • Prioritizes hard examples and COVID class detection")
    elif args.loss_type == 'f1':
        print(f"   • Directly optimizes F1 score for balanced precision/recall")
    elif args.loss_type == 'sens_spec':
        print(f"   • Balances sensitivity ({args.sensitivity_weight}) and specificity ({args.specificity_weight})")
    elif args.loss_type == 'weighted_ce':
        print(f"   • Higher penalty for COVID misclassification (weight: {args.covid_class_weight})")
    elif args.loss_type == 'mcc':
        print(f"   • Optimizes Matthews Correlation Coefficient (MCC)")
        print(f"   • Considers all confusion matrix elements (TP, TN, FP, FN)")
    elif args.loss_type == 'balanced_acc':
        print(f"   • Balanced Accuracy - Average of sensitivity and specificity")
    
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        eval_result = evaluate(model, test_loader, device)
        
        # Handle both single accuracy and tuple (accuracy, bio_metrics)
        if isinstance(eval_result, tuple):
            test_acc, bio_metrics = eval_result
        else:
            test_acc = eval_result
            bio_metrics = None
        
        test_accs.append(test_acc)
        
        model_selection_metric = test_acc  # Fallback to accuracy
        
        # Update learning rate (scheduler needs float, not tuple)
        scheduler.step(model_selection_metric)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%, LR: {current_lr:.2e}")
        
        # Save best model based on bio-focused metric
        if model_selection_metric > best_acc:
            best_acc = model_selection_metric
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_acc,
                'best_test_acc': test_acc,
                'best_bio_metrics': bio_metrics,
                'args': args,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'kmer_vocab': train_loader.dataset.kmer_to_idx,  # Save the vocabulary!
                'vocab_size': train_loader.dataset.vocab_size
            }, args.save_model)
        
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(args.save_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation with bio-detection metrics
    print("\n" + "="*60)
    print("FINAL EVALUATION - COVID-19 BIO-DETECTION")
    print("="*60)
    
    # Final evaluation with bio-detection metrics
    eval_result = evaluate(model, test_loader, device)
    
    # Handle tuple return (accuracy, bio_metrics)
    if isinstance(eval_result, tuple):
        final_acc, final_bio_metrics = eval_result
    else:
        final_acc = eval_result
        final_bio_metrics = None


    # TALL evaluation if requested
    tall_acc = None
    tall_bio_metrics = None
    if args.use_tall:
        print(f"\n🎯 TALL Evaluation (num_iter={args.tall_num_iter}, flip_p={args.tall_flip_p}):")
        print("   • Time-Augmented Last Layer for improved accuracy")
        print("   • Multiple noisy passes with majority voting")
        
        tall_model = TALLClassifier(model, num_iter=args.tall_num_iter, flip_p=args.tall_flip_p)
        tall_model = tall_model.to(device)
        
        # TALL evaluation with bio-detection metrics
        tall_eval_result = evaluate(tall_model, test_loader, device, use_tall=True)
        
        # Handle tuple return (accuracy, bio_metrics)
        if isinstance(tall_eval_result, tuple):
            tall_acc, tall_bio_metrics = tall_eval_result
        else:
            tall_acc = tall_eval_result
            tall_bio_metrics = None
    
    
    # Save training history with bio-metrics
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
        'final_acc': final_acc,
        'tall_acc': tall_acc,
        'tall_bio_metrics': tall_bio_metrics,
        'training_time': training_time,
        'methodology': 'COVID-19 Bio-Detection using BNN in HDCAM',
        'args': vars(args)
    }
    
    history_file = args.save_model.replace('.pth', '_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_file}")
    print(f"Model saved to {args.save_model}")


if __name__ == '__main__':
    main()
