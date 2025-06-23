#!/usr/bin/env python3
"""
Evaluation script for trained BNN models on FASTA dataset (COVID vs Non-COVID)

This script loads a trained BNN model and evaluates it on the FASTA test set,
with options for TALL (Time-Augmented Last Layer) evaluation and detailed analysis.

NEW INFERENCE MODE:
- Use --inference_mode to load ALL labeled data for inference (no train/test split)
- Use --inference_data_dir to specify a different directory for inference data
- Requires labeled data with COVID/ and Not_COVID/ subdirectories
- Uses ALL available sequences (no data leakage concerns for separate validation sets)
- Automatically uses the training vocabulary from the model checkpoint
- Perfect for evaluating on completely separate validation/test datasets

Usage Examples:
  # Standard evaluation on test split (respects train_ratio)
  python evaluate_fasta.py --model_path model.pth --data_dir ./data/datasetCOVBNN
  
  # Inference on ALL data in validation set (no splitting)
  python evaluate_fasta.py --model_path model.pth --inference_mode --inference_data_dir ./validation_set/
  
  # Inference using same data directory but ALL data (no splitting)
  python evaluate_fasta.py --model_path model.pth --inference_mode
"""

import torch
import torch.nn as nn
import argparse
import time
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep,build_cam4_shallow_fully_binary, build_cam4_shallow
from train_fasta import FASTADataset, get_fasta_loaders


class FASTAInferenceDataset(FASTADataset):
    """
    FASTA Dataset loader for inference-only data (no training/test split)
    
    This class loads ALL FASTA sequences for inference without any train/test splitting.
    Perfect for evaluating on a separate validation/test dataset where you want to use
    all available data and still get accuracy metrics.
    """
    
    def __init__(self, data_dir, kmer_vocab, k=3, max_seq_length=2048, max_kmers=1000):
        """
        Initialize inference dataset that loads ALL data without splitting
        
        Args:
            data_dir: Path to directory containing COVID/Not_COVID subdirectories
            kmer_vocab: Pre-trained K-mer vocabulary (dict from training)
            k: K-mer length (must match training)
            max_seq_length: Maximum sequence length (legacy parameter)
            max_kmers: Maximum number of K-mers (must match training)
        """
        self.data_dir = data_dir
        self.k = k
        self.max_seq_length = max_seq_length
        self.max_kmers = max_kmers
        self.nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
        
        # Use pre-trained vocabulary
        self.kmer_to_idx = kmer_vocab
        self.vocab_size = len(self.kmer_to_idx)
        print(f"Using pre-trained K-mer vocabulary with {self.vocab_size} unique {self.k}-mers")
        
        # Load ALL sequences without any splitting
        self.sequences = []
        self.labels = []
        self._load_all_labeled_data()
        
        print(f"Loaded {len(self.sequences)} sequences for inference (NO TRAIN/TEST SPLIT)")
        if len(self.labels) > 0:
            covid_count = len(self.labels) - sum(self.labels)
            non_covid_count = sum(self.labels)
            print(f"COVID sequences: {covid_count}, Non-COVID sequences: {non_covid_count}")
    
    def _load_all_labeled_data(self):
        """Load ALL data with labels from COVID/Not_COVID subdirectories (no splitting)"""
        total_covid = 0
        total_non_covid = 0
        
        # Load ALL COVID sequences (label = 0)
        covid_dir = os.path.join(self.data_dir, 'COVID')
        if os.path.exists(covid_dir):
            for fasta_file in os.listdir(covid_dir):
                if fasta_file.endswith('.fasta'):
                    file_path = os.path.join(covid_dir, fasta_file)
                    sequences = self._parse_fasta(file_path)
                    self.sequences.extend(sequences)
                    self.labels.extend([0] * len(sequences))  # COVID = 0
                    total_covid += len(sequences)
                    print(f"  Loaded {len(sequences)} COVID sequences from {fasta_file}")
        
        # Load ALL Non-COVID sequences (label = 1)
        non_covid_dir = os.path.join(self.data_dir, 'Not_COVID')
        if os.path.exists(non_covid_dir):
            for fasta_file in os.listdir(non_covid_dir):
                if fasta_file.endswith('.fasta'):
                    file_path = os.path.join(non_covid_dir, fasta_file)
                    sequences = self._parse_fasta(file_path)
                    self.sequences.extend(sequences)
                    self.labels.extend([1] * len(sequences))  # Non-COVID = 1
                    total_non_covid += len(sequences)
                    print(f"  Loaded {len(sequences)} Non-COVID sequences from {fasta_file}")
        
        if total_covid == 0 and total_non_covid == 0:
            print("Warning: No labeled data found! Expected COVID/ and Not_COVID/ subdirectories")
        else:
            print(f"✓ Successfully loaded ALL sequences without train/test split")
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode sequence using K-mer representation with pre-trained vocabulary
        kmer_vector = self._encode_sequence_kmers(sequence)
        
        return kmer_vector, torch.tensor(label, dtype=torch.long)


def get_inference_data_loader(data_dir, kmer_vocab, batch_size=32, k=3, max_seq_length=2048, max_kmers=1000):
    """
    Get data loader for inference-only data (loads ALL labeled data without train/test split)
    
    Args:
        data_dir: Path to inference data directory (should contain COVID/ and Not_COVID/ subdirs)
        kmer_vocab: Pre-trained K-mer vocabulary from training
        batch_size: Batch size for inference
        k: K-mer length (must match training)
        max_seq_length: Maximum sequence length (legacy parameter)
        max_kmers: Maximum number of K-mers (must match training)
    
    Returns:
        DataLoader for inference data
    """
    dataset = FASTAInferenceDataset(
        data_dir=data_dir,
        kmer_vocab=kmer_vocab,
        k=k,
        max_seq_length=max_seq_length,
        max_kmers=max_kmers
    )
    
    if len(dataset) == 0:
        print("Warning: No sequences found in inference dataset!")
        return None
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return loader


def evaluate(model, test_loader, device, use_tall=False):
    """
    Simple evaluation function that matches train_fasta.py format
    
    Returns:
        tuple: (accuracy, bio_metrics) where bio_metrics contains TP, FP, TN, FN
    """
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'forward'):
                # TALL classifier returns class predictions directly
                pred = model(data)
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
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
    
    return accuracy, bio_metrics


def evaluate_detailed(model, test_loader, device, use_tall=False, save_plots=True, output_dir='./', section_name=''):
    """
    Enhanced evaluation with bio-detection metrics and confusion matrix analysis
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        use_tall: Whether using TALL evaluation
        save_plots: Whether to save plots
        output_dir: Directory to save outputs
        section_name: Name of the evaluation section for plot titles
    """
    if not use_tall:
        model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    print("Running detailed evaluation...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'forward'):
                # TALL classifier returns class predictions directly
                pred = model(data)
                # For TALL, we don't have probabilities, so use dummy ones
                probs = torch.zeros(len(pred), 2, device=device)
                probs[range(len(pred)), pred] = 1.0  # One-hot for predictions
                probs[range(len(pred)), pred] = 1.0
            else:
                output = model(data)
                pred = output.argmax(dim=1)
                # Convert logits to probabilities
                probs = torch.softmax(output, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Detailed classification metrics
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_targets, all_predictions, 
                              target_names=['COVID', 'Non-COVID'], digits=4))
    
    # Calculate confusion matrix with new class mapping: COVID=0, Non-COVID=1
    cm = confusion_matrix(all_targets, all_predictions)
    print("\nConfusion Matrix (COVID=0, Non-COVID=1):")
    print(f"                 Predicted")
    print(f"                 COVID       Non-COVID")
    print(f"Actual COVID     {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Non-COVID        {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Calculate detailed metrics with correct mapping
    # For binary classification with COVID=0, Non-COVID=1:
    #                           [TP, FN]  <- COVID (class 0)  
    #                           [FP, TN]  <- Non-COVID (class 1)
    if cm.shape == (2, 2):
        TP, FN = cm[0]  # COVID cases: correctly predicted (TP), missed (FN)
        FP, TN = cm[1]  # Non-COVID cases: false alarms (FP), correctly rejected (TN)
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # COVID detection rate
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Non-COVID identification rate  
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0    # Positive predictive value
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0          # Negative predictive value
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\nDetailed Bio-Detection Metrics:")
        print(f"True Positives (TP):  {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"True Negatives (TN):  {TN}")
        print(f"False Negatives (FN): {FN}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        print(f"Precision (PPV):      {precision:.4f}")
        print(f"NPV:                  {npv:.4f}")
        print(f"F1-Score:             {f1:.4f}")
        
        # Create bio_metrics dictionary for consistency with train script
        bio_metrics = {
            'TP': int(TP),
            'FP': int(FP), 
            'TN': int(TN),
            'FN': int(FN),
            'precision': float(precision),
            'recall': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'accuracy': float(accuracy / 100.0)  # Convert to 0-1 range
        }
    else:
        bio_metrics = None
    
    if save_plots and not use_tall:  # Only create plots for non-TALL models (with probabilities)
        # ROC Curve - Note: COVID is now class 0, so we use probabilities[:, 0] for positive class
        fpr, tpr, _ = roc_curve(all_targets, all_probabilities[:, 0], pos_label=0)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12, 5))
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        roc_title = f'ROC Curve - COVID Detection'
        if section_name:
            roc_title = f'{section_name}: ROC Curve - COVID Detection'
        plt.title(roc_title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Precision-Recall Curve - COVID is positive class (label 0)
        precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_probabilities[:, 0], pos_label=0)
        avg_precision = average_precision_score(all_targets, all_probabilities[:, 0], pos_label=0)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        pr_title = f'Precision-Recall Curve - COVID Detection'
        if section_name:
            pr_title = f'{section_name}: Precision-Recall Curve - COVID Detection'
        plt.title(pr_title)
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'evaluation_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nROC and PR curves saved to {plot_file}")
        plt.close()
        
        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['COVID', 'Non-COVID'],
                   yticklabels=['COVID', 'Non-COVID'])
        cm_title = f'Confusion Matrix - COVID Detection'
        if section_name:
            cm_title = f'{section_name}: Confusion Matrix - COVID Detection'
        plt.title(cm_title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_file = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix heatmap saved to {cm_file}")
        plt.close()
    
    # Return detailed results
    results = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist(),
        'bio_metrics': bio_metrics
    }
    
    if not use_tall:
        fpr, tpr, _ = roc_curve(all_targets, all_probabilities[:, 0], pos_label=0)
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(all_targets, all_probabilities[:, 0], pos_label=0)
        results.update({
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'probabilities': all_probabilities.tolist()
        })
    
    return results


def evaluate_inference(model, data_loader, device, use_tall=False):
    """
    Evaluation function for inference-only data (labeled data, no train/test split)
    
    This function evaluates on ALL loaded data without any splitting.
    Perfect for validation sets where you want to use every available sequence.
    
    Returns:
        dict: Contains accuracy, predictions, probabilities, and detailed metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Running Inference')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'forward'):
                # TALL classifier returns class predictions directly
                pred = model(data)
                # For TALL, we don't have probabilities, so use dummy ones
                probs = torch.zeros(len(pred), 2, device=device)
                probs[range(len(pred)), pred] = 1.0  # One-hot for predictions
            else:
                output = model(data)
                pred = output.argmax(dim=1)
                # Convert logits to probabilities
                probs = torch.softmax(output, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    accuracy = 100. * correct / total
    
    print(f"\nInference Results (ALL DATA - NO SPLIT):")
    print(f"Total sequences evaluated: {len(all_predictions)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Calculate detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['COVID', 'Non-COVID'], digits=4))
    
    print("\nConfusion Matrix (COVID=0, Non-COVID=1):")
    print(f"                 Predicted")
    print(f"                 COVID       Non-COVID")
    print(f"Actual COVID     {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Non-COVID        {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Calculate bio-detection metrics
    if cm.shape == (2, 2):
        TP, FN = cm[0]  # COVID cases: correctly predicted (TP), missed (FN)
        FP, TN = cm[1]  # Non-COVID cases: false alarms (FP), correctly rejected (TN)
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\nDetailed Bio-Detection Metrics:")
        print(f"True Positives (TP):  {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"True Negatives (TN):  {TN}")
        print(f"False Negatives (FN): {FN}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        print(f"Precision (PPV):      {precision:.4f}")
        print(f"NPV:                  {npv:.4f}")
        print(f"F1-Score:             {f1:.4f}")
        
        bio_metrics = {
            'TP': int(TP),
            'FP': int(FP), 
            'TN': int(TN),
            'FN': int(FN),
            'precision': float(precision),
            'recall': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'accuracy': float(accuracy / 100.0)  # Convert to 0-1 range
        }
    else:
        bio_metrics = None
    
    # Prediction summary
    covid_predictions = (all_predictions == 0).sum()
    non_covid_predictions = (all_predictions == 1).sum()
    print(f"\nPrediction Summary:")
    print(f"Predicted COVID: {covid_predictions}")
    print(f"Predicted Non-COVID: {non_covid_predictions}")
    
    results = {
        'accuracy': accuracy,
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist(),
        'total_sequences': len(all_predictions),
        'confusion_matrix': cm.tolist(),
        'bio_metrics': bio_metrics,
        'classification_report': classification_report(
            all_targets, all_predictions, 
            target_names=['COVID', 'Non-COVID'], 
            output_dict=True
        )
    }
    
    if not use_tall:
        # Add probabilities and confidence statistics
        covid_confidences = all_probabilities[:, 0]  # COVID probabilities
        non_covid_confidences = all_probabilities[:, 1]  # Non-COVID probabilities
        
        print(f"\nConfidence Statistics:")
        print(f"Average COVID confidence: {covid_confidences.mean():.4f} ± {covid_confidences.std():.4f}")
        print(f"Average Non-COVID confidence: {non_covid_confidences.mean():.4f} ± {non_covid_confidences.std():.4f}")
        
        results.update({
            'probabilities': all_probabilities.tolist(),
            'confidence_stats': {
                'covid_mean': float(covid_confidences.mean()),
                'covid_std': float(covid_confidences.std()),
                'non_covid_mean': float(non_covid_confidences.mean()),
                'non_covid_std': float(non_covid_confidences.std())
            }
        })
        
        # Add additional metrics similar to evaluate_detailed
        if bio_metrics:
            results.update({
                'sensitivity': bio_metrics['recall'],
                'specificity': bio_metrics['specificity'],
                'precision': bio_metrics['precision'],
                'npv': npv,
                'f1_score': bio_metrics['f1_score']
            })
    
    return results


def analyze_sequence_predictions(model, test_loader, device, use_tall=False, num_examples=10):
    """
    Analyze predictions on individual sequences for interpretability
    """
    if not use_tall:
        model.eval()
    
    print(f"\nAnalyzing {num_examples} example predictions...")
    
    examples_found = 0
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            
            if use_tall and hasattr(model, 'forward'):
                batch_preds = model(batch_data)
                batch_probs = None
            else:
                batch_outputs = model(batch_data)
                batch_preds = batch_outputs.argmax(dim=1)
                batch_probs = torch.softmax(batch_outputs, dim=1)
            
            for i in range(len(batch_data)):
                if examples_found >= num_examples:
                    return
                
                target = batch_targets[i].item()
                pred = batch_preds[i].item()
                
                target_name = "COVID" if target == 1 else "Non-COVID"
                pred_name = "COVID" if pred == 1 else "Non-COVID"
                correct = "✓" if target == pred else "✗"
                
                print(f"\nExample {examples_found + 1}:")
                print(f"  Actual: {target_name}")
                print(f"  Predicted: {pred_name} {correct}")
                
                if batch_probs is not None:
                    prob_non_covid = batch_probs[i, 0].item()
                    prob_covid = batch_probs[i, 1].item()
                    print(f"  Confidence: Non-COVID={prob_non_covid:.4f}, COVID={prob_covid:.4f}")
                
                examples_found += 1


def run_tall_sensitivity_analysis(model, test_loader, device, output_dir='./'):
    """
    Run sensitivity analysis for TALL hyperparameters
    """
    print("\nRunning TALL sensitivity analysis...")
    
    num_iter_values = [10, 20, 30, 50, 100]
    flip_p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    
    for num_iter in num_iter_values:
        for flip_p in flip_p_values:
            print(f"Testing num_iter={num_iter}, flip_p={flip_p}")
            
            tall_model = TALLClassifier(model, num_iter=num_iter, flip_p=flip_p)
            tall_model = tall_model.to(device)
            
            # Quick evaluation
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = tall_model(data)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            accuracy = 100. * correct / total
            results.append({
                'num_iter': num_iter,
                'flip_p': flip_p,
                'accuracy': accuracy
            })
            print(f"  Accuracy: {accuracy:.2f}%")
    
    # Save results
    results_file = os.path.join(output_dir, 'tall_sensitivity_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTALL sensitivity analysis saved to {results_file}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"Best TALL configuration: num_iter={best_result['num_iter']}, "
          f"flip_p={best_result['flip_p']}, accuracy={best_result['accuracy']:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained BNN on FASTA dataset')
    parser.add_argument('--model_path', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', default='./data/datasetCOVBNN',
                       help='Path to FASTA dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                       help='Maximum sequence length (legacy parameter, kept for compatibility)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio (must match training)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', default='./fasta_results',
                       help='Directory to save evaluation results')
    
    # NEW: Inference-only data loading option
    parser.add_argument('--inference_data_dir', default=None,
                       help='Path to inference-only FASTA dataset directory (no train/test split)')
    parser.add_argument('--inference_mode', action='store_true',
                       help='Load data for inference only (no training data split required)')
    
    # K-mer parameters (must match training configuration)
    parser.add_argument('--k', type=int, default=3,
                       help='K-mer length (must match training)')
    parser.add_argument('--max_kmers', type=int, default=1000,
                       help='Maximum number of K-mers in vocabulary (must match training)')
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True,
                       help='Use balanced sampling for dataset loading')
    
    # TALL evaluation options
    parser.add_argument('--use_tall', action='store_true',
                       help='Use TALL (Time-Augmented Last Layer) for evaluation')
    parser.add_argument('--tall_num_iter', type=int, default=30,
                       help='Number of iterations for TALL voting')
    parser.add_argument('--tall_flip_p', type=float, default=0.30,
                       help='Bit flip probability for TALL')
    parser.add_argument('--tall_sensitivity', action='store_true',
                       help='Run TALL sensitivity analysis')
    
    # Analysis options
    parser.add_argument('--analyze_examples', type=int, default=10,
                       help='Number of example predictions to analyze')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dedicated subdirectories for each evaluation type
    popcount_dir = os.path.join(args.output_dir, "1_popcount_evaluation")
    fully_binary_dir = os.path.join(args.output_dir, "2_fully_binary_evaluation")
    tall_dir = os.path.join(args.output_dir, "3_tall_evaluation")
    
    os.makedirs(popcount_dir, exist_ok=True)
    os.makedirs(fully_binary_dir, exist_ok=True)
    os.makedirs(tall_dir, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  - Popcount: {popcount_dir}")
    print(f"  - Fully Binary: {fully_binary_dir}")
    print(f"  - TALL: {tall_dir}")
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # CRITICAL: Load the training vocabulary from checkpoint
    training_vocab = None
    vocab_size = None
    if 'kmer_vocab' in checkpoint:
        training_vocab = checkpoint['kmer_vocab']
        vocab_size = checkpoint.get('vocab_size', len(training_vocab))
        print(f"✓ Loaded training vocabulary with {len(training_vocab)} k-mers")
    else:
        print("⚠️  WARNING: No vocabulary found in checkpoint!")
        print("   This evaluation may use different vocabulary than training!")
    
    # Get model configuration from checkpoint
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        model_type = train_args.model_type if hasattr(train_args, 'model_type') else 'deep'
        # Use K-mer vocabulary size for input dimension
        k = train_args.k if hasattr(train_args, 'k') else args.k
        max_kmers = train_args.max_kmers if hasattr(train_args, 'max_kmers') else args.max_kmers
        input_dim = vocab_size if vocab_size is not None else max_kmers  # Use actual vocab size
    else:
        # Fallback defaults
        model_type = 'deep'
        k = args.k
        max_kmers = args.max_kmers
        input_dim = vocab_size if vocab_size is not None else max_kmers
    
    print(f"Model type: {model_type}")
    print(f"K-mer length: {k}")
    print(f"Max K-mers: {max_kmers}")
    print(f"Input dimension: {input_dim}")
    
    # Create model
    if model_type == 'deep':
        model = build_cam4_deep(num_classes=2, in_features=input_dim)
    else:
        model = build_cam4_shallow(num_classes=2, in_features=input_dim)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'best_acc' in checkpoint:
        print(f"Loaded model with training accuracy: {checkpoint['best_acc']:.2f}%")
    
    # Load data for evaluation
    if args.inference_mode or args.inference_data_dir:
        # INFERENCE MODE: Load ALL labeled data without train/test split
        inference_dir = args.inference_data_dir if args.inference_data_dir else args.data_dir
        print(f"Loading inference data from {inference_dir}...")
        
        if training_vocab is None:
            print("❌ ERROR: Cannot run inference mode without training vocabulary!")
            print("   The model checkpoint must contain 'kmer_vocab' for inference.")
            return
        
        # Check if data has proper structure (COVID/Not_COVID directories)
        covid_dir = os.path.join(inference_dir, 'COVID')
        non_covid_dir = os.path.join(inference_dir, 'Not_COVID')
        
        if not (os.path.exists(covid_dir) and os.path.exists(non_covid_dir)):
            print("❌ ERROR: Inference mode expects labeled data with COVID/ and Not_COVID/ subdirectories")
            print(f"   Looking for: {covid_dir} and {non_covid_dir}")
            return
        
        print("✓ Detected labeled inference data (COVID/Not_COVID directories)")
        print("✓ Will load ALL data without train/test split for inference")
        
        test_loader = get_inference_data_loader(
            data_dir=inference_dir,
            kmer_vocab=training_vocab,
            batch_size=args.batch_size,
            k=k,
            max_seq_length=args.max_seq_length,
            max_kmers=max_kmers
        )
        
        if test_loader is None or len(test_loader.dataset) == 0:
            print("Error: No inference data found!")
            return
        
        print(f"Inference dataset size: {len(test_loader.dataset)} sequences")
        
    else:
        # STANDARD MODE: Load test split from training data
        print(f"Loading test data from {args.data_dir}...")
        
        if training_vocab is not None:
            # Create dataset directly with training vocabulary
            full_dataset = FASTADataset(
                data_dir=args.data_dir,
                k=k,
                max_seq_length=args.max_seq_length,
                max_kmers=max_kmers
            )
            
            # Force the dataset to use training vocabulary
            full_dataset.kmer_to_idx = training_vocab
            full_dataset.vocab_size = len(training_vocab)
            print(f"✓ Using training vocabulary for evaluation (vocab_size: {full_dataset.vocab_size})")
            
            # Split dataset to get test portion
            train_size = int(args.train_ratio * len(full_dataset))
            test_size = len(full_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
            )
        else:
            # Fallback: use get_fasta_loaders (may have different vocabulary)
            print("⚠️  Building new vocabulary (may cause inconsistency)")
            _, test_loader, _ = get_fasta_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                k=k,
                max_seq_length=args.max_seq_length,  # Legacy parameter
                max_kmers=max_kmers,
                train_ratio=args.train_ratio,
                use_balanced_sampling=args.use_balanced_sampling
            )
        
        if len(test_loader.dataset) == 0:
            print("Error: No test data found!")
            return
        
        print(f"Test set size: {len(test_loader.dataset)} sequences")
    
    # Store all results
    all_results = {}
    
    # ================================================================
    # SECTION 1: LAST LAYER POPCOUNT (Standard BNN with FP32 inference)
    # ================================================================
    print("\n" + "="*70)
    print("SECTION 1: LAST LAYER POPCOUNT (FP32 inference)")
    print("="*70)
    
    start_time = time.time()
    
    # Choose evaluation function based on mode
    if args.inference_mode or args.inference_data_dir:
        popcount_results = evaluate_inference(model, test_loader, device, use_tall=False)
    else:
        popcount_results = evaluate_detailed(model, test_loader, device, 
                                           use_tall=False, save_plots=args.save_plots,
                                           output_dir=popcount_dir, section_name="Section 1: Popcount")
    
    popcount_time = time.time() - start_time
    
    print(f"\nPopcount evaluation completed in {popcount_time:.2f} seconds")
    
    # Save popcount results
    popcount_results['evaluation_time'] = popcount_time
    popcount_results['method'] = 'popcount'
    popcount_results['inference_mode'] = args.inference_mode or bool(args.inference_data_dir)
    all_results['popcount'] = popcount_results
    popcount_file = os.path.join(popcount_dir, 'popcount_evaluation_results.json')
    with open(popcount_file, 'w') as f:
        json.dump(popcount_results, f, indent=2)
    print(f"Popcount evaluation results saved to {popcount_file}")
    
    # ================================================================
    # SECTION 2: FULLY BINARY (All layers binary including last layer)
    # ================================================================
    print("\n" + "="*70)
    print("SECTION 2: FULLY BINARY (All layers binary)")
    print("="*70)
    
    # Create fully binary model
    print("Creating fully binary model...")
    if model_type == 'deep':
        # For now, use shallow fully binary - can be extended for deep if needed
        fully_binary_model = build_cam4_shallow_fully_binary(num_classes=2, in_features=input_dim)
    else:
        fully_binary_model = build_cam4_shallow_fully_binary(num_classes=2, in_features=input_dim)
    
    # Load the same weights (this assumes the architecture is compatible)
    try:
        # Try to load compatible weights
        fully_binary_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded compatible weights into fully binary model")
    except Exception as e:
        print(f"Warning: Could not load weights into fully binary model: {e}")
        print("Using randomly initialized fully binary model")
    
    fully_binary_model = fully_binary_model.to(device)
    
    start_time = time.time()
    
    # Choose evaluation function based on mode
    if args.inference_mode or args.inference_data_dir:
        fully_binary_results = evaluate_inference(fully_binary_model, test_loader, device, use_tall=False)
    else:
        fully_binary_results = evaluate_detailed(fully_binary_model, test_loader, device, 
                                                use_tall=False, save_plots=args.save_plots,
                                                output_dir=fully_binary_dir, section_name="Section 2: Fully Binary")
    
    fully_binary_time = time.time() - start_time
    
    print(f"\nFully binary evaluation completed in {fully_binary_time:.2f} seconds")
    
    # Save fully binary results
    fully_binary_results['evaluation_time'] = fully_binary_time
    fully_binary_results['method'] = 'fully_binary'
    fully_binary_results['inference_mode'] = args.inference_mode or bool(args.inference_data_dir)
    all_results['fully_binary'] = fully_binary_results
    fully_binary_file = os.path.join(fully_binary_dir, 'fully_binary_evaluation_results.json')
    with open(fully_binary_file, 'w') as f:
        json.dump(fully_binary_results, f, indent=2)
    print(f"Fully binary evaluation results saved to {fully_binary_file}")
    
    # ================================================================
    # SECTION 3: TALL EVALUATION (Time-Augmented Last Layer)
    # ================================================================
    if args.use_tall:
        print("\n" + "="*70)
        print(f"SECTION 3: TALL EVALUATION (num_iter={args.tall_num_iter}, flip_p={args.tall_flip_p})")
        print("="*70)
        
        tall_model = TALLClassifier(model, num_iter=args.tall_num_iter, flip_p=args.tall_flip_p)
        tall_model = tall_model.to(device)
        
        start_time = time.time()
        
        # Choose evaluation function based on mode
        if args.inference_mode or args.inference_data_dir:
            tall_results = evaluate_inference(tall_model, test_loader, device, use_tall=True)
        else:
            tall_results = evaluate_detailed(tall_model, test_loader, device, 
                                           use_tall=True, save_plots=args.save_plots,
                                           output_dir=tall_dir, section_name="Section 3: TALL")
        
        tall_time = time.time() - start_time
        
        print(f"\nTALL evaluation completed in {tall_time:.2f} seconds")
        
        # Save TALL results
        tall_results['evaluation_time'] = tall_time
        tall_results['num_iter'] = args.tall_num_iter
        tall_results['flip_p'] = args.tall_flip_p
        tall_results['method'] = 'tall'
        tall_results['inference_mode'] = args.inference_mode or bool(args.inference_data_dir)
        all_results['tall'] = tall_results
        tall_file = os.path.join(tall_dir, 'tall_evaluation_results.json')
        with open(tall_file, 'w') as f:
            json.dump(tall_results, f, indent=2)
        print(f"TALL evaluation results saved to {tall_file}")
    else:
        print("\n" + "="*70)
        print("SECTION 3: TALL EVALUATION (SKIPPED - use --use_tall to enable)")
        print("="*70)
    
    # ================================================================
    # PERFORMANCE COMPARISON
    # ================================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    # In inference mode, we always have accuracy since we always have labeled data
    print(f"{'Method':<20} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<12}")
    print("-" * 56)
    
    # Popcount results
    acc_str = f"{popcount_results['accuracy']:.2f}" if 'accuracy' in popcount_results else "N/A"
    f1_str = f"{popcount_results.get('f1_score', 0):.4f}" if popcount_results.get('f1_score') else "N/A"
    print(f"{'Popcount':<20} {acc_str:<12} {f1_str:<12} {popcount_results['evaluation_time']:<12.2f}")
    
    # Fully binary results  
    acc_str = f"{fully_binary_results['accuracy']:.2f}" if 'accuracy' in fully_binary_results else "N/A"
    f1_str = f"{fully_binary_results.get('f1_score', 0):.4f}" if fully_binary_results.get('f1_score') else "N/A"
    print(f"{'Fully Binary':<20} {acc_str:<12} {f1_str:<12} {fully_binary_results['evaluation_time']:<12.2f}")
    
    # TALL results (if available)
    if args.use_tall and 'tall' in all_results:
        tall_results = all_results['tall']
        acc_str = f"{tall_results['accuracy']:.2f}" if 'accuracy' in tall_results else "N/A"
        f1_str = f"{tall_results.get('f1_score', 0):.4f}" if tall_results.get('f1_score') else "N/A"
        print(f"{'TALL':<20} {acc_str:<12} {f1_str:<12} {tall_results['evaluation_time']:<12.2f}")
        
        if 'accuracy' in tall_results and 'accuracy' in popcount_results:
            print(f"\nTALL vs Popcount improvement: {tall_results['accuracy'] - popcount_results['accuracy']:+.2f}%")
        if 'accuracy' in tall_results and 'accuracy' in fully_binary_results:
            print(f"TALL vs Fully Binary improvement: {tall_results['accuracy'] - fully_binary_results['accuracy']:+.2f}%")
    
    if 'accuracy' in fully_binary_results and 'accuracy' in popcount_results:
        print(f"Fully Binary vs Popcount difference: {fully_binary_results['accuracy'] - popcount_results['accuracy']:+.2f}%")
    
    # Show data usage mode
    if args.inference_mode or args.inference_data_dir:
        print(f"\n📊 Data Usage: INFERENCE MODE - All data used (NO train/test split)")
    else:
        print(f"\n📊 Data Usage: STANDARD MODE - Only test split used ({(1-args.train_ratio)*100:.0f}% of data)")
    
    # Save combined results
    combined_file = os.path.join(args.output_dir, 'combined_evaluation_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined evaluation results saved to {combined_file}")
    
    # TALL sensitivity analysis (only if TALL evaluation was run)
    if args.tall_sensitivity and args.use_tall:
        print("\n" + "="*70)
        print("TALL SENSITIVITY ANALYSIS")
        print("="*70)
        
        sensitivity_results = run_tall_sensitivity_analysis(model, test_loader, device, tall_dir)
        all_results['tall_sensitivity'] = sensitivity_results
    
    # Analyze example predictions
    if args.analyze_examples > 0:
        print("\n" + "="*70)
        print("EXAMPLE PREDICTION ANALYSIS")
        print("="*70)
        
        # Analyze popcount predictions
        print("Popcount model predictions:")
        analyze_sequence_predictions(model, test_loader, device, 
                                   use_tall=False, num_examples=args.analyze_examples)
        
        # Analyze fully binary predictions
        print(f"\nFully binary model predictions:")
        analyze_sequence_predictions(fully_binary_model, test_loader, device, 
                                   use_tall=False, num_examples=args.analyze_examples)
        
        # Analyze TALL predictions (if available)
        if args.use_tall and 'tall_model' in locals():
            print(f"\nTALL model predictions:")
            analyze_sequence_predictions(tall_model, test_loader, device, 
                                       use_tall=True, num_examples=args.analyze_examples)
    
    inference_mode_str = " (INFERENCE MODE)" if (args.inference_mode or args.inference_data_dir) else ""
    print(f"\nEvaluation completed{inference_mode_str}! Results organized in:")
    print(f"  📁 {args.output_dir}/")
    
    if args.inference_mode or args.inference_data_dir:
        print(f"     ├── 1_popcount_evaluation/ (predictions and confidences)")
        print(f"     ├── 2_fully_binary_evaluation/ (binary predictions)")
        if args.use_tall:
            print(f"     ├── 3_tall_evaluation/ (TALL predictions)")
        print(f"     └── combined_evaluation_results.json (all inference results)")
        print(f"\n💡 Inference Mode Usage:")
        print(f"   --inference_mode: Use for new unlabeled FASTA data")
        print(f"   --inference_data_dir: Specify different data directory for inference")
        print(f"   Supports both labeled (COVID/Not_COVID dirs) and unlabeled data")
    else:
        print(f"     ├── 1_popcount_evaluation/")
        print(f"     ├── 2_fully_binary_evaluation/") 
        print(f"     ├── 3_tall_evaluation/")
        print(f"     └── combined_evaluation_results.json")


if __name__ == '__main__':
    main()
