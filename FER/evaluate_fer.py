#!/usr/bin/env python3
"""
Evaluation script for trained BNN models on FER2013 dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from FER.train_fer import FER2013Dataset, build_fer_model
from BNN_model import BinaryMLP, TALLClassifier


def get_fer_test_loader(data_dir, batch_size=256, split='test'):
    """Get FER2013 test data loader"""
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_dataset = FER2013Dataset(data_dir, split=split, transform=transform)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return test_loader


def evaluate_model(model, test_loader, device, use_tall=False, verbose=True):
    """Evaluate model and return detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    class_correct = np.zeros(7)  # 7 emotion classes
    class_total = np.zeros(7)
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating', disable=not verbose):
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'backbone'):
                pred = model(data)  # TALL returns class indices directly
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
    
    eval_time = time.time() - start_time
    accuracy = 100. * correct / total
    
    if verbose:
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Evaluation time: {eval_time:.2f}s")
        print("\nPer-class accuracy:")
        for i in range(7):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"  {i} ({emotion_names[i]}): {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
            else:
                print(f"  {i} ({emotion_names[i]}): N/A (no samples)")
    
    return accuracy, class_correct / np.maximum(class_total, 1), all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix - FER2013')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained BNN on FER2013')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/FER2013',
                        help='path to FER2013 dataset directory (default: data/FER2013)')
    parser.add_argument('--model-type', type=str, default='shallow', choices=['deep', 'shallow'],
                        help='model architecture (default: deep)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for evaluation (default: 256)')
    parser.add_argument('--use-tall', action='store_true', default=False,
                        help='use TALL voting for inference')
    parser.add_argument('--tall-iter', type=int, default=30,
                        help='number of TALL iterations (default: 30)')
    parser.add_argument('--tall-flip-p', type=float, default=0.3,
                        help='TALL bit flip probability (default: 0.3)')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'test'],
                        help='which dataset split to evaluate on (default: test)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='directory to save outputs (default: ./outputs)')
    parser.add_argument('--plot-confusion', action='store_true', default=True,
                        help='plot confusion matrix')
    parser.add_argument('--detailed-report', action='store_true', default=True,
                        help='show detailed classification report')
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading FER2013 {args.split} set...")
    from FER.train_fer import check_fer2013_folder_structure
    if not check_fer2013_folder_structure(args.data_dir):
        print("Please ensure FER2013 dataset is properly downloaded with train/test folder structure.")
        return
    test_loader = get_fer_test_loader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        split=args.split
    )
    
    # Create model
    print(f"Creating {args.model_type} BNN model...")
    model = build_fer_model(model_type=args.model_type, num_classes=7)
    
    # Load trained weights
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
    print(f"Best test accuracy: {checkpoint.get('best_test_acc', 'unknown'):.2f}%")
    
    # Evaluate with standard inference
    print("\n" + "="*60)
    print("STANDARD INFERENCE")
    print("="*60)
    std_acc, class_accs, predictions, targets = evaluate_model(
        model, test_loader, device, use_tall=False
    )
    
    # Evaluate with TALL if requested or compare both
    if args.use_tall:
        print("\n" + "="*60)
        print("TALL VOTING INFERENCE")
        print("="*60)
        tall_model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
        tall_acc, _, tall_predictions, _ = evaluate_model(
            tall_model, test_loader, device, use_tall=True
        )
        
        print(f"\nComparison:")
        print(f"Standard inference: {std_acc:.2f}%")
        print(f"TALL voting:        {tall_acc:.2f}%")
        print(f"Improvement:        {tall_acc - std_acc:+.2f}%")
    
    # Detailed classification report
    if args.detailed_report:
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        if args.use_tall:
            report = classification_report(
                targets, tall_predictions, 
                target_names=emotion_names, 
                digits=4
            )
        else:
            report = classification_report(
                targets, predictions, 
                target_names=emotion_names, 
                digits=4
            )
        
        print(report)
    
    # Plot confusion matrix
    if args.plot_confusion:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.use_tall:
            cm_path = os.path.join(args.output_dir, f"fer_{args.model_type}_tall_confusion_matrix.png")
            plot_confusion_matrix(targets, tall_predictions, cm_path)
        else:
            cm_path = os.path.join(args.output_dir, f"fer_{args.model_type}_confusion_matrix.png")
            plot_confusion_matrix(targets, predictions, cm_path)
    
    # Test with different TALL parameters if enabled
    if args.use_tall:
        print("\n" + "="*60)
        print("TALL PARAMETER SWEEP")
        print("="*60)
        
        flip_probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        iterations = [1, 5, 10, 20, 30, 50, 100, 200]
        
        print("Testing combinations of flip probabilities and iteration counts:")
        print("Format: flip_p | " + " | ".join(f"{i:3d} iter" for i in iterations))
        print("-" * 70)
        
        best_acc = 0
        best_params = None
        results = {}
        
        for flip_p in flip_probs:
            row_results = []
            for num_iter in iterations:
                tall_model = TALLClassifier(model, num_iter=num_iter, flip_p=flip_p).to(device)
                acc, _, _, _ = evaluate_model(tall_model, test_loader, device, use_tall=True, verbose=False)
                row_results.append(acc)
                results[(flip_p, num_iter)] = acc
                
                # Track best combination
                if acc > best_acc:
                    best_acc = acc
                    best_params = (flip_p, num_iter)
            
            # Print row with all iteration results for this flip_p
            row_str = f"{flip_p:5.2f} |"
            for acc in row_results:
                row_str += f" {acc:6.2f}% |"
            print(row_str)
        
        print("-" * 70)
        print(f"Best combination: flip_p={best_params[0]:.2f}, iter={best_params[1]}, accuracy={best_acc:.2f}%")
        
        # Also show individual parameter analysis
        print("\nParameter analysis:")
        print("Average accuracy by flip probability:")
        for flip_p in flip_probs:
            avg_acc = np.mean([results[(flip_p, num_iter)] for num_iter in iterations])
            print(f"  flip_p={flip_p:.2f}: {avg_acc:.2f}%")
        
        print("\nAverage accuracy by iteration count:")
        for num_iter in iterations:
            avg_acc = np.mean([results[(flip_p, num_iter)] for flip_p in flip_probs])
            print(f"  iter={num_iter:3d}: {avg_acc:.2f}%")


if __name__ == '__main__':
    main()
