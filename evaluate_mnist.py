#!/usr/bin/env python3
"""
Evaluation script for trained BNN models on MNIST
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import numpy as np
from tqdm import tqdm

from BNN_model import BinaryMLP, TALLClassifier, build_cam4_deep, build_cam4_shallow, build_cam4_deep_fully_binary, build_cam4_shallow_fully_binary


def get_mnist_test_loader(batch_size=1000, data_dir='./data'):
    """Get MNIST test data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return test_loader


def evaluate_model(model, test_loader, device, use_tall=False, verbose=True):
    """Evaluate model and return detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating', disable=not verbose):
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'backbone'):
                # Use TALL voting
                pred = model(data)  # Already returns class indices
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Per-class accuracy
            for i in range(10):
                class_mask = (target == i)
                if class_mask.sum() > 0:
                    class_correct[i] += pred[class_mask].eq(target[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()
    
    eval_time = time.time() - start_time
    accuracy = 100. * correct / total
    
    if verbose:
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Evaluation time: {eval_time:.2f}s")
        print("\nPer-class accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"  Class {i}: {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    return accuracy, class_correct / class_total


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained BNN on MNIST')
    parser.add_argument('--model-path', type=str, required=False,
                        help='path to trained model checkpoint', default='./bnn_deep_best.pth')
    parser.add_argument('--model-type', type=str, default='deep', choices=['deep', 'shallow'],
                        help='model architecture (default: deep)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='batch size for evaluation (default: 1000)')
    parser.add_argument('--use-tall', action='store_true', default=False,
                        help='use TALL voting for inference')
    parser.add_argument('--tall-iter', type=int, default=30,
                        help='number of TALL iterations (default: 30)')
    parser.add_argument('--tall-flip-p', type=float, default=0.3,
                        help='TALL bit flip probability (default: 0.3)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory with MNIST data')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading MNIST test set...")
    test_loader = get_mnist_test_loader(batch_size=args.batch_size, data_dir=args.data_dir)
    
    # Create model
    print(f"Creating {args.model_type} BNN model...")
    if args.model_type == 'deep':
        model = build_cam4_deep_fully_binary(num_classes=10)
    else:
        model = build_cam4_shallow_fully_binary(num_classes=10)

    # Load trained weights
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best training accuracy: {checkpoint.get('best_acc', 'unknown'):.2f}%")
    
    # Evaluate with standard inference
    print("\n" + "="*60)
    print("STANDARD INFERENCE")
    print("="*60)
    std_acc, _ = evaluate_model(model, test_loader, device, use_tall=False)
    
    # Evaluate with TALL if requested or compare both
    if args.use_tall:
        print("\n" + "="*60)
        print("TALL VOTING INFERENCE")
        print("="*60)
        if args.model_type == 'deep':
            binary_model = build_cam4_deep_fully_binary(num_classes=10)
        else:
            binary_model = build_cam4_shallow_fully_binary(num_classes=10)

        binary_model.load_state_dict(checkpoint['model_state_dict'])
        binary_model = binary_model.to(device)
        tall_model = TALLClassifier(
            binary_model, num_iter=args.tall_iter, flip_p=args.tall_flip_p, majority_threshold=0.95
        ).to(device)
        tall_acc, _ = evaluate_model(tall_model, test_loader, device, use_tall=True)
        
        print(f"\nComparison:")
        print(f"Standard inference: {std_acc:.2f}%")
        print(f"TALL voting:        {tall_acc:.2f}%")
        print(f"Improvement:        {tall_acc - std_acc:+.2f}%")
    
    # Test with different TALL parameters
    if args.use_tall:
        print("\n" + "="*60)
        print("TALL PARAMETER SWEEP")
        print("="*60)
        
        flip_probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] 
        iterations = [1, 5, 10 , 20 , 30, 40, 50, 100]

        print("Testing all combinations of flip probabilities and iteration counts:")
        print("Format: \n flip_p | " + " | ".join(f"{i:3d}" for i in iterations))
        print("-" * 65)
        
        best_acc = 0
        best_params = None
        results = {}
        
        for flip_p in flip_probs:
            row_results = []
            for num_iter in iterations:
                tall_model = TALLClassifier(binary_model, num_iter=num_iter, flip_p=flip_p, majority_threshold=0.95).to(device)
                acc, _ = evaluate_model(tall_model, test_loader, device, use_tall=True, verbose=False)
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
        
        print("-" * 65)
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
