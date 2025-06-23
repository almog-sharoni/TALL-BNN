#!/usr/bin/env python3
"""
Evaluation script for trained BNN models on KDEF dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from BNN_model import BinaryMLP, TALLClassifier
from train_kdef import KDEFDataset, get_kdef_transforms, build_kdef_model


def evaluate_model_detailed(model, test_loader, device, emotion_names, use_tall=False, verbose=True):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating', disable=not verbose):
            data, target = data.to(device), target.to(device)
            
            if use_tall and hasattr(model, 'backbone'):
                pred = model(data)  # TALL returns class indices
            else:
                output = model(data)
                pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    
    # Calculate per-class metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Classification report
    report = classification_report(
        all_targets, all_preds, 
        target_names=emotion_names, 
        output_dict=True
    )
    
    if verbose:
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print("\nPer-class accuracy:")
        for i, emotion in enumerate(emotion_names):
            class_mask = all_targets == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(all_preds[class_mask] == all_targets[class_mask]) * 100
                print(f"  {emotion}: {class_acc:.2f}% ({np.sum(all_preds[class_mask] == all_targets[class_mask])}/{np.sum(class_mask)})")
        
        print(f"\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=emotion_names))
    
    return accuracy, cm, report, all_preds, all_targets


def plot_confusion_matrix(cm, emotion_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def analyze_misclassifications(all_preds, all_targets, emotion_names, test_dataset):
    """Analyze common misclassifications"""
    print("\nMisclassification Analysis:")
    print("="*50)
    
    misclass_count = {}
    for true_idx, pred_idx in zip(all_targets, all_preds):
        if true_idx != pred_idx:
            pair = (emotion_names[true_idx], emotion_names[pred_idx])
            misclass_count[pair] = misclass_count.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_misclass = sorted(misclass_count.items(), key=lambda x: x[1], reverse=True)
    
    print("Most common misclassifications:")
    for (true_emotion, pred_emotion), count in sorted_misclass[:10]:
        total_true = np.sum(np.array(all_targets) == emotion_names.index(true_emotion))
        percentage = (count / total_true) * 100
        print(f"  {true_emotion} -> {pred_emotion}: {count} times ({percentage:.1f}% of {true_emotion})")


def evaluate_by_gender(model, test_dataset, device, emotion_names, batch_size=64):
    """Evaluate model performance by gender"""
    print("\nGender-based Analysis:")
    print("="*30)
    
    # Separate samples by gender
    male_samples = [s for s in test_dataset.samples if s['gender'] == 0]
    female_samples = [s for s in test_dataset.samples if s['gender'] == 1]
    
    print(f"Male samples: {len(male_samples)}")
    print(f"Female samples: {len(female_samples)}")
    
    for gender_name, samples in [("Male", male_samples), ("Female", female_samples)]:
        if len(samples) == 0:
            continue
            
        # Create temporary dataset
        temp_dataset = KDEFDataset(test_dataset.root_dir, transform=test_dataset.transform)
        temp_dataset.samples = samples
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        accuracy, _, _, _, _ = evaluate_model_detailed(
            model, temp_loader, device, emotion_names, verbose=False
        )
        print(f"{gender_name} accuracy: {accuracy:.2f}%")


def evaluate_by_angle(model, test_dataset, device, emotion_names, batch_size=64):
    """Evaluate model performance by camera angle"""
    print("\nAngle-based Analysis:")
    print("="*25)
    
    angle_names = {'S': 'Straight', 'FL': '45° Left', 'FR': '45° Right', 
                   'HL': '90° Left', 'HR': '90° Right'}
    
    # Group samples by angle
    angle_samples = {}
    for sample in test_dataset.samples:
        angle = sample['angle']
        if angle not in angle_samples:
            angle_samples[angle] = []
        angle_samples[angle].append(sample)
    
    for angle, samples in angle_samples.items():
        if len(samples) == 0:
            continue
            
        angle_name = angle_names.get(angle, angle)
        print(f"{angle_name} ({angle}): {len(samples)} samples")
        
        # Create temporary dataset
        temp_dataset = KDEFDataset(test_dataset.root_dir, transform=test_dataset.transform)
        temp_dataset.samples = samples
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        accuracy, _, _, _, _ = evaluate_model_detailed(
            model, temp_loader, device, emotion_names, verbose=False
        )
        print(f"  Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained BNN on KDEF')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='directory containing KDEF images')
    parser.add_argument('--model-type', type=str, default='deep', choices=['deep', 'shallow'],
                        help='model architecture (default: deep)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for evaluation (default: 64)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='input image size (default: 224)')
    parser.add_argument('--use-tall', action='store_true', default=False,
                        help='use TALL voting for inference')
    parser.add_argument('--tall-iter', type=int, default=30,
                        help='number of TALL iterations (default: 30)')
    parser.add_argument('--tall-flip-p', type=float, default=0.3,
                        help='TALL bit flip probability (default: 0.3)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='test set size (default: 0.2)')
    parser.add_argument('--angles', nargs='+', default=['S'],
                        choices=['S', 'FL', 'FR', 'HL', 'HR'],
                        help='which camera angles to include (default: S)')
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='save evaluation plots')
    parser.add_argument('--detailed-analysis', action='store_true', default=True,
                        help='perform detailed analysis by gender and angle')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Create model
    model = build_kdef_model(
        model_type=args.model_type,
        image_size=args.image_size,
        num_classes=7
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create TALL wrapper if requested
    if args.use_tall:
        print(f"Using TALL voting (iter={args.tall_iter}, flip_p={args.tall_flip_p})")
        model = TALLClassifier(
            model, num_iter=args.tall_iter, flip_p=args.tall_flip_p
        ).to(device)
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best training accuracy: {checkpoint.get('best_acc', 'unknown'):.2f}%")
    
    # Load test data
    print("Loading KDEF test dataset...")
    _, test_transform = get_kdef_transforms(args.image_size, augment=False)
    
    # Load dataset information using the class method
    image_paths, labels, subjects, metadata = KDEFDataset.load_dataset_info(args.data_dir, angles=args.angles)
    
    if len(image_paths) == 0:
        print(f"No KDEF images found in {args.data_dir}")
        return
    
    # Use same split as training
    train_indices, test_indices = train_test_split(
        range(len(image_paths)), 
        test_size=args.test_size, 
        random_state=42,  # Same as training
        stratify=labels
    )
    
    # Create test dataset with proper constructor
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_subjects_list = [subjects[i] for i in test_indices]
    test_metadata = [metadata[i] for i in test_indices]
    test_dataset = KDEFDataset(test_paths, test_labels, test_subjects_list, transform=test_transform)
    
    # Store metadata for analysis functions
    test_dataset.metadata = test_metadata
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Emotion names
    emotion_names = ['angry', 'disgusted', 'afraid', 'happy', 'neutral', 'sad', 'surprised']
    
    # Evaluate model
    print("\nEvaluating model...")
    start_time = time.time()
    accuracy, cm, report, all_preds, all_targets = evaluate_model_detailed(
        model, test_loader, device, emotion_names, use_tall=args.use_tall
    )
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {eval_time:.1f} seconds")
    
    # Plot confusion matrix
    if args.save_plots:
        plot_confusion_matrix(cm, emotion_names, 'kdef_confusion_matrix.png')
    
    # Analyze misclassifications
    analyze_misclassifications(all_preds, all_targets, emotion_names, test_dataset)
    
    # Detailed analysis
    if args.detailed_analysis:
        evaluate_by_gender(model, test_dataset, device, emotion_names, args.batch_size)
        evaluate_by_angle(model, test_dataset, device, emotion_names, args.batch_size)
    
    # Save detailed results
    results = {
        'overall_accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'args': vars(args),
        'eval_time': eval_time
    }
    
    import json
    with open('kdef_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to kdef_evaluation_results.json")


if __name__ == '__main__':
    main()
