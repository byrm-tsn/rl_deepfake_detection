"""
GRPO Phase 3 Evaluation and Visualization Module
Generates comprehensive evaluation metrics and graphs for GRPO training
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GRPOEvaluator:
    """Comprehensive evaluation suite for Phase 3 GRPO training"""
    
    def __init__(self, 
                 grpo_model,
                 phase2_model,
                 env,
                 device: str = 'cuda',
                 output_dir: Path = None):
        """
        Initialize GRPO evaluator
        
        Args:
            grpo_model: GRPO fine-tuned model
            phase2_model: Original PPO model from Phase 2
            env: Evaluation environment
            device: Device for computation
            output_dir: Directory for saving results
        """
        self.grpo_model = grpo_model
        self.phase2_model = phase2_model
        self.env = env
        self.device = device
        self.output_dir = output_dir or Path('results/phase3_grpo')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _dict_to_tensor(self, obs_dict: Dict) -> torch.Tensor:
        """Convert observation dict to tensor matching expected format"""
        import numpy as np
        from src.config import OBSERVATION_COMPONENTS, STATE_DIM
        
        # If it's already a tensor or numpy array
        if isinstance(obs_dict, (torch.Tensor, np.ndarray)):
            if isinstance(obs_dict, np.ndarray):
                obs_tensor = torch.FloatTensor(obs_dict)
            else:
                obs_tensor = obs_dict
            
            # Verify dimension
            if obs_tensor.shape[-1] != STATE_DIM:
                raise ValueError(f"Observation dimension mismatch. Expected {STATE_DIM}, got {obs_tensor.shape[-1]}")
            return obs_tensor
        
        # Build observation tensor with correct dimensions from config
        # Expected format from config.py (1794 total):
        # frame_features: 768
        # temporal_memory: 1024
        # frame_position: 1
        # uncertainty: 1
        
        components = []
        
        # Add components in the expected order with correct dimensions
        for key, expected_dim in OBSERVATION_COMPONENTS.items():
            if key in obs_dict:
                component = np.array(obs_dict[key]).flatten()
                if len(component) != expected_dim:
                    # Pad or truncate to expected dimension
                    if len(component) < expected_dim:
                        component = np.pad(component, (0, expected_dim - len(component)))
                    else:
                        component = component[:expected_dim]
                components.append(component)
            else:
                # Add zeros if component is missing
                components.append(np.zeros(expected_dim))
        
        # Concatenate all components
        full_obs = np.concatenate(components)
        
        # Verify total dimension matches STATE_DIM (1794)
        if len(full_obs) != STATE_DIM:
            raise ValueError(f"Observation dimension mismatch. Expected {STATE_DIM}, got {len(full_obs)}")
        
        return torch.FloatTensor(full_obs)
    
    def plot_classification_metrics(self, num_episodes: int = 100) -> Dict:
        """
        Compute and plot comprehensive classification metrics including
        ROC AUC, F1, Precision, Recall, and Confusion Matrix.
        """
        logging.info("Computing classification metrics...")
        
        # Collect predictions and labels
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        for episode in range(num_episodes):
            # Handle both old and new Gym API
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  
            else:
                obs = reset_result  
            done = False
            
            while not done:
                # Get model prediction with probabilities
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_to_tensor(obs)
                    else:
                        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    
                    # Get action logits
                    if hasattr(self.grpo_model, 'forward'):
                        action_logits, _ = self.grpo_model.forward(obs_tensor)
                    else:
                        action_logits = self.grpo_model(obs_tensor)
                    
                    # Get probabilities
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                    
                    # Convert action to binary prediction
                    # Action 3 (STOP_REAL) -> 0, Action 4 (STOP_FAKE) -> 1
                    action_idx = action.cpu().numpy()[0]
                    
                    # Only record predictions for terminal actions
                    if action_idx in [3, 4]:  # STOP_REAL or STOP_FAKE
                        binary_pred = 0 if action_idx == 3 else 1
                        # For probability, use the confidence of the chosen action
                        fake_prob = probs[0, 4].cpu().numpy()  # Probability of STOP_FAKE action
                        all_probabilities.append(fake_prob)
                        all_predictions.append(binary_pred)
                        
                        # Get the true label when we make a prediction
                        if hasattr(self.env, 'current_label'):
                            all_labels.append(self.env.current_label)
                        elif hasattr(self.env, 'get_current_label'):
                            all_labels.append(self.env.get_current_label())
                
                # Step environment
                step_result = self.env.step(action_idx)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                # If we made a terminal decision and didn't get label yet, get it from info
                if action_idx in [3, 4] and len(all_labels) < len(all_predictions):
                    if 'true_label' in info:
                        all_labels.append(info['true_label'])
        
        # Check if we have valid predictions and labels
        if not all_predictions or not all_labels:
            logging.warning("No terminal predictions made during evaluation")
            # Return empty metrics
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': None,
                'f1_real': 0.0,
                'f1_fake': 0.0,
                'precision_real': 0.0,
                'precision_fake': 0.0,
                'recall_real': 0.0,
                'recall_fake': 0.0
            }
        
        # Convert to numpy arrays - ensure same length
        min_len = min(len(all_predictions), len(all_labels))
        predictions = np.array(all_predictions[:min_len])
        probabilities = np.array(all_probabilities[:min_len]) if all_probabilities else np.zeros(min_len)
        labels = np.array(all_labels[:min_len])
        
        # Compute metrics with safe defaults
        try:
            metrics = {
                'accuracy': np.mean(predictions == labels) if len(predictions) > 0 else 0.0,
                'f1_score': f1_score(labels, predictions, average='weighted') if len(predictions) > 0 else 0.0,
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0) if len(predictions) > 0 else 0.0,
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0) if len(predictions) > 0 else 0.0,
                'roc_auc': roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 and len(probabilities) > 0 else None
            }
            
            # Per-class metrics with error handling
            metrics['f1_real'] = f1_score(labels, predictions, pos_label=0) if 0 in labels else 0.0
            metrics['f1_fake'] = f1_score(labels, predictions, pos_label=1) if 1 in labels else 0.0
            metrics['precision_real'] = precision_score(labels, predictions, pos_label=0, zero_division=0) if 0 in labels else 0.0
            metrics['precision_fake'] = precision_score(labels, predictions, pos_label=1, zero_division=0) if 1 in labels else 0.0
            metrics['recall_real'] = recall_score(labels, predictions, pos_label=0, zero_division=0) if 0 in labels else 0.0
            metrics['recall_fake'] = recall_score(labels, predictions, pos_label=1, zero_division=0) if 1 in labels else 0.0
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            metrics = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': None,
                'f1_real': 0.0,
                'f1_fake': 0.0,
                'precision_real': 0.0,
                'precision_fake': 0.0,
                'recall_real': 0.0,
                'recall_fake': 0.0
            }
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Add percentages
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=9, color='gray')
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        if metrics['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(labels, probabilities)
            ax2.plot(fpr, tpr, 'b-', linewidth=2, 
                    label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
            ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        if len(np.unique(labels)) == 2:
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            avg_precision = average_precision_score(labels, probabilities)
            ax3.plot(recall, precision, 'g-', linewidth=2,
                    label=f'PR Curve (AP = {avg_precision:.3f})')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax3.legend(loc='lower left')
            ax3.grid(True, alpha=0.3)
        
        # 4. Class-wise Metrics Bar Plot
        ax4 = plt.subplot(2, 3, 4)
        metrics_data = {
            'F1 Score': [metrics['f1_real'], metrics['f1_fake']],
            'Precision': [metrics['precision_real'], metrics['precision_fake']],
            'Recall': [metrics['recall_real'], metrics['recall_fake']]
        }
        x = np.arange(2)
        width = 0.25
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax4.bar(x + i * width, values, width, label=metric_name)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Score')
        ax4.set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(['Real', 'Fake'])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            for j, v in enumerate(values):
                ax4.text(j + i * width, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 5. Confidence Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(probabilities[labels == 0], bins=30, alpha=0.5, label='Real', color='blue')
        ax5.hist(probabilities[labels == 1], bins=30, alpha=0.5, label='Fake', color='red')
        ax5.set_xlabel('Predicted Fake Probability')
        ax5.set_ylabel('Count')
        ax5.set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Overall Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Format ROC AUC value
        roc_auc_str = f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "N/A"
        
        summary_text = f"""PHASE 3 GRPO EVALUATION SUMMARY

Overall Metrics:
• Accuracy: {metrics['accuracy']:.3f}
• F1 Score (weighted): {metrics['f1_score']:.3f}
• Precision (weighted): {metrics['precision']:.3f}
• Recall (weighted): {metrics['recall']:.3f}
• ROC AUC: {roc_auc_str}

Real Class Performance:
• F1 Score: {metrics['f1_real']:.3f}
• Precision: {metrics['precision_real']:.3f}
• Recall: {metrics['recall_real']:.3f}

Fake Class Performance:
• F1 Score: {metrics['f1_fake']:.3f}
• Precision: {metrics['precision_fake']:.3f}
• Recall: {metrics['recall_fake']:.3f}

Confusion Matrix:
• True Positives (Fake): {cm[1, 1]}
• True Negatives (Real): {cm[0, 0]}
• False Positives: {cm[0, 1]}
• False Negatives: {cm[1, 0]}

Total Samples: {len(labels)}"""
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        ax6.set_title('Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Phase 3 GRPO - Classification Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / 'classification_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Classification metrics saved to {save_path}")
        
        # Generate classification report
        report = classification_report(labels, predictions, 
                                      target_names=['Real', 'Fake'],
                                      output_dict=True)
        
        # Save detailed report
        report_path = self.output_dir / 'classification_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'num_samples': len(labels),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return metrics
    
    def plot_group_relative_performance(self, num_episodes: int = 100) -> Dict:
        """
        Analyze group relative performance - core GRPO mechanism
        """
        logging.info("Analyzing group relative performance...")
        
        # Collect rewards in groups
        group_size = 8
        all_rewards = []
        
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  
            else:
                obs = reset_result  
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_to_tensor(obs).to(self.device).unsqueeze(0)
                    else:
                        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    
                    if hasattr(self.grpo_model, 'forward'):
                        action_logits, _ = self.grpo_model.forward(obs_tensor)
                    else:
                        action_logits = self.grpo_model(obs_tensor)
                    
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                
                step_result = self.env.step(action.cpu().numpy()[0])
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_reward += reward
            
            all_rewards.append(episode_reward)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GRPO Group Relative Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Reward distribution
        ax1 = axes[0, 0]
        ax1.hist(all_rewards, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(all_rewards), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_rewards):.2f}')
        ax1.set_xlabel('Episode Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution')
        ax1.legend()
        
        # Plot 2: Group advantages
        ax2 = axes[0, 1]
        num_groups = len(all_rewards) // group_size
        group_advantages = []
        
        for i in range(num_groups):
            group_rewards = all_rewards[i*group_size:(i+1)*group_size]
            if len(group_rewards) == group_size:
                mean_reward = np.mean(group_rewards)
                advantages = [r - mean_reward for r in group_rewards]
                group_advantages.extend(advantages)
        
        ax2.hist(group_advantages, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Advantage Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Group-Relative Advantages (Group Size={group_size})')
        
        # Plot 3: Reward progression
        ax3 = axes[1, 0]
        ax3.plot(all_rewards, alpha=0.5)
        window = min(20, len(all_rewards) // 5)
        rolling_mean = pd.Series(all_rewards).rolling(window=window, min_periods=1).mean()
        ax3.plot(rolling_mean, color='red', linewidth=2, label=f'Rolling Mean (w={window})')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward Evolution')
        ax3.legend()
        
        # Plot 4: Group performance heatmap
        ax4 = axes[1, 1]
        max_groups = min(20, num_groups)
        heatmap_data = []
        
        for i in range(max_groups):
            group_rewards = all_rewards[i*group_size:(i+1)*group_size]
            if len(group_rewards) == group_size:
                mean_reward = np.mean(group_rewards)
                std_reward = np.std(group_rewards) + 1e-8
                relative_rewards = [(r - mean_reward) / std_reward for r in group_rewards]
                heatmap_data.append(relative_rewards)
        
        if heatmap_data:
            sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax4,
                       cbar_kws={'label': 'Normalized Advantage'})
            ax4.set_xlabel('Trajectory in Group')
            ax4.set_ylabel('Group Index')
            ax4.set_title('Group-Relative Advantages Heatmap')
        
        plt.tight_layout()
        save_path = self.output_dir / 'group_relative_performance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {'mean_reward': np.mean(all_rewards), 'std_reward': np.std(all_rewards)}
    
    def plot_phase_comparison(self, num_episodes: int = 50) -> Dict:
        """
        Compare Phase 2 vs Phase 3 performance
        """
        logging.info("Comparing Phase 2 vs Phase 3 performance...")
        
        # Evaluate both models
        phase2_rewards = []
        phase3_rewards = []
        
        for episode in range(num_episodes):
            # Phase 2 evaluation
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action, _ = self.phase2_model.predict(obs, deterministic=True)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_reward += reward
            
            phase2_rewards.append(episode_reward)
            
            # Phase 3 evaluation
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_to_tensor(obs).to(self.device).unsqueeze(0)
                    else:
                        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    
                    if hasattr(self.grpo_model, 'forward'):
                        action_logits, _ = self.grpo_model.forward(obs_tensor)
                    else:
                        action_logits = self.grpo_model(obs_tensor)
                    
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                
                step_result = self.env.step(action.cpu().numpy()[0])
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_reward += reward
            
            phase3_rewards.append(episode_reward)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Phase 2 (PPO) vs Phase 3 (GRPO) Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Reward distributions
        ax1 = axes[0]
        ax1.hist(phase2_rewards, bins=20, alpha=0.5, label='Phase 2', color='blue')
        ax1.hist(phase3_rewards, bins=20, alpha=0.5, label='Phase 3', color='red')
        ax1.axvline(np.mean(phase2_rewards), color='blue', linestyle='--', linewidth=2)
        ax1.axvline(np.mean(phase3_rewards), color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Episode Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distributions')
        ax1.legend()
        
        # Plot 2: Performance metrics
        ax2 = axes[1]
        metrics = ['Mean Reward', 'Std Reward', 'Max Reward']
        phase2_values = [np.mean(phase2_rewards), np.std(phase2_rewards), np.max(phase2_rewards)]
        phase3_values = [np.mean(phase3_rewards), np.std(phase3_rewards), np.max(phase3_rewards)]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, phase2_values, width, label='Phase 2', color='blue')
        ax2.bar(x + width/2, phase3_values, width, label='Phase 3', color='red')
        ax2.set_ylabel('Value')
        ax2.set_title('Performance Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        
        # Plot 3: Improvement
        ax3 = axes[2]
        improvements = [(p3 - p2) / abs(p2) * 100 if p2 != 0 else 0 
                       for p2, p3 in zip(phase2_values, phase3_values)]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax3.bar(metrics, improvements, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Relative Performance Gains')
        ax3.set_xticklabels(metrics, rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / 'phase_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'phase2_mean': np.mean(phase2_rewards),
            'phase3_mean': np.mean(phase3_rewards),
            'improvement': np.mean(phase3_rewards) - np.mean(phase2_rewards)
        }
    
    def plot_training_dynamics(self, training_history: Dict) -> None:
        """
        Plot training dynamics including losses and entropy
        """
        if not training_history:
            logging.warning("No training history provided")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Training Dynamics', fontsize=16, fontweight='bold')
        
        # Plot 1: Policy loss
        ax1 = axes[0, 0]
        if 'policy_losses' in training_history:
            ax1.plot(training_history['policy_losses'], alpha=0.7)
            ax1.set_xlabel('Update Step')
            ax1.set_ylabel('Policy Loss')
            ax1.set_title('Policy Loss Evolution')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Value loss
        ax2 = axes[0, 1]
        if 'value_losses' in training_history:
            ax2.plot(training_history['value_losses'], alpha=0.7, color='red')
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Value Loss')
            ax2.set_title('Value Loss Evolution')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Entropy
        ax3 = axes[1, 0]
        if 'entropies' in training_history:
            ax3.plot(training_history['entropies'], alpha=0.7, color='green')
            ax3.axhspan(1.3, 1.5, alpha=0.2, color='green')
            ax3.set_xlabel('Update Step')
            ax3.set_ylabel('Entropy')
            ax3.set_title('Entropy Evolution')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gradient norms
        ax4 = axes[1, 1]
        if 'gradient_norms' in training_history:
            ax4.plot(training_history['gradient_norms'], alpha=0.7, color='purple')
            ax4.axhline(y=0.5, color='red', linestyle='--')
            ax4.set_xlabel('Update Step')
            ax4.set_ylabel('Gradient Norm')
            ax4.set_title('Gradient Stability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'training_dynamics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_action_distribution_evolution(self, num_episodes: int = 100) -> Dict:
        """
        Analyze action distribution changes
        """
        logging.info("Analyzing action distribution evolution...")
        
        # Collect action statistics
        action_counts = defaultdict(int)
        total_actions = 0
        
        for episode in range(num_episodes):
            # Handle both old and new Gym API
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result 
            else:
                obs = reset_result  
            done = False
            
            while not done:
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_to_tensor(obs).to(self.device).unsqueeze(0)
                    else:
                        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    
                    if hasattr(self.grpo_model, 'forward'):
                        action_logits, _ = self.grpo_model.forward(obs_tensor)
                    else:
                        action_logits = self.grpo_model(obs_tensor)
                    
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                
                action_counts[action.item()] += 1
                total_actions += 1
                
                step_result = self.env.step(action.cpu().numpy()[0])
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
        
        # Create visualization
        action_names = ['NEXT', 'FOCUS', 'AUGMENT', 'STOP_REAL', 'STOP_FAKE']
        action_probs = [action_counts.get(i, 0) / total_actions for i in range(5)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(action_names)))
        bars = ax.bar(action_names, action_probs, color=colors)
        
        ax.set_ylabel('Probability')
        ax.set_title('Action Distribution (Phase 3 GRPO)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(action_probs) * 1.2])
        
        # Add value labels
        for bar, prob in zip(bars, action_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / 'action_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {'action_distribution': dict(zip(action_names, action_probs))}
    
    def plot_confidence_calibration(self, num_episodes: int = 100) -> Dict:
        """
        Analyze confidence calibration
        """
        logging.info("Analyzing confidence calibration...")
        
        # Collect predictions and confidences
        all_predictions = []
        all_confidences = []
        all_labels = []
        
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result 
            else:
                obs = reset_result  
            done = False
            
            while not done:
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_to_tensor(obs).to(self.device).unsqueeze(0)
                    else:
                        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    
                    if hasattr(self.grpo_model, 'forward'):
                        action_logits, _ = self.grpo_model.forward(obs_tensor)
                    else:
                        action_logits = self.grpo_model(obs_tensor)
                    
                    probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(probs, dim=-1)
                    confidence = torch.max(probs).cpu().numpy()
                    
                    # Convert action to binary prediction for terminal actions
                    action_idx = action.cpu().numpy()[0]
                    if action_idx in [3, 4]:  # STOP_REAL or STOP_FAKE
                        binary_pred = 0 if action_idx == 3 else 1
                        all_predictions.append(binary_pred)
                        all_confidences.append(confidence)
                        
                        # Get the true label when we make a prediction
                        if hasattr(self.env, 'current_label'):
                            all_labels.append(self.env.current_label)
                        elif hasattr(self.env, 'get_current_label'):
                            all_labels.append(self.env.get_current_label())
                
                step_result = self.env.step(action_idx)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                # If we made a terminal decision and didn't get label yet, get it from info
                if action_idx in [3, 4] and len(all_labels) < len(all_predictions):
                    if 'true_label' in info:
                        all_labels.append(info['true_label'])
        
        if not all_labels:
            logging.warning("No labels found for calibration analysis")
            return {}
        
        predictions = np.array(all_predictions[:len(all_labels)])
        confidences = np.array(all_confidences[:len(all_labels)])
        labels = np.array(all_labels)
        
        # Calculate calibration
        correct = predictions == labels
        
        # Create calibration plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Confidence Calibration Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Calibration curve
        ax1 = axes[0]
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(correct[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'ro-', linewidth=2, markersize=8, label='GRPO Model')
        ax1.set_xlabel('Mean Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence histogram
        ax2 = axes[1]
        ax2.hist(confidences[correct], bins=20, alpha=0.5, label='Correct', color='green')
        ax2.hist(confidences[~correct], bins=20, alpha=0.5, label='Incorrect', color='red')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'confidence_calibration.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate ECE
        ece = 0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(correct[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
        
        return {'ece': ece, 'mean_confidence': np.mean(confidences)}
    
    def generate_evaluation_summary(self) -> Dict:
        """
        Generate evaluation summary report
        """
        summary = {
            'phase': 'phase3_grpo',
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir)
        }
        
        # Save summary
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Evaluation summary saved to {self.output_dir / 'evaluation_summary.json'}")
        
        return summary
    
    def run_full_evaluation(self, num_episodes: int = 50, 
                           training_history: Optional[Dict] = None):
        """
        Run complete evaluation suite for Phase 3 GRPO.
        """
        logging.info("\n" + "=" * 80)
        logging.info("📊 Running comprehensive GRPO evaluation...")
        logging.info("=" * 80)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # High priority visualizations
        logging.info("\n1️⃣ Generating group relative performance analysis...")
        group_metrics = self.plot_group_relative_performance(num_episodes)
        
        logging.info("\n2️⃣ Comparing Phase 2 vs Phase 3 performance...")
        comparison_metrics = self.plot_phase_comparison(num_episodes)
        
        logging.info("\n3️⃣ Plotting training dynamics...")
        if training_history:
            self.plot_training_dynamics(training_history)
        
        logging.info("\n4️⃣ Analyzing action distribution evolution...")
        action_metrics = self.plot_action_distribution_evolution(num_episodes)
        
        logging.info("\n5️⃣ Computing classification metrics (ROC, F1, Confusion Matrix)...")
        classification_metrics = self.plot_classification_metrics(num_episodes)
        
        logging.info("\n6️⃣ Evaluating confidence calibration...")
        calibration_metrics = self.plot_confidence_calibration(num_episodes)
        
        # Generate summary report
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'group_metrics': group_metrics,
            'comparison_metrics': comparison_metrics,
            'action_metrics': action_metrics,
            'classification_metrics': classification_metrics,
            'calibration_metrics': calibration_metrics,
            'output_directory': str(self.output_dir)
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save summary with proper type conversion
        serializable_summary = convert_to_serializable(summary)
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        logging.info("\n" + "=" * 80)
        logging.info("✅ GRPO evaluation completed successfully!")
        logging.info(f"📁 Results saved to: {self.output_dir}")
        logging.info("=" * 80)
        
        return summary