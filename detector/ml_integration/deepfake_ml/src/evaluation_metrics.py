"""
Comprehensive evaluation metrics and visualization for EAGER training phases.
Generates confusion matrices, ROC curves, and detailed performance reports.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PhaseEvaluator:
    """
    Comprehensive evaluator for each training phase.
    Generates detailed metrics and visualizations.
    """
    
    def __init__(self, phase_name: str, save_dir: Path):
        """
        Initialize phase evaluator.
        
        Args:
            phase_name: Name of the phase (e.g., "phase1_warmstart")
            save_dir: Directory to save evaluation results
        """
        self.phase_name = phase_name
        self.save_dir = Path(save_dir) / phase_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all stored metrics."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.video_ids = []
        self.confidences = []
        
        # Per-epoch metrics
        self.epoch_metrics = []
        
    def add_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
        video_ids: List[str] = None,
        confidences: torch.Tensor = None
    ):
        """
        Add a batch of predictions for evaluation.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
            video_ids: Optional video identifiers
            confidences: Optional confidence scores
        """
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy().tolist())
        
        if video_ids is not None:
            self.video_ids.extend(video_ids)
        
        if confidences is not None:
            self.confidences.extend(confidences.cpu().numpy().tolist())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        metrics['precision_real'] = precision_score(y_true, y_pred, pos_label=0)
        metrics['recall_real'] = recall_score(y_true, y_pred, pos_label=0)
        metrics['f1_real'] = f1_score(y_true, y_pred, pos_label=0)
        
        metrics['precision_fake'] = precision_score(y_true, y_pred, pos_label=1)
        metrics['recall_fake'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['f1_fake'] = f1_score(y_true, y_pred, pos_label=1)
        
        # If probabilities available, compute AUC
        if self.probabilities:
            y_prob = np.array(self.probabilities)
            if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics['auc'] = auc(fpr, tpr)
                metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
        
        # Confidence metrics if available
        if self.confidences:
            confidences = np.array(self.confidences)
            metrics['mean_confidence'] = np.mean(confidences)
            metrics['std_confidence'] = np.std(confidences)
            
            # Confidence calibration
            correct = (y_true == y_pred)
            metrics['mean_confidence_correct'] = np.mean(confidences[correct]) if any(correct) else 0
            metrics['mean_confidence_incorrect'] = np.mean(confidences[~correct]) if any(~correct) else 0
        
        return metrics
    
    def generate_confusion_matrix(self, save_path: Optional[Path] = None) -> np.ndarray:
        """
        Generate and optionally save confusion matrix.
        
        Args:
            save_path: Path to save the confusion matrix plot
            
        Returns:
            Confusion matrix array
        """
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if save_path is None:
            save_path = self.save_dir / 'confusion_matrix.png'
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {self.phase_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7,
                        f'{cm_normalized[i, j]:.1%}',
                        ha='center', va='center',
                        color='gray', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return cm
    
    def generate_roc_curve(self, save_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate and save ROC curve.
        
        Args:
            save_path: Path to save the ROC curve plot
            
        Returns:
            FPR, TPR, and AUC score
        """
        if not self.probabilities:
            logger.warning("No probabilities available for ROC curve")
            return None, None, None
        
        y_true = np.array(self.labels)
        y_prob = np.array(self.probabilities)
        
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1] 
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        if save_path is None:
            save_path = self.save_dir / 'roc_curve.png'
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.phase_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
        return fpr, tpr, roc_auc
    
    def generate_precision_recall_curve(self, save_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate and save Precision-Recall curve.
        
        Args:
            save_path: Path to save the PR curve plot
            
        Returns:
            Precision, Recall, and Average Precision score
        """
        if not self.probabilities:
            logger.warning("No probabilities available for PR curve")
            return None, None, None
        
        y_true = np.array(self.labels)
        y_prob = np.array(self.probabilities)
        
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        if save_path is None:
            save_path = self.save_dir / 'pr_curve.png'
        
        # Plot PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.phase_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {save_path}")
        return precision, recall, avg_precision
    
    def generate_classification_report(self, save_path: Optional[Path] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Real', 'Fake'],
            digits=4
        )
        
        if save_path is None:
            save_path = self.save_dir / 'classification_report.txt'
        
        with open(save_path, 'w') as f:
            f.write(f"Classification Report - {self.phase_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            
            # Add additional metrics
            f.write("\n\nAdditional Metrics:\n")
            f.write("-" * 30 + "\n")
            metrics = self.compute_metrics()
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key:25s}: {value:.4f}\n")
        
        logger.info(f"Classification report saved to {save_path}")
        return report
    
    def save_predictions(self, save_path: Optional[Path] = None):
        """
        Save detailed predictions to CSV.
        
        Args:
            save_path: Path to save predictions
        """
        if save_path is None:
            save_path = self.save_dir / 'predictions.csv'
        
        data = {
            'video_id': self.video_ids if self.video_ids else list(range(len(self.labels))),
            'true_label': self.labels,
            'predicted_label': self.predictions,
            'correct': [int(t == p) for t, p in zip(self.labels, self.predictions)]
        }
        
        if self.probabilities:
            probs = np.array(self.probabilities)
            if len(probs.shape) == 2:
                data['prob_real'] = probs[:, 0].tolist()
                data['prob_fake'] = probs[:, 1].tolist()
        
        if self.confidences:
            data['confidence'] = self.confidences
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        logger.info(f"Predictions saved to {save_path}")
    
    def generate_epoch_plot(self, metric_history: Dict[str, List[float]], save_path: Optional[Path] = None):
        """
        Generate training progress plot over epochs.
        
        Args:
            metric_history: Dictionary of metric histories
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = self.save_dir / 'training_progress.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot loss
        if 'train_loss' in metric_history and 'val_loss' in metric_history:
            ax = axes[0, 0]
            epochs = range(1, len(metric_history['train_loss']) + 1)
            ax.plot(epochs, metric_history['train_loss'], 'b-', label='Train Loss')
            ax.plot(epochs, metric_history['val_loss'], 'r-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'train_acc' in metric_history and 'val_acc' in metric_history:
            ax = axes[0, 1]
            epochs = range(1, len(metric_history['train_acc']) + 1)
            ax.plot(epochs, metric_history['train_acc'], 'b-', label='Train Acc')
            ax.plot(epochs, metric_history['val_acc'], 'r-', label='Val Acc')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'learning_rate' in metric_history:
            ax = axes[1, 0]
            epochs = range(1, len(metric_history['learning_rate']) + 1)
            ax.plot(epochs, metric_history['learning_rate'], 'g-')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
        
        # Plot per-class accuracy if available
        if 'val_acc_real' in metric_history and 'val_acc_fake' in metric_history:
            ax = axes[1, 1]
            epochs = range(1, len(metric_history['val_acc_real']) + 1)
            ax.plot(epochs, metric_history['val_acc_real'], 'b-', label='Real')
            ax.plot(epochs, metric_history['val_acc_fake'], 'r-', label='Fake')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - {self.phase_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress plot saved to {save_path}")
    
    def save_summary(self, metrics: Dict[str, Any], save_path: Optional[Path] = None):
        """
        Save evaluation summary as JSON.
        
        Args:
            metrics: Dictionary of metrics to save
            save_path: Path to save summary
        """
        if save_path is None:
            save_path = self.save_dir / 'evaluation_summary.json'
        
        summary = {
            'phase': self.phase_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'total_samples': len(self.labels),
            'total_correct': sum(1 for p, l in zip(self.predictions, self.labels) if p == l)
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Evaluation summary saved to {save_path}")
    
    def generate_full_report(self, metric_history: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate complete evaluation report with all visualizations.
        
        Args:
            metric_history: Optional training history metrics
            
        Returns:
            Dictionary of all computed metrics
        """
        logger.info(f"Generating full evaluation report for {self.phase_name}")
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Generate visualizations
        self.generate_confusion_matrix()
        self.generate_roc_curve()
        self.generate_precision_recall_curve()
        self.generate_classification_report()
        self.save_predictions()
        
        # Generate training progress plot if history available
        if metric_history:
            self.generate_epoch_plot(metric_history)
        
        # Save summary
        self.save_summary(metrics)
        
        logger.info(f"Full report generated and saved to {self.save_dir}")
        return metrics


class RLEvaluator(PhaseEvaluator):
    """
    Extended evaluator for RL training phase with additional metrics.
    """
    
    def __init__(self, phase_name: str, save_dir: Path):
        super().__init__(phase_name, save_dir)
        
        # Additional RL-specific metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_distributions = []
        self.confidence_thresholds = []
        
        # Enhanced metrics for detailed analysis
        self.confidence_trajectories = []  # Confidence evolution per episode
        self.uncertainty_trajectories = []  # Uncertainty evolution per episode
        self.action_sequences = []  # Sequence of actions taken
        self.decision_frames = []  # Frame where decision was made
        self.frames_analyzed = []  # Total frames analyzed
        self.frames_skipped = []  # Total frames skipped
        self.final_confidences = []  # Final confidence values
        self.final_uncertainties = []  # Final uncertainty values
        self.correct_predictions = []  # Whether prediction was correct
        self.true_labels = []  # Ground truth labels
        self.predicted_labels = []  # Model predictions
    
    def add_episode(
        self,
        reward: float,
        length: int,
        action_dist: Dict[str, float],
        confidence_threshold: float = None,
        # Enhanced metrics
        confidence_trajectory: List[float] = None,
        uncertainty_trajectory: List[float] = None,
        action_sequence: List[int] = None,
        decision_frame: int = None,
        frames_analyzed_count: int = None,
        frames_skipped_count: int = None,
        final_confidence: float = None,
        final_uncertainty: float = None,
        correct_prediction: bool = None,
        true_label: int = None,
        predicted_label: int = None
    ):
        """
        Add episode metrics for RL evaluation with enhanced tracking.
        
        Args:
            reward: Episode reward
            length: Episode length
            action_dist: Action distribution
            confidence_threshold: Current confidence threshold (for curriculum)
            confidence_trajectory: Confidence values throughout episode
            uncertainty_trajectory: Uncertainty values throughout episode
            action_sequence: Sequence of actions taken
            decision_frame: Frame where final decision was made
            frames_analyzed_count: Number of frames analyzed
            frames_skipped_count: Number of frames skipped
            final_confidence: Final confidence value
            final_uncertainty: Final uncertainty value
            correct_prediction: Whether prediction was correct
            true_label: Ground truth label
            predicted_label: Model's prediction
        """
        # Original metrics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.action_distributions.append(action_dist)
        
        if confidence_threshold is not None:
            self.confidence_thresholds.append(confidence_threshold)
        
        # Enhanced metrics
        if confidence_trajectory is not None:
            self.confidence_trajectories.append(confidence_trajectory)
        if uncertainty_trajectory is not None:
            self.uncertainty_trajectories.append(uncertainty_trajectory)
        if action_sequence is not None:
            self.action_sequences.append(action_sequence)
        if decision_frame is not None:
            self.decision_frames.append(decision_frame)
        if frames_analyzed_count is not None:
            self.frames_analyzed.append(frames_analyzed_count)
        if frames_skipped_count is not None:
            self.frames_skipped.append(frames_skipped_count)
        if final_confidence is not None:
            self.final_confidences.append(final_confidence)
        if final_uncertainty is not None:
            self.final_uncertainties.append(final_uncertainty)
        if correct_prediction is not None:
            self.correct_predictions.append(correct_prediction)
        if true_label is not None:
            self.true_labels.append(true_label)
        if predicted_label is not None:
            self.predicted_labels.append(predicted_label)
    
    def generate_rl_plots(self, save_dir: Optional[Path] = None):
        """
        Generate RL-specific plots.
        
        Args:
            save_dir: Directory to save plots
        """
        if save_dir is None:
            save_dir = self.save_dir
        
        # Plot episode rewards over time
        if self.episode_rewards:
            plt.figure(figsize=(10, 6))
            episodes = range(1, len(self.episode_rewards) + 1)
            
            # Plot raw rewards
            plt.subplot(2, 1, 1)
            plt.plot(episodes, self.episode_rewards, alpha=0.3, label='Raw')
            
            # Plot smoothed rewards
            window = min(100, len(self.episode_rewards) // 10)
            if window > 1:
                smoothed = pd.Series(self.episode_rewards).rolling(window).mean()
                plt.plot(episodes, smoothed, label=f'Smoothed (window={window})')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot episode lengths
            plt.subplot(2, 1, 2)
            plt.plot(episodes, self.episode_lengths, alpha=0.3, label='Raw')
            
            if window > 1:
                smoothed = pd.Series(self.episode_lengths).rolling(window).mean()
                plt.plot(episodes, smoothed, label=f'Smoothed (window={window})')
            
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('Episode Lengths')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'rl_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot action distribution evolution
        if self.action_distributions:
            from src.config import ACTION_NAMES
            
            # Extract action percentages over time
            action_history = {name: [] for name in ACTION_NAMES}
            for dist in self.action_distributions:
                for name in ACTION_NAMES:
                    action_history[name].append(dist.get(name, 0))
            
            plt.figure(figsize=(10, 6))
            episodes = range(1, len(self.action_distributions) + 1)
            
            for name in ACTION_NAMES:
                if action_history[name]:
                    plt.plot(episodes, action_history[name], label=name)
            
            plt.xlabel('Episode')
            plt.ylabel('Action Probability')
            plt.title('Action Distribution Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot confidence threshold progression (curriculum learning)
        if self.confidence_thresholds:
            plt.figure(figsize=(10, 6))
            episodes = range(1, len(self.confidence_thresholds) + 1)
            plt.plot(episodes, self.confidence_thresholds)
            plt.xlabel('Episode')
            plt.ylabel('Confidence Threshold')
            plt.title('Curriculum Learning: Confidence Threshold Progression')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'curriculum_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"RL plots saved to {save_dir}")
        
        # Generate enhanced plots if we have the data
        if self.final_confidences:
            self._plot_confidence_uncertainty_analysis(save_dir)
        if self.decision_frames:
            self._plot_decision_timing_analysis(save_dir)
        if self.action_sequences:
            self._plot_action_pattern_analysis(save_dir)
        if len(self.final_confidences) > 10:
            self._plot_correlation_analysis(save_dir)
        
        # Generate interactive dashboard
        try:
            self.generate_interactive_dashboard(save_dir)
        except Exception as e:
            logger.warning(f"Could not generate interactive dashboard: {e}")
    
    def _plot_confidence_uncertainty_analysis(self, save_dir: Path):
        """Plot detailed confidence and uncertainty analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confidence evolution
        ax = axes[0, 0]
        episodes = range(1, len(self.final_confidences) + 1)
        ax.plot(episodes, self.final_confidences, alpha=0.3, label='Raw')
        
        # Add smoothed line
        window = min(50, len(episodes) // 10)
        if window > 1:
            smoothed = pd.Series(self.final_confidences).rolling(window, center=True).mean()
            ax.plot(episodes, smoothed, label=f'Smoothed (window={window})', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Final Confidence')
        ax.set_title('Confidence Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty evolution
        if self.final_uncertainties:
            ax = axes[0, 1]
            ax.plot(episodes, self.final_uncertainties, alpha=0.3, color='orange', label='Raw')
            
            if window > 1:
                smoothed = pd.Series(self.final_uncertainties).rolling(window, center=True).mean()
                ax.plot(episodes, smoothed, color='darkorange', 
                       label=f'Smoothed (window={window})', linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Final Uncertainty')
            ax.set_title('Uncertainty Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Confidence distribution
        ax = axes[1, 0]
        ax.hist(self.final_confidences, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.final_confidences), color='red', linestyle='--',
                  label=f'Mean: {np.mean(self.final_confidences):.3f}')
        ax.set_xlabel('Final Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Confidence vs Accuracy
        if self.correct_predictions and self.final_uncertainties:
            ax = axes[1, 1]
            # Ensure all arrays have the same length
            min_len = min(len(self.correct_predictions), len(self.final_confidences), len(self.final_uncertainties))
            correct = np.array(self.correct_predictions[:min_len])
            conf = np.array(self.final_confidences[:min_len])
            unc = np.array(self.final_uncertainties[:min_len])
            
            ax.scatter(conf[~correct], unc[~correct], alpha=0.5, color='red', 
                      label='Incorrect', s=20)
            ax.scatter(conf[correct], unc[correct], alpha=0.5, color='green', 
                      label='Correct', s=20)
            
            ax.set_xlabel('Final Confidence')
            ax.set_ylabel('Final Uncertainty')
            ax.set_title('Confidence vs Uncertainty by Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confidence_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_timing_analysis(self, save_dir: Path):
        """Plot when decisions are made and their effectiveness."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Decision frame distribution
        ax = axes[0, 0]
        ax.hist(self.decision_frames, bins=range(0, 52, 2), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.decision_frames), color='red', linestyle='--',
                  label=f'Mean: {np.mean(self.decision_frames):.1f}')
        ax.set_xlabel('Decision Frame')
        ax.set_ylabel('Count')
        ax.set_title('When Decisions Are Made')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Frames analyzed over time (with averaging for readability)
        if self.frames_analyzed:
            ax = axes[0, 1]
            episodes = range(1, len(self.frames_analyzed) + 1)
            
            # Use only rolling average for large datasets
            if len(episodes) > 100:
                # For large datasets, just show the rolling average
                window = min(100, len(episodes) // 20)
                if window > 1:
                    smoothed = pd.Series(self.frames_analyzed).rolling(window, center=True, min_periods=1).mean()
                    ax.plot(episodes, smoothed, linewidth=2, color='blue', label=f'Avg (window={window})')
                else:
                    ax.plot(episodes, self.frames_analyzed, linewidth=1, color='blue')
            else:
                # For small datasets, show both raw and smoothed
                ax.plot(episodes, self.frames_analyzed, alpha=0.3, color='gray', label='Raw')
                window = min(10, len(episodes) // 5)
                if window > 1:
                    smoothed = pd.Series(self.frames_analyzed).rolling(window, center=True, min_periods=1).mean()
                    ax.plot(episodes, smoothed, linewidth=2, color='blue', label=f'Avg (window={window})')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Frames Analyzed (Avg)')
            ax.set_title('Efficiency Over Time')
            if len(episodes) <= 100 or window > 1:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Early vs Late decision accuracy
        if self.correct_predictions and self.decision_frames:
            ax = axes[1, 0]
            # Ensure arrays have same length
            min_len = min(len(self.decision_frames), len(self.correct_predictions))
            decision_frames = np.array(self.decision_frames[:min_len])
            correct = np.array(self.correct_predictions[:min_len])
            
            early_mask = decision_frames <= 15
            late_mask = decision_frames > 15
            
            early_acc = correct[early_mask].mean() * 100 if early_mask.any() else 0
            late_acc = correct[late_mask].mean() * 100 if late_mask.any() else 0
            
            bars = ax.bar(['Early (≤15)', 'Late (>15)'], [early_acc, late_acc],
                          color=['lightblue', 'lightcoral'])
            
            for bar, count in zip(bars, [early_mask.sum(), late_mask.sum()]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%\n(n={count})', ha='center', va='bottom')
            
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Decision Timing vs Accuracy')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Frames analyzed vs skipped
        if self.frames_skipped and self.frames_analyzed:
            ax = axes[1, 1]
            # Ensure arrays have same length
            min_len = min(len(self.frames_analyzed), len(self.frames_skipped))
            frames_analyzed = self.frames_analyzed[:min_len]
            frames_skipped = self.frames_skipped[:min_len]
            ax.scatter(frames_analyzed, frames_skipped, alpha=0.5, s=20)
            ax.set_xlabel('Frames Analyzed')
            ax.set_ylabel('Frames Skipped')
            ax.set_title('Analysis vs Skip Pattern')
            
            # Add diagonal line
            max_val = max(max(frames_analyzed), max(frames_skipped))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, label='Equal')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'decision_timing_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_action_pattern_analysis(self, save_dir: Path):
        """Analyze action patterns and strategies."""
        from src.config import ACTION_NAMES
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Action frequency distribution
        ax = axes[0, 0]
        all_actions = []
        for seq in self.action_sequences:
            all_actions.extend(seq)
        
        if all_actions:
            action_counts = [all_actions.count(i) for i in range(len(ACTION_NAMES))]
            ax.bar(ACTION_NAMES, action_counts)
            ax.set_xlabel('Action')
            ax.set_ylabel('Total Count')
            ax.set_title('Overall Action Usage')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Action sequence patterns
        ax = axes[0, 1]
        from collections import Counter
        
        # Find common 3-action patterns
        patterns = []
        for seq in self.action_sequences:
            for i in range(len(seq) - 2):
                patterns.append(tuple(seq[i:i+3]))
        
        if patterns:
            pattern_counts = Counter(patterns)
            top_patterns = pattern_counts.most_common(10)
            
            pattern_strs = []
            counts = []
            for pattern, count in top_patterns:
                pattern_str = '→'.join([ACTION_NAMES[a][:4] for a in pattern])
                pattern_strs.append(pattern_str)
                counts.append(count)
            
            y_pos = np.arange(len(pattern_strs))
            ax.barh(y_pos, counts)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pattern_strs, fontsize=8)
            ax.set_xlabel('Frequency')
            ax.set_title('Top Action Sequences')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Skip ratio evolution
        ax = axes[1, 0]
        skip_ratios = []
        for seq in self.action_sequences:
            skip_count = seq.count(2)  # SKIP_FORWARD
            total = len(seq)
            skip_ratios.append(skip_count / total * 100 if total > 0 else 0)
        
        if skip_ratios:
            episodes = range(1, len(skip_ratios) + 1)
            ax.plot(episodes, skip_ratios, alpha=0.3)
            
            window = min(50, len(episodes) // 10)
            if window > 1:
                smoothed = pd.Series(skip_ratios).rolling(window, center=True).mean()
                ax.plot(episodes, smoothed, linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Skip Ratio (%)')
            ax.set_title('Skip Action Usage Over Time')
            ax.grid(True, alpha=0.3)
        
        # 4. Episode length distribution
        ax = axes[1, 1]
        action_lengths = [len(seq) for seq in self.action_sequences]
        if action_lengths:
            ax.hist(action_lengths, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(action_lengths), color='red', linestyle='--',
                      label=f'Mean: {np.mean(action_lengths):.1f}')
            ax.set_xlabel('Actions per Episode')
            ax.set_ylabel('Count')
            ax.set_title('Episode Length Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'action_pattern_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, save_dir: Path):
        """Plot correlation between different metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Episode length vs Reward
        if self.episode_rewards and self.episode_lengths:
            ax = axes[0, 0]
            # Ensure arrays have same length
            min_len = min(len(self.episode_lengths), len(self.episode_rewards))
            lengths = self.episode_lengths[:min_len]
            rewards = self.episode_rewards[:min_len]
            
            ax.scatter(lengths, rewards, alpha=0.5, s=20)
            
            # Add trend line
            if len(lengths) > 1:  # Need at least 2 points for polyfit
                z = np.polyfit(lengths, rewards, 1)
                p = np.poly1d(z)
                ax.plot(lengths, p(lengths), "r--", alpha=0.5)
            
            ax.set_xlabel('Episode Length')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Length vs Reward')
            ax.grid(True, alpha=0.3)
        
        # 2. Frames analyzed vs Accuracy
        if self.frames_analyzed and self.correct_predictions:
            ax = axes[0, 1]
            from collections import defaultdict
            
            # Ensure arrays have same length
            min_len = min(len(self.frames_analyzed), len(self.correct_predictions))
            frames_analyzed = self.frames_analyzed[:min_len]
            correct_predictions = self.correct_predictions[:min_len]
            
            frame_groups = defaultdict(list)
            for f, c in zip(frames_analyzed, correct_predictions):
                frame_groups[f].append(c)
            
            x_vals = sorted(frame_groups.keys())
            y_vals = [np.mean(frame_groups[x]) * 100 for x in x_vals]
            sizes = [min(len(frame_groups[x]) * 5, 100) for x in x_vals]
            
            ax.scatter(x_vals, y_vals, s=sizes, alpha=0.6)
            ax.set_xlabel('Frames Analyzed')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Frames vs Accuracy (size=count)')
            ax.grid(True, alpha=0.3)
        
        # 3. Confidence vs Episode length
        if self.final_confidences and self.episode_lengths:
            ax = axes[1, 0]
            # Ensure arrays have same length
            min_len = min(len(self.final_confidences), len(self.episode_lengths))
            confidences = self.final_confidences[:min_len]
            lengths = self.episode_lengths[:min_len]
            
            ax.scatter(confidences, lengths, alpha=0.5, s=20)
            
            if len(confidences) > 1: 
                z = np.polyfit(confidences, lengths, 1)
                p = np.poly1d(z)
                ax.plot(confidences, p(confidences), "r--", alpha=0.5)
            
            ax.set_xlabel('Final Confidence')
            ax.set_ylabel('Episode Length')
            ax.set_title('Confidence vs Decision Speed')
            ax.grid(True, alpha=0.3)
        
        # 4. Uncertainty vs Errors
        if self.final_uncertainties and self.correct_predictions:
            ax = axes[1, 1]
            # Ensure arrays have same length
            min_len = min(len(self.final_uncertainties), len(self.correct_predictions))
            uncertainties = np.array(self.final_uncertainties[:min_len])
            errors = ~np.array(self.correct_predictions[:min_len])
            
            # Bin uncertainties and calculate error rates
            if len(uncertainties) > 0:
                bins = np.linspace(uncertainties.min(), uncertainties.max(), 15)
            else:
                bins = np.linspace(0, 1, 15)  
            bin_centers = (bins[:-1] + bins[1:]) / 2
            error_rates = []
            
            for i in range(len(bins) - 1):
                mask = (uncertainties >= bins[i]) & (uncertainties < bins[i+1])
                if mask.sum() > 0:
                    error_rates.append(errors[mask].mean() * 100)
                else:
                    error_rates.append(0)
            
            ax.bar(bin_centers, error_rates, width=(bins[1] - bins[0]) * 0.8, alpha=0.7)
            ax.set_xlabel('Uncertainty')
            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Uncertainty vs Error Rate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_interactive_dashboard(self, save_dir: Optional[Path] = None):
        """
        Generate simple interactive HTML dashboard with averaged metrics for large-scale training.
        """
        import json
        
        if save_dir is None:
            save_dir = self.save_dir
        
        # Function to create rolling averages for smoother plots
        def create_averaged_data(data, window_size=100, max_points=1000):
            """Create averaged data points for cleaner visualization."""
            if not data or len(data) == 0:
                return "[]", "[]"
            
            # Calculate rolling average
            if len(data) < window_size:
                window_size = max(1, len(data) // 4)
            
            # Use pandas rolling mean
            averaged = pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()
            
            # Downsample if needed
            if len(averaged) <= max_points:
                x = list(range(len(averaged)))
                y = averaged
            else:
                step = len(averaged) / max_points
                indices = [int(i * step) for i in range(max_points)]
                x = indices
                y = [averaged[i] for i in indices]
            
            return json.dumps(x), json.dumps(y)
        
        # Prepare averaged data for all metrics
        rewards_x, rewards_y = create_averaged_data(self.episode_rewards if self.episode_rewards else [], window_size=100)
        lengths_x, lengths_y = create_averaged_data(self.episode_lengths if self.episode_lengths else [], window_size=100)
        conf_x, conf_y = create_averaged_data(self.final_confidences if self.final_confidences else [], window_size=50)
        unc_x, unc_y = create_averaged_data(self.final_uncertainties if self.final_uncertainties else [], window_size=50)
        
        # Calculate accuracy over time
        accuracy_data = []
        if self.correct_predictions:
            window = 100
            for i in range(0, len(self.correct_predictions), window):
                batch = self.correct_predictions[i:i+window]
                if batch:
                    accuracy_data.append(np.mean(batch) * 100)
        acc_x, acc_y = create_averaged_data(accuracy_data if accuracy_data else [], window_size=10)
        
        # Calculate summary stats
        total_episodes = len(self.episode_rewards) if self.episode_rewards else 0
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards and len(self.episode_rewards) > 0 else 0
        avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths and len(self.episode_lengths) > 0 else 0
        accuracy = np.mean(self.correct_predictions[-100:]) * 100 if self.correct_predictions and len(self.correct_predictions) > 0 else 0
        final_conf = np.mean(self.final_confidences[-100:]) if self.final_confidences and len(self.final_confidences) > 0 else 0
        
        # Create simple HTML with better organized plots
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>EAGER Training Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ text-align: center; color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
                .stat {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 28px; font-weight: bold; color: #4A90E2; }}
                .stat-label {{ color: #666; font-size: 12px; margin-top: 5px; text-transform: uppercase; }}
                .plot-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .plot {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                @media (max-width: 1200px) {{ .plot-row {{ grid-template-columns: 1fr; }} }}
            </style>
        </head>
        <body>
            <h1>🤖 EAGER Training Dashboard</h1>
            <p style="text-align: center; color: #888;">Rolling Averages for Clean Visualization</p>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{total_episodes:,}</div>
                    <div class="stat-label">Total Episodes</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{avg_reward:.2f}</div>
                    <div class="stat-label">Avg Reward</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{accuracy:.1f}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{avg_length:.1f}</div>
                    <div class="stat-label">Avg Steps</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{final_conf:.3f}</div>
                    <div class="stat-label">Confidence</div>
                </div>
            </div>
            
            <h2>Performance Metrics</h2>
            <div class="plot-row">
                <div id="rewards-plot" class="plot"></div>
                <div id="accuracy-plot" class="plot"></div>
            </div>
            
            <h2>Decision Making</h2>
            <div class="plot-row">
                <div id="confidence-plot" class="plot"></div>
                <div id="lengths-plot" class="plot"></div>
            </div>
            
            <script>
                // Common layout settings
                const layout_template = {{
                    font: {{ family: 'Arial, sans-serif' }},
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white',
                    margin: {{ t: 40, r: 20, b: 40, l: 50 }},
                    height: 350,
                    showlegend: false,
                    hovermode: 'x unified'
                }};
                
                // 1. Average Reward Plot (100-episode rolling average)
                Plotly.newPlot('rewards-plot', [{{
                    x: {rewards_x},
                    y: {rewards_y},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: '#4A90E2', width: 2}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(74, 144, 226, 0.1)'
                }}], {{
                    ...layout_template,
                    title: {{text: 'Average Episode Reward', font: {{size: 16}}}},
                    xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                    yaxis: {{title: 'Reward (100-ep avg)', gridcolor: '#e0e0e0'}}
                }});
                
                // 2. Accuracy Plot (Rolling accuracy)
                Plotly.newPlot('accuracy-plot', [{{
                    x: {acc_x},
                    y: {acc_y},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: '#52C41A', width: 2}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(82, 196, 26, 0.1)'
                }}], {{
                    ...layout_template,
                    title: {{text: 'Classification Accuracy', font: {{size: 16}}}},
                    xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                    yaxis: {{title: 'Accuracy %', gridcolor: '#e0e0e0', range: [0, 100]}}
                }});
                
                // 3. Confidence Plot (50-episode rolling average)
                Plotly.newPlot('confidence-plot', [{{
                    x: {conf_x},
                    y: {conf_y},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: '#722ED1', width: 2}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(114, 46, 209, 0.1)'
                }}], {{
                    ...layout_template,
                    title: {{text: 'Decision Confidence', font: {{size: 16}}}},
                    xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                    yaxis: {{title: 'Confidence (50-ep avg)', gridcolor: '#e0e0e0', range: [0, 1]}}
                }});
                
                // 4. Episode Length Plot (100-episode rolling average)
                Plotly.newPlot('lengths-plot', [{{
                    x: {lengths_x},
                    y: {lengths_y},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: '#FA8C16', width: 2}},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(250, 140, 22, 0.1)'
                }}], {{
                    ...layout_template,
                    title: {{text: 'Decision Speed', font: {{size: 16}}}},
                    xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                    yaxis: {{title: 'Steps to Decision (100-ep avg)', gridcolor: '#e0e0e0'}}
                }});
            </script>
        </body>
        </html>
        '''
        
        # Save the dashboard
        dashboard_path = save_dir / 'interactive_dashboard.html'
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        # Also save raw data for later
        import pickle
        data_path = save_dir / 'dashboard_data.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump({
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'final_confidences': self.final_confidences,
                'final_uncertainties': self.final_uncertainties,
                'correct_predictions': self.correct_predictions
            }, f)



if __name__ == "__main__":
    # Test evaluation metrics
    import torch
    
    # Create test evaluator
    evaluator = PhaseEvaluator("test_phase", Path("test_logs"))
    
    # Add some dummy data
    for _ in range(10):
        batch_size = 32
        predictions = torch.randint(0, 2, (batch_size,))
        labels = torch.randint(0, 2, (batch_size,))
        probabilities = torch.softmax(torch.randn(batch_size, 2), dim=1)
        confidences = torch.rand(batch_size)
        
        evaluator.add_batch(predictions, labels, probabilities, confidences=confidences)
    
    # Generate report
    metrics = evaluator.generate_full_report()
    print("Metrics:", metrics)
    print("Evaluation test completed!")