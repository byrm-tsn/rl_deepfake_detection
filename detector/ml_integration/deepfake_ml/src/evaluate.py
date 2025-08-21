"""
Comprehensive evaluation framework for EAGER algorithm.
Assesses performance across multiple dimensions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from collections import defaultdict

from src.config import *
from data_loader import ProcessedVideoDataset, create_data_loaders
from deepfake_env import DeepfakeEnv

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    video_id: str
    true_label: int
    predicted_label: Optional[int]
    confidence: float
    correct: bool
    episode_length: int
    frames_analyzed: int
    action_history: List[int]
    confidence_history: List[float]
    reward_total: float
    termination_reason: str
    
    # Action counts
    num_next: int = 0
    num_focus: int = 0
    num_augment: int = 0
    
    def __post_init__(self):
        """Calculate action counts."""
        for action in self.action_history:
            if action == 0:
                self.num_next += 1
            elif action == 1:
                self.num_focus += 1
            elif action == 2:
                self.num_augment += 1


class EagerEvaluator:
    """
    Comprehensive evaluator for EAGER agent.
    """
    
    def __init__(
        self,
        model,
        test_env: DeepfakeEnv,
        explanation_head: Optional[Any] = None,  
        saliency_head: Optional[Any] = None,  
        device: str = DEVICE
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PPO model
            test_env: Test environment
            explanation_head: Optional explanation generator
            saliency_head: Optional saliency generator
            device: Computing device
        """
        self.model = model
        self.test_env = test_env
        self.explanation_head = explanation_head
        self.saliency_head = saliency_head
        self.device = device
        
        self.episode_metrics = []
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        
        logger.info("Initialized EAGER Evaluator")
    
    def evaluate_episode(
        self,
        video_idx: Optional[int] = None,
        deterministic: bool = True,
        render: bool = False
    ) -> EpisodeMetrics:
        """
        Evaluate single episode.
        
        Args:
            video_idx: Specific video index to evaluate
            deterministic: Use deterministic policy
            render: Render episode
            
        Returns:
            Episode metrics
        """
        # Reset environment
        if video_idx is not None:
            video_id = self.test_env.dataset.video_ids[video_idx]
            obs, info = self.test_env.reset(options={'video_id': video_id})
        else:
            obs, info = self.test_env.reset()
        
        # Track metrics
        action_history = []
        confidence_history = []
        total_reward = 0.0
        
        done = False
        steps = 0
        
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            action = int(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated
            
            # Track metrics
            action_history.append(action)
            confidence_history.append(info['current_confidence'])
            total_reward += reward
            steps += 1
            
            if render:
                self.test_env.render()
        
        # Determine predicted label
        if info['termination_reason'] == 'stop_fake':
            predicted_label = 1
        elif info['termination_reason'] == 'stop_real':
            predicted_label = 0
        else:
            # No decision made - use confidence threshold
            if confidence_history[-1] > 0.5:
                predicted_label = 1
            else:
                predicted_label = 0
        
        # Create metrics
        metrics = EpisodeMetrics(
            video_id=info['video_id'],
            true_label=info['true_label'],
            predicted_label=predicted_label,
            confidence=confidence_history[-1] if confidence_history else 0.0,
            correct=(predicted_label == info['true_label']),
            episode_length=steps,
            frames_analyzed=info['frames_analyzed'],
            action_history=action_history,
            confidence_history=confidence_history,
            reward_total=total_reward,
            termination_reason=info['termination_reason']
        )
        
        return metrics
    
    def evaluate_dataset(
        self,
        num_episodes: Optional[int] = None,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on entire test dataset.
        
        Args:
            num_episodes: Number of episodes to evaluate (None for all)
            deterministic: Use deterministic policy
            
        Returns:
            Evaluation results dictionary
        """
        if num_episodes is None:
            num_episodes = len(self.test_env.dataset)
        
        logger.info(f"Evaluating {num_episodes} episodes...")
        
        # Reset metrics
        self.episode_metrics = []
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        
        # Evaluate episodes
        for i in tqdm(range(num_episodes), desc="Evaluating"):
            metrics = self.evaluate_episode(
                video_idx=i if i < len(self.test_env.dataset) else None,
                deterministic=deterministic
            )
            
            self.episode_metrics.append(metrics)
            self.predictions.append(metrics.predicted_label)
            self.ground_truths.append(metrics.true_label)
            self.confidences.append(metrics.confidence)
        
        # Calculate aggregate metrics
        results = self.calculate_metrics()
        
        return results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.predictions)
        ground_truths = np.array(self.ground_truths)
        confidences = np.array(self.confidences)
        
        # Classification metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions)
        recall = recall_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions)
        
        # ROC and PR curves
        if len(np.unique(ground_truths)) > 1:
            auc_roc = roc_auc_score(ground_truths, confidences)
            ap = average_precision_score(ground_truths, confidences)
        else:
            auc_roc = 0.0
            ap = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        
        # Efficiency metrics
        avg_episode_length = np.mean([m.episode_length for m in self.episode_metrics])
        avg_frames_analyzed = np.mean([m.frames_analyzed for m in self.episode_metrics])
        
        # Action distribution
        total_next = sum([m.num_next for m in self.episode_metrics])
        total_focus = sum([m.num_focus for m in self.episode_metrics])
        total_augment = sum([m.num_augment for m in self.episode_metrics])
        total_actions = total_next + total_focus + total_augment
        
        action_distribution = {
            'NEXT': total_next / total_actions if total_actions > 0 else 0,
            'FOCUS': total_focus / total_actions if total_actions > 0 else 0,
            'AUGMENT': total_augment / total_actions if total_actions > 0 else 0
        }
        
        # Confidence calibration
        confidence_bins = np.linspace(0, 1, 11)
        calibration_data = self.calculate_calibration(
            confidences, predictions, ground_truths, confidence_bins
        )
        
        # Strategic behavior analysis
        strategic_metrics = self.analyze_strategic_behavior()
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'average_precision': ap,
            'confusion_matrix': cm.tolist(),
            'avg_episode_length': avg_episode_length,
            'avg_frames_analyzed': avg_frames_analyzed,
            'action_distribution': action_distribution,
            'calibration_data': calibration_data,
            'strategic_metrics': strategic_metrics,
            'total_episodes': len(self.episode_metrics)
        }
        
        return results
    
    def calculate_calibration(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        bins: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Calculate confidence calibration metrics.
        
        Args:
            confidences: Confidence scores
            predictions: Predictions
            ground_truths: Ground truth labels
            bins: Confidence bins
            
        Returns:
            Calibration data
        """
        calibration_data = {
            'bins': bins.tolist(),
            'accuracy_per_bin': [],
            'avg_confidence_per_bin': [],
            'count_per_bin': []
        }
        
        for i in range(len(bins) - 1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            bin_predictions = predictions[bin_mask]
            bin_truths = ground_truths[bin_mask]
            bin_confidences = confidences[bin_mask]
            
            if len(bin_predictions) > 0:
                bin_accuracy = accuracy_score(bin_truths, bin_predictions)
                avg_confidence = np.mean(bin_confidences)
            else:
                bin_accuracy = 0.0
                avg_confidence = (bins[i] + bins[i+1]) / 2
            
            calibration_data['accuracy_per_bin'].append(bin_accuracy)
            calibration_data['avg_confidence_per_bin'].append(avg_confidence)
            calibration_data['count_per_bin'].append(len(bin_predictions))
        
        # Calculate ECE (Expected Calibration Error)
        ece = 0.0
        total_samples = len(predictions)
        for i in range(len(bins) - 1):
            if calibration_data['count_per_bin'][i] > 0:
                weight = calibration_data['count_per_bin'][i] / total_samples
                error = abs(calibration_data['accuracy_per_bin'][i] - 
                          calibration_data['avg_confidence_per_bin'][i])
                ece += weight * error
        
        calibration_data['ece'] = ece
        
        return calibration_data
    
    def analyze_strategic_behavior(self) -> Dict[str, Any]:
        """
        Analyze agent's strategic decision-making patterns.
        
        Returns:
            Strategic behavior metrics
        """
        strategic_metrics = {}
        
        # Analyze decision timing
        decision_frames = []
        early_decisions = 0
        late_decisions = 0
        
        for metrics in self.episode_metrics:
            if metrics.termination_reason in ['stop_real', 'stop_fake']:
                decision_frames.append(metrics.frames_analyzed)
                if metrics.frames_analyzed < 10:
                    early_decisions += 1
                elif metrics.frames_analyzed > 30:
                    late_decisions += 1
        
        if decision_frames:
            strategic_metrics['avg_decision_frame'] = np.mean(decision_frames)
            strategic_metrics['early_decision_rate'] = early_decisions / len(decision_frames)
            strategic_metrics['late_decision_rate'] = late_decisions / len(decision_frames)
        
        # Analyze uncertainty-driven behavior
        high_confidence_correct = 0
        high_confidence_total = 0
        
        for metrics in self.episode_metrics:
            if metrics.confidence > CONFIDENCE_THRESHOLD:
                high_confidence_total += 1
                if metrics.correct:
                    high_confidence_correct += 1
        
        if high_confidence_total > 0:
            strategic_metrics['high_confidence_accuracy'] = (
                high_confidence_correct / high_confidence_total
            )
        
        # Analyze action patterns for correct vs incorrect
        correct_action_dist = defaultdict(int)
        incorrect_action_dist = defaultdict(int)
        
        for metrics in self.episode_metrics:
            action_dist = correct_action_dist if metrics.correct else incorrect_action_dist
            for action in metrics.action_history:
                action_dist[action] += 1
        
        strategic_metrics['correct_action_pattern'] = dict(correct_action_dist)
        strategic_metrics['incorrect_action_pattern'] = dict(incorrect_action_dist)
        
        return strategic_metrics
    
    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("="*60)
        report.append("EAGER EVALUATION REPORT")
        report.append("="*60)
        
        # Classification Performance
        report.append("\n## Classification Performance")
        report.append(f"Accuracy: {results['accuracy']:.4f}")
        report.append(f"Precision: {results['precision']:.4f}")
        report.append(f"Recall: {results['recall']:.4f}")
        report.append(f"F1 Score: {results['f1_score']:.4f}")
        report.append(f"AUC-ROC: {results['auc_roc']:.4f}")
        report.append(f"Average Precision: {results['average_precision']:.4f}")
        
        # Efficiency Metrics
        report.append("\n## Efficiency Metrics")
        report.append(f"Average Episode Length: {results['avg_episode_length']:.2f} steps")
        report.append(f"Average Frames Analyzed: {results['avg_frames_analyzed']:.2f}")
        
        # Action Distribution
        report.append("\n## Action Distribution")
        for action, prob in results['action_distribution'].items():
            report.append(f"{action}: {prob:.3f}")
        
        # Confidence Calibration
        report.append("\n## Confidence Calibration")
        report.append(f"Expected Calibration Error (ECE): {results['calibration_data']['ece']:.4f}")
        
        # Strategic Behavior
        report.append("\n## Strategic Behavior")
        strategic = results['strategic_metrics']
        if 'avg_decision_frame' in strategic:
            report.append(f"Average Decision Frame: {strategic['avg_decision_frame']:.2f}")
            report.append(f"Early Decision Rate: {strategic['early_decision_rate']:.3f}")
            report.append(f"Late Decision Rate: {strategic['late_decision_rate']:.3f}")
        if 'high_confidence_accuracy' in strategic:
            report.append(f"High Confidence Accuracy: {strategic['high_confidence_accuracy']:.4f}")
        
        # Confusion Matrix
        report.append("\n## Confusion Matrix")
        report.append("        Pred Real  Pred Fake")
        cm = results['confusion_matrix']
        report.append(f"Real    {cm[0][0]:9d}  {cm[0][1]:9d}")
        report.append(f"Fake    {cm[1][0]:9d}  {cm[1][1]:9d}")
        
        report.append("\n" + "="*60)
        
        report_str = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {save_path}")
        
        return report_str
    
    def plot_results(
        self,
        results: Dict[str, Any],
        save_dir: Optional[Path] = None
    ):
        """
        Generate visualization plots.
        
        Args:
            results: Evaluation results
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        if save_dir:
            plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Action Distribution Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        actions = list(results['action_distribution'].keys())
        probs = list(results['action_distribution'].values())
        ax.bar(actions, probs, color=['blue', 'green', 'orange'])
        ax.set_title('Action Distribution')
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, 1])
        for i, v in enumerate(probs):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        if save_dir:
            plt.savefig(save_dir / 'action_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Confidence Calibration Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        cal_data = results['calibration_data']
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(
            cal_data['avg_confidence_per_bin'],
            cal_data['accuracy_per_bin'],
            'o-',
            label=f'Model (ECE={cal_data["ece"]:.3f})'
        )
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Confidence Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_dir:
            plt.savefig(save_dir / 'calibration_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Episode Length Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        episode_lengths = [m.episode_length for m in self.episode_metrics]
        ax.hist(episode_lengths, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(episode_lengths):.2f}')
        ax.set_xlabel('Episode Length (steps)')
        ax.set_ylabel('Count')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        if save_dir:
            plt.savefig(save_dir / 'episode_lengths.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    logger.info("Evaluation framework loaded successfully")