"""
Utility functions and debugging tools for EAGER algorithm.
"""

import torch
import numpy as np
import random
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from collections import deque
import psutil
import GPUtil

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed}")


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        System information dictionary
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    # GPU information
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        # GPU memory
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            info['gpu_memory_used'] = gpu.memoryUsed
            info['gpu_memory_total'] = gpu.memoryTotal
            info['gpu_utilization'] = gpu.load * 100
    else:
        info['cuda_available'] = False
    
    return info


class EpisodeDebugger:
    """
    Debug tool for analyzing episode trajectories.
    """
    
    def __init__(self, log_dir: Path = None):
        """
        Initialize episode debugger.
        
        Args:
            log_dir: Directory to save debug logs
        """
        self.log_dir = log_dir or Path("debug_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = []
    
    def log_episode(
        self,
        episode_id: str,
        trajectory: Dict[str, Any]
    ):
        """
        Log episode trajectory for debugging.
        
        Args:
            episode_id: Episode identifier
            trajectory: Episode trajectory data
        """
        # Save trajectory
        episode_path = self.log_dir / f"episode_{episode_id}.pkl"
        with open(episode_path, 'wb') as f:
            pickle.dump(trajectory, f)
        
        self.episodes.append(episode_id)
        
        # Save summary
        summary = {
            'episode_id': episode_id,
            'video_id': trajectory.get('video_id', 'unknown'),
            'true_label': trajectory.get('true_label', -1),
            'prediction': trajectory.get('prediction', -1),
            'correct': trajectory.get('correct', False),
            'episode_length': len(trajectory.get('actions', [])),
            'total_reward': sum(trajectory.get('rewards', [])),
            'final_confidence': trajectory.get('final_confidence', 0.0)
        }
        
        summary_path = self.log_dir / f"summary_{episode_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        Load episode trajectory from logs.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Episode trajectory
        """
        episode_path = self.log_dir / f"episode_{episode_id}.pkl"
        with open(episode_path, 'rb') as f:
            trajectory = pickle.load(f)
        return trajectory
    
    def visualize_trajectory(
        self,
        episode_id: str,
        save_path: Optional[Path] = None
    ):
        """
        Visualize episode trajectory.
        
        Args:
            episode_id: Episode identifier
            save_path: Path to save visualization
        """
        trajectory = self.load_episode(episode_id)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Action sequence
        actions = trajectory.get('actions', [])
        action_names = ['NEXT', 'FOCUS', 'AUGMENT', 'STOP_REAL', 'STOP_FAKE']
        ax = axes[0]
        ax.plot(actions, 'o-')
        ax.set_ylabel('Action')
        ax.set_title(f'Action Sequence - Episode {episode_id}')
        ax.set_yticks(range(5))
        ax.set_yticklabels(action_names)
        ax.grid(True, alpha=0.3)
        
        # Confidence progression
        confidences = trajectory.get('confidence_history', [])
        ax = axes[1]
        ax.plot(confidences, 'g-', linewidth=2)
        ax.axhline(y=0.85, color='r', linestyle='--', label='Threshold')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Progression')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reward progression
        rewards = trajectory.get('rewards', [])
        cumulative_rewards = np.cumsum(rewards)
        ax = axes[2]
        ax.plot(cumulative_rewards, 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_failures(self) -> List[Dict[str, Any]]:
        """
        Analyze failed episodes.
        
        Returns:
            List of failure analysis results
        """
        failures = []
        
        for episode_id in self.episodes:
            summary_path = self.log_dir / f"summary_{episode_id}.json"
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            if not summary['correct']:
                trajectory = self.load_episode(episode_id)
                
                # Analyze failure pattern
                analysis = {
                    'episode_id': episode_id,
                    'video_id': summary['video_id'],
                    'true_label': summary['true_label'],
                    'prediction': summary['prediction'],
                    'confidence': summary['final_confidence'],
                    'episode_length': summary['episode_length'],
                    'action_counts': self._count_actions(trajectory['actions'])
                }
                
                # Check for specific failure patterns
                if summary['final_confidence'] > 0.85:
                    analysis['failure_type'] = 'high_confidence_error'
                elif summary['episode_length'] < 5:
                    analysis['failure_type'] = 'premature_decision'
                elif summary['episode_length'] > 40:
                    analysis['failure_type'] = 'over_analysis'
                else:
                    analysis['failure_type'] = 'uncertain'
                
                failures.append(analysis)
        
        return failures
    
    def _count_actions(self, actions: List[int]) -> Dict[str, int]:
        """Count occurrences of each action."""
        counts = {
            'NEXT': 0,
            'FOCUS': 0,
            'AUGMENT': 0,
            'STOP_REAL': 0,
            'STOP_FAKE': 0
        }
        action_names = list(counts.keys())
        
        for action in actions:
            if 0 <= action < len(action_names):
                counts[action_names[action]] += 1
        
        return counts


class MemoryTracker:
    """
    Track memory usage during training.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize memory tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.cpu_memory = deque(maxlen=window_size)
        self.gpu_memory = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        """Update memory statistics."""
        # CPU memory
        cpu_mem = psutil.virtual_memory().used / (1024**3)  # GB
        self.cpu_memory.append(cpu_mem)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.gpu_memory.append(gpu_mem)
        else:
            self.gpu_memory.append(0)
        
        self.timestamps.append(datetime.now())
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get memory statistics.
        
        Returns:
            Memory statistics dictionary
        """
        stats = {}
        
        if self.cpu_memory:
            stats['cpu_memory_mean'] = np.mean(self.cpu_memory)
            stats['cpu_memory_max'] = np.max(self.cpu_memory)
            stats['cpu_memory_current'] = self.cpu_memory[-1]
        
        if self.gpu_memory and torch.cuda.is_available():
            stats['gpu_memory_mean'] = np.mean(self.gpu_memory)
            stats['gpu_memory_max'] = np.max(self.gpu_memory)
            stats['gpu_memory_current'] = self.gpu_memory[-1]
        
        return stats
    
    def plot_memory_usage(self, save_path: Optional[Path] = None):
        """
        Plot memory usage over time.
        
        Args:
            save_path: Path to save plot
        """
        if not self.timestamps:
            logger.warning("No memory data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # CPU memory
        ax = axes[0]
        ax.plot(list(self.cpu_memory), 'b-', label='CPU Memory')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('CPU Memory Usage')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # GPU memory
        if torch.cuda.is_available():
            ax = axes[1]
            ax.plot(list(self.gpu_memory), 'r-', label='GPU Memory')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Memory (GB)')
            ax.set_title('GPU Memory Usage')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10
):
    """
    Create video from frame sequence.
    
    Args:
        frames: List of frame arrays
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        logger.warning("No frames to create video")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Video saved to {output_path}")


def load_checkpoint(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint


def save_metrics_plot(
    metrics: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training Metrics"
):
    """
    Save training metrics plot.
    
    Args:
        metrics: Dictionary of metric lists
        save_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, values in metrics.items():
        ax.plot(values, label=name)
    
    ax.set_xlabel('Epoch/Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Metrics plot saved to {save_path}")


class ExperimentTracker:
    """
    Track and manage experiments.
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory for experiment logs
        """
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = self.experiment_dir / self.experiment_id
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        
        logger.info(f"Created experiment {self.experiment_id}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_path = self.experiment_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a training step."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
    
    def save_model(self, model_state: Dict[str, Any], name: str):
        """Save model checkpoint."""
        model_path = self.experiment_path / f"{name}.pth"
        torch.save(model_state, model_path)
        logger.info(f"Saved model to {model_path}")
    
    def finalize(self):
        """Finalize experiment and save all metrics."""
        # Save metrics
        metrics_path = self.experiment_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Generate plots
        for metric_name, values in self.metrics.items():
            if values:
                steps, vals = zip(*values)
                plt.figure(figsize=(8, 6))
                plt.plot(steps, vals)
                plt.xlabel('Step')
                plt.ylabel(metric_name)
                plt.title(metric_name)
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    self.experiment_path / f"{metric_name}.png",
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()
        
        logger.info(f"Experiment {self.experiment_id} finalized")


def dimension_validator(
    expected_shape: tuple,
    tensor: torch.Tensor,
    name: str,
    raise_on_mismatch: bool = True
) -> bool:
    """
    Validate tensor dimensions and print debug information.
    As per PDF Step 2.4: Assert tensor.shape == expected_shape and print tensor.mean()/std() for debug.
    
    Args:
        expected_shape: Expected shape of the tensor
        tensor: Tensor to validate
        name: Name of the tensor for logging
        raise_on_mismatch: Whether to raise assertion error on mismatch
        
    Returns:
        True if dimensions match, False otherwise
        
    Raises:
        AssertionError: If dimensions don't match and raise_on_mismatch=True
    """
    actual_shape = tuple(tensor.shape)
    expected_shape = tuple(expected_shape)
    
    if actual_shape == expected_shape:
        # Dimensions match - print debug statistics as per PDF
        logger.debug(f"✓ {name} dimension check passed: {actual_shape}")
        
        # Print mean and std for debugging
        if tensor.numel() > 0:  # Check tensor is not empty
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            logger.debug(f"  {name} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        return True
    else:
        # Dimension mismatch
        error_msg = (f"✗ {name} dimension mismatch! Expected: {expected_shape}, Got: {actual_shape}")
        
        if raise_on_mismatch:
            assert False, error_msg
        else:
            logger.error(error_msg)
            return False


def validate_state_concatenation(
    frame_features: torch.Tensor,
    temporal_memory: torch.Tensor,
    expected_concat_dim: int
) -> torch.Tensor:
    """
    Validate and perform state concatenation as per PDF Step 2.5.
    Ensures frame features and temporal memory can be concatenated properly.
    
    Args:
        frame_features: Current frame features from ViT
        temporal_memory: LSTM hidden state
        expected_concat_dim: Expected dimension after concatenation
        
    Returns:
        Concatenated state tensor
    """
    from src.config import VISION_EMBEDDING_DIM, LSTM_EFFECTIVE_DIM
    
    # Validate individual dimensions
    dimension_validator(
        (frame_features.shape[0], VISION_EMBEDDING_DIM) if frame_features.dim() == 2 else (VISION_EMBEDDING_DIM,),
        frame_features,
        "frame_features"
    )
    
    dimension_validator(
        (temporal_memory.shape[0], LSTM_EFFECTIVE_DIM) if temporal_memory.dim() == 2 else (LSTM_EFFECTIVE_DIM,),
        temporal_memory,
        "temporal_memory"
    )
    
    # Perform concatenation
    state = torch.cat([frame_features, temporal_memory], dim=-1)
    
    # Validate concatenated dimension
    dimension_validator(
        (state.shape[0], expected_concat_dim) if state.dim() == 2 else (expected_concat_dim,),
        state,
        "concatenated_state"
    )
    
    return state


if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utility functions...")
    
    # Test system info
    info = get_system_info()
    print("System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test memory tracker
    tracker = MemoryTracker()
    for _ in range(10):
        tracker.update()
    
    stats = tracker.get_stats()
    print("\nMemory Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f} GB")
    
    # Test dimension validation (Step 2.4 from PDF)
    print("\nTesting dimension validation...")
    test_tensor = torch.randn(2, 768)
    dimension_validator((2, 768), test_tensor, "test_tensor")
    
    # Test state concatenation validation (Step 2.5 from PDF)
    from src.config import VISION_EMBEDDING_DIM, LSTM_EFFECTIVE_DIM
    frame_feat = torch.randn(2, VISION_EMBEDDING_DIM)
    temp_mem = torch.randn(2, LSTM_EFFECTIVE_DIM)
    concat_state = validate_state_concatenation(frame_feat, temp_mem, VISION_EMBEDDING_DIM + LSTM_EFFECTIVE_DIM)
    print(f"Concatenated state shape: {concat_state.shape}")
    
    print("\nUtility tests completed successfully!")