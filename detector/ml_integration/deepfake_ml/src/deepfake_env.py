"""
Custom Gymnasium environment for EAGER deepfake detection.
Simulates sequential video analysis with strategic decision-making.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
# For automatic mixed precision optimization
from torch.amp import autocast 
from typing import Dict, Tuple, Optional, Any, List
import logging
import random
from dataclasses import dataclass, field
import cv2
import time 


from src.config import (
    FRAMES_PER_VIDEO, VISION_EMBEDDING_DIM, LSTM_EFFECTIVE_DIM,
    MAX_EPISODE_STEPS, MIN_FRAMES_BEFORE_DECISION,
    CONFIDENCE_THRESHOLD, DEVICE,
    AUGMENT_BRIGHTNESS_RANGE, AUGMENT_JPEG_QUALITY_MIN,
    AUGMENT_JPEG_QUALITY_MAX, AUGMENT_ROTATION_RANGE,
    USE_BAYESIAN_UNCERTAINTY, MC_SAMPLES, USE_MIXED_PRECISION
)
from src.feature_extractor import VisionBackbone, TemporalMemory, ClassifierHead
from src.reward_system import RewardCalculator
from src.data_loader import ProcessedVideoDataset

logger = logging.getLogger(__name__)


@dataclass
class EpisodeInfo:
    """Track episode information for debugging and analysis."""
    video_id: str = ""
    true_label: int = 0
    action_history: List[int] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    frames_analyzed: int = 0
    episode_steps: int = 0
    final_prediction: Optional[int] = None
    final_confidence: float = 0.0
    termination_reason: str = ""


class DeepfakeEnv(gym.Env):
    """
    Custom RL environment for sequential deepfake detection.
    Agent learns to analyze video frames strategically.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        dataset: ProcessedVideoDataset,
        vision_backbone: VisionBackbone,
        temporal_memory: TemporalMemory,
        classifier_head: ClassifierHead,
        reward_calculator: RewardCalculator,
        device: str = DEVICE,
        training: bool = True,
        verbose: bool = False
    ):
        """
        Initialize deepfake detection environment.
        
        Args:
            dataset: Video dataset
            vision_backbone: Frozen vision feature extractor
            temporal_memory: LSTM network for temporal modeling
            classifier_head: Classification head for confidence estimation
            reward_calculator: Reward calculation system
            device: Computing device
            training: Whether in training mode
            verbose: Enable detailed logging
        """
        super().__init__()
        
        self.dataset = dataset
        self.vision_backbone = vision_backbone
        self.temporal_memory = temporal_memory
        self.classifier_head = classifier_head
        self.reward_calculator = reward_calculator
        self.device = device
        self.training = training
        self.verbose = verbose
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        # 0: NEXT, 1: FOCUS, 2: AUGMENT, 3: STOP_REAL, 4: STOP_FAKE
        self.action_names = ["NEXT", "FOCUS", "AUGMENT", "STOP_REAL", "STOP_FAKE"]
        
        # Observation space: Dictionary of components
        self.observation_space = spaces.Dict({
            'frame_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(VISION_EMBEDDING_DIM,), dtype=np.float32
            ),
            'temporal_memory': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(LSTM_EFFECTIVE_DIM,), dtype=np.float32
            ),
            'frame_position': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
            'uncertainty': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            )
        })
        
        # Episode state variables
        self._reset_episode_state()
        
        logger.info(f"Initialized DeepfakeEnv (training={training})")
    
    def _reset_episode_state(self):
        """Reset all episode state variables."""
        self.current_video_id = None
        self.current_video_frames = None
        self.current_frame_features = None
        self.true_label = None
        self.current_frame_idx = 0
        self.lstm_hidden = None
        self.lstm_cell = None
        self.episode_info = EpisodeInfo()
        self.episode_done = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options (e.g., specific video_id)
            
        Returns:
            observation: Initial observation
            info: Episode information
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self._reset_episode_state()
        
        # Select video (specific or random)
        if options and 'video_id' in options:
            video_idx = self.dataset.video_ids.index(options['video_id'])
        else:
            video_idx = random.randint(0, len(self.dataset) - 1)
        
        # Load video data
        frames_tensor, label, video_id = self.dataset[video_idx]
        frames_tensor = frames_tensor.to(self.device)
        
        self.current_video_id = video_id
        self.current_video_frames = frames_tensor
        self.true_label = label
        
        # Extract all frame features upfront (frozen backbone)
        with torch.no_grad():
            self.current_frame_features = self.vision_backbone(frames_tensor)
        
        # Initialize LSTM states
        self.lstm_hidden, self.lstm_cell = self.temporal_memory.init_hidden(
            batch_size=1, device=self.device
        )
        
        # Initialize episode info
        self.episode_info = EpisodeInfo(
            video_id=video_id,
            true_label=label
        )
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        if self.verbose:
            logger.info(f"Reset environment with video {video_id} (label={label})")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute action in environment.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        start_time = time.time() # PROFILING START

        # Apply confidence gating
        original_action = action
        action, gate_penalty = self.reward_calculator.apply_confidence_gate(
            action=action,
            confidence=self.episode_info.final_confidence,
            frames_analyzed=self.episode_info.frames_analyzed
        )
        
        if action != original_action and self.verbose:
            logger.info(f"Confidence gate: {self.action_names[original_action]} -> {self.action_names[action]}")
        
        # Record action
        self.episode_info.action_history.append(action)
        self.episode_info.episode_steps += 1
        
        # Get uncertainty before action
        # Profile MCDO calculations for performance monitoring
        uncertainty_start_time = time.time() # PROFILING
        uncertainty_before = self._get_current_uncertainty()
        uncertainty_time = time.time() - uncertainty_start_time 
        
        # Execute action
        action_start_time = time.time()
        reward = gate_penalty 
        terminated = False
        
        if action == 0:  # NEXT
            reward += self._execute_next_action()
        elif action == 1:  # FOCUS
            reward += self._execute_focus_action(uncertainty_before)
        elif action == 2:  # AUGMENT
            reward += self._execute_augment_action(uncertainty_before)
        elif action == 3:  # STOP_REAL
            reward += self._execute_stop_action(prediction=0)
            terminated = True
        elif action == 4:  # STOP_FAKE
            reward += self._execute_stop_action(prediction=1)
            terminated = True
        
        action_time = time.time() - action_start_time 

        
        # Check truncation conditions
        truncated = False
        if not terminated:
            if self.current_frame_idx >= FRAMES_PER_VIDEO:
                # Reached end of video without decision
                reward_penalty, _ = self.reward_calculator.calculate_no_decision_penalty(
                    episode_length=self.episode_info.episode_steps,
                    frames_analyzed=self.episode_info.frames_analyzed,
                    final_confidence=self.episode_info.final_confidence
                )
                reward += reward_penalty
                terminated = True
                self.episode_info.termination_reason = "end_of_video"
            elif self.episode_info.episode_steps >= MAX_EPISODE_STEPS:
                # Max steps reached
                truncated = True
                self.episode_info.termination_reason = "max_steps"
        
        # Record reward
        self.episode_info.reward_history.append(reward)
        
        # Get new observation
        obs_start_time = time.time() 
        observation = self._get_observation()
        info = self._get_info()
        obs_time = time.time() - obs_start_time 

        total_step_time = time.time() - start_time 

        # Add profiling info to the info dict (for monitoring via callbacks)
        info['profiling'] = {
            'total_step_ms': total_step_time * 1000,
            'uncertainty_calc_ms': uncertainty_time * 1000,
            'action_exec_ms': action_time * 1000,
            'get_obs_ms': obs_time * 1000
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_next_action(self) -> float:
        """
        Execute NEXT action: move to next frame.
        
        Returns:
            Step reward
        """
        # Move to next frame
        if self.current_frame_idx < FRAMES_PER_VIDEO - 1:
            self.current_frame_idx += 1
            self.episode_info.frames_analyzed += 1
            
            # Update LSTM with new frame
            self._update_temporal_memory(self.current_frame_idx)
        
        # Calculate step cost
        step_reward, _ = self.reward_calculator.calculate_step_reward(
            action=0,
            uncertainty_before=0,
            uncertainty_after=0,
            frames_analyzed=self.episode_info.frames_analyzed,
            confidence=self.episode_info.final_confidence
        )
        
        return step_reward
    
    def _execute_focus_action(self, uncertainty_before: float) -> float:
        """
        Execute FOCUS action: detailed analysis of current frame.
        
        Args:
            uncertainty_before: Uncertainty before action
            
        Returns:
            Step reward including information gain
        """
        # Apply attention-based analysis (simulated by re-processing with attention)
        current_features = self.current_frame_features[self.current_frame_idx:self.current_frame_idx+1]
        
        # Simulate attention mechanism by emphasizing certain features
        attention_weights = torch.softmax(current_features.abs().mean(dim=-1), dim=0)
        attended_features = current_features * attention_weights.unsqueeze(-1)
        
        # Update LSTM with refined features
        # Ensure attended_features has shape (batch_size=1, seq_len, feature_dim)
        # Ensure deterministic temporal memory updates for training stability
        with torch.no_grad():
            # attended_features is already (seq_len, feature_dim), need to add batch dim
            if attended_features.dim() == 2:
                attended_features = attended_features.unsqueeze(0)
            
            # Ensure temporal memory is in eval mode for deterministic update
            self.temporal_memory.eval()

            output, (self.lstm_hidden, self.lstm_cell) = self.temporal_memory(
                attended_features,
                (self.lstm_hidden, self.lstm_cell)
            )
        
        # Get new uncertainty
        uncertainty_after = self._get_current_uncertainty()
        
        # Calculate reward with information gain
        step_reward, _ = self.reward_calculator.calculate_step_reward(
            action=1,
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after,
            frames_analyzed=self.episode_info.frames_analyzed,
            confidence=self.episode_info.final_confidence
        )
        
        return step_reward
        
    
    def _execute_augment_action(self, uncertainty_before: float) -> float:
        """
        Execute AUGMENT action: apply augmentation and re-analyze.
        
        Args:
            uncertainty_before: Uncertainty before action
            
        Returns:
            Step reward including information gain
        """
        # Get current frame
        current_frame = self.current_video_frames[self.current_frame_idx]
        
        # Apply augmentation
        augmented_frame = self._apply_augmentation(current_frame)
        
        # Extract features from augmented frame
        with torch.no_grad():
            augmented_features = self.vision_backbone(augmented_frame.unsqueeze(0))
        
        # Update LSTM with augmented features
        # Ensure deterministic temporal memory updates for training stability
        with torch.no_grad():
            # Ensure augmented_features has shape (batch_size=1, seq_len=1, feature_dim)
            if augmented_features.dim() == 2:
                augmented_features = augmented_features.unsqueeze(0)
            
            # Ensure temporal memory is in eval mode for deterministic update
            self.temporal_memory.eval()

            output, (self.lstm_hidden, self.lstm_cell) = self.temporal_memory(
                augmented_features,
                (self.lstm_hidden, self.lstm_cell)
            )
        
        # Get new uncertainty
        uncertainty_after = self._get_current_uncertainty()
        
        # Calculate reward with information gain
        step_reward, _ = self.reward_calculator.calculate_step_reward(
            action=2,
            uncertainty_before=uncertainty_before,
            uncertainty_after=uncertainty_after,
            frames_analyzed=self.episode_info.frames_analyzed,
            confidence=self.episode_info.final_confidence
        )
        
        return step_reward
    
    def _execute_stop_action(self, prediction: int) -> float:
        """
        Execute STOP action: make final classification.
        
        Args:
            prediction: 0 for real, 1 for fake
            
        Returns:
            Terminal reward
        """
        self.episode_info.final_prediction = prediction
        self.episode_info.termination_reason = f"stop_{'fake' if prediction else 'real'}"
        
        # Calculate terminal reward
        terminal_reward, _ = self.reward_calculator.calculate_terminal_reward(
            prediction=prediction,
            true_label=self.true_label,
            confidence=self.episode_info.final_confidence,
            episode_length=self.episode_info.episode_steps,
            frames_analyzed=self.episode_info.frames_analyzed
        )
        
        if self.verbose:
            correct = (prediction == self.true_label)
            logger.info(f"Classification: {'CORRECT' if correct else 'WRONG'} "
                       f"(pred={prediction}, true={self.true_label}, "
                       f"conf={self.episode_info.final_confidence:.3f})")
        
        return terminal_reward
    
    def _update_temporal_memory(self, frame_idx: int):
        """Update LSTM with features from specified frame."""
        # Ensure frame features are properly shaped (batch_size=1, seq_len=1, feature_dim)
        frame_features = self.current_frame_features[frame_idx:frame_idx+1].unsqueeze(0)
        
        # Ensure hidden states maintain correct dimensions
        # Ensure deterministic temporal memory updates for training stability
        with torch.no_grad():
            # Ensure temporal memory is in eval mode for deterministic update
            self.temporal_memory.eval()
            output, (self.lstm_hidden, self.lstm_cell) = self.temporal_memory(
                frame_features,
                (self.lstm_hidden, self.lstm_cell)
            )
    
    def _get_current_uncertainty(self) -> float:
        """Calculate current uncertainty from classifier confidence."""
        # Import validation utility
        # Import reward calculation module
        from src.utils import validate_state_concatenation
        
        # Get temporal memory representation (B=1, D_lstm)
        temporal_memory = self.temporal_memory.get_final_hidden((self.lstm_hidden, self.lstm_cell))
        
        # Get current frame features (B=1, D_frame)
        # Handle edge case where index might be out of bounds at the very end of an episode
        if self.current_frame_idx < len(self.current_frame_features):
            current_frame_features = self.current_frame_features[self.current_frame_idx:self.current_frame_idx+1]
        else:
            # Use the last frame's features if index is at the limit
             current_frame_features = self.current_frame_features[-1:]
        
        # combined_features shape: (B=1, D_frame + D_lstm)
        combined_features = validate_state_concatenation(
            current_frame_features,
            temporal_memory,
            VISION_EMBEDDING_DIM + LSTM_EFFECTIVE_DIM
        )
        
        # The model used for uncertainty estimation in the environment state.
        eval_model = self.classifier_head

        # Get classification confidence and uncertainty
        if USE_BAYESIAN_UNCERTAINTY:
            # --- Monte Carlo Dropout (MCDO) Implementation - BATCHED OPTIMIZATION ---
            
            # 1. Activate dropout layers (sets dropout modules to train() mode)
            eval_model.enable_mc_dropout()
            
            # 2. Prepare batched input
            # Replicate the input features MC_SAMPLES times.
            # Shape before: (1, D_features) -> Shape after: (MC_SAMPLES, D_features)
            batched_features = combined_features.repeat(MC_SAMPLES, 1)

            # 3. Perform a single batched forward pass
            with torch.no_grad():
                # Optimize sampling using AMP if enabled (for RTX 5090)
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                
                with autocast(device_type=device_type, enabled=(USE_MIXED_PRECISION and device_type=='cuda')):
                    # Shape: (MC_SAMPLES, num_classes=2)
                    # Dropout acts independently on each item in the batch.
                    _, all_probs, _ = eval_model(batched_features)

            # 4. Restore to eval mode
            eval_model.eval() 

            # 5. Aggregate results
            # Shape: (1, num_classes=2) - calculating the mean across the samples
            mean_probs = torch.mean(all_probs, dim=0, keepdim=True)
            
            # 6. Calculate Confidence (max of mean probabilities)
            # Shape: (1,)
            confidence_tensor = torch.max(mean_probs, dim=-1)[0]
            
            # 7. Calculate Uncertainty (Normalized Predictive Entropy)
            epsilon = 1e-6 # Stability constant
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)
            
            # Normalize entropy
            max_entropy = torch.log(torch.tensor(2.0, device=entropy.device)) 
            uncertainty_tensor = entropy / max_entropy
            
            # Ensure it is capped at 1.0
            uncertainty_tensor = torch.clamp(uncertainty_tensor, 0.0, 1.0)

            # Extract scalar values
            confidence = confidence_tensor.item()
            uncertainty = uncertainty_tensor.item()

        else:
            # --- Deterministic Implementation (Original) ---
            # Model should already be in eval mode
            with torch.no_grad():
                # Ensure model is in eval mode just in case
                eval_model.eval()
                _, probs, confidence_tensor = eval_model(combined_features)
            
            confidence = confidence_tensor.item()
            # Calculate uncertainty as 1 - confidence
            uncertainty = (1.0 - confidence)
        
        # Update tracking
        self.episode_info.final_confidence = confidence
        self.episode_info.confidence_history.append(confidence)
        self.episode_info.uncertainty_history.append(uncertainty)
        
        return uncertainty
    
    def _apply_augmentation(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to frame for AUGMENT action.
        
        Args:
            frame: Input frame tensor (C, H, W)
            
        Returns:
            Augmented frame tensor
        """
        # Convert to numpy for augmentation
        frame_np = frame.cpu().permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Random brightness adjustment
        brightness_factor = 1.0 + random.uniform(-AUGMENT_BRIGHTNESS_RANGE, AUGMENT_BRIGHTNESS_RANGE)
        frame_np = cv2.convertScaleAbs(frame_np, alpha=brightness_factor, beta=0)
        
        # Random JPEG compression
        quality = random.randint(AUGMENT_JPEG_QUALITY_MIN, AUGMENT_JPEG_QUALITY_MAX)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame_np, encode_param)
        frame_np = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Random rotation
        angle = random.uniform(-AUGMENT_ROTATION_RANGE, AUGMENT_ROTATION_RANGE)
        center = (frame_np.shape[1] // 2, frame_np.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame_np = cv2.warpAffine(frame_np, rotation_matrix, (frame_np.shape[1], frame_np.shape[0]))
        
        # Convert back to tensor
        frame_tensor = torch.from_numpy(frame_np).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).to(self.device)
        
        return frame_tensor
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation dictionary.
        
        Returns:
            Observation containing state components
        """
        # Import dimension validator as per PDF Step 2.1
        from src.utils import dimension_validator
        
        # Get current frame features
        if self.current_frame_idx < len(self.current_frame_features):
            frame_features = self.current_frame_features[self.current_frame_idx]
        else:
            frame_features = torch.zeros(VISION_EMBEDDING_DIM, device=self.device)
        
        # Get temporal memory
        temporal_memory = self.temporal_memory.get_final_hidden((self.lstm_hidden, self.lstm_cell))
        temporal_memory = temporal_memory.squeeze(0)
        
        # Add dimension validation for frame_features and temporal_memory
        dimension_validator(
            (VISION_EMBEDDING_DIM,),
            frame_features,
            "frame_features in _get_observation"
        )
        
        dimension_validator(
            (LSTM_EFFECTIVE_DIM,),
            temporal_memory,
            "temporal_memory in _get_observation"
        )
        
        # Calculate frame position
        frame_position = self.current_frame_idx / FRAMES_PER_VIDEO
        
        # Get current uncertainty
        uncertainty = self._get_current_uncertainty()
        
        # Handle tensor conversion - use detach() if tensor requires grad
        if frame_features.requires_grad:
            frame_features_np = frame_features.cpu().detach().numpy().astype(np.float32)
        else:
            frame_features_np = frame_features.cpu().numpy().astype(np.float32)
            
        if temporal_memory.requires_grad:
            temporal_memory_np = temporal_memory.cpu().detach().numpy().astype(np.float32)
        else:
            temporal_memory_np = temporal_memory.cpu().numpy().astype(np.float32)
        
        observation = {
            'frame_features': frame_features_np,
            'temporal_memory': temporal_memory_np,
            'frame_position': np.array([frame_position], dtype=np.float32),
            'uncertainty': np.array([uncertainty], dtype=np.float32)
        }
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get episode information dictionary.
        
        Returns:
            Info dict with episode statistics
        """
        # Get the latest uncertainty value from the history
        current_uncertainty = self.episode_info.uncertainty_history[-1] if self.episode_info.uncertainty_history else 0.0

        return {
            'video_id': self.episode_info.video_id,
            'true_label': self.episode_info.true_label,
            'final_prediction': self.episode_info.final_prediction,
            'frames_analyzed': self.episode_info.frames_analyzed,
            'episode_steps': self.episode_info.episode_steps,
            'current_confidence': self.episode_info.final_confidence,
            'current_uncertainty': current_uncertainty,
            'action_history': self.episode_info.action_history.copy(),
            'termination_reason': self.episode_info.termination_reason,
            'current_frame_idx': self.current_frame_idx
        }
    
    def get_current_frame(self) -> torch.Tensor:
        """
        Get the current RGB frame being analyzed.
        
        Returns:
            Current frame tensor (C, H, W) or None if not available
        """
        if self.current_video_frames is not None and self.current_frame_idx < len(self.current_video_frames):
            # Return frame on same device as the environment
            return self.current_video_frames[self.current_frame_idx].clone()
        return None
    
    def get_current_label(self):
        """Get the true label of the current video."""
        return self.true_label
    
    @property
    def current_label(self):
        """Property to get the true label of the current video."""
        return self.true_label
    
    def render(self, mode='human'):
        """Render environment (optional visualization)."""
        if mode == 'human' and self.verbose:
            print(f"Step {self.episode_info.episode_steps}: "
                  f"Frame {self.current_frame_idx}/{FRAMES_PER_VIDEO}, "
                  f"Confidence: {self.episode_info.final_confidence:.3f}")
    
    def close(self):
        """Clean up environment resources."""
        pass


if __name__ == "__main__":
    # Test environment
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    logger.info("Testing DeepfakeEnv...")
    
    # Create dummy components
    from data_loader import ProcessedVideoDataset
    
    dataset = ProcessedVideoDataset("train", validate_frames=False)
    vision_backbone = VisionBackbone().to(DEVICE)
    temporal_memory = TemporalMemory().to(DEVICE)
    classifier_head = ClassifierHead().to(DEVICE)
    reward_calculator = RewardCalculator(verbose=True)
    
    # Create environment
    env = DeepfakeEnv(
        dataset=dataset,
        vision_backbone=vision_backbone,
        temporal_memory=temporal_memory,
        classifier_head=classifier_head,
        reward_calculator=reward_calculator,
        verbose=True
    )
    
    if len(dataset) > 0:
        # Run test episode
        obs, info = env.reset()
        logger.info(f"Initial observation keys: {obs.keys()}")
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 20:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            logger.info(f"Step {steps}: Action={env.action_names[action]}, "
                       f"Reward={reward:.3f}, Done={done}")
        
        logger.info(f"Episode finished: Total reward={total_reward:.3f}, "
                   f"Steps={steps}, Reason={info['termination_reason']}")
        
        print("Environment test completed successfully!")
    else:
        logger.warning("No data available for testing")