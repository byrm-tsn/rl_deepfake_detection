"""
Reward system for EAGER algorithm.
Implements sophisticated reward structure to incentivize correct classification
while penalizing guessing and encouraging efficient evidence gathering.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from src.config import (
    CORRECT_CLASSIFICATION_REWARD,
    INCORRECT_CLASSIFICATION_PENALTY,
    NO_DECISION_PENALTY,
    NEXT_ACTION_COST,
    FOCUS_ACTION_COST,
    AUGMENT_ACTION_COST,
    MAX_INFO_GAIN_BONUS,
    INFO_GAIN_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MIN_FRAMES_BEFORE_DECISION,
    CONFIDENCE_GATE_PENALTY,
    EARLY_DECISION_BONUS,
    OPTIMAL_DECISION_STEPS,
    MAX_EPISODE_STEPS,
    CONFIDENCE_CURRICULUM_START,
    CONFIDENCE_CURRICULUM_END,
    CONFIDENCE_CURRICULUM_STEPS
)

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Data class to track individual reward components."""
    terminal_reward: float = 0.0
    step_cost: float = 0.0
    info_gain_bonus: float = 0.0
    confidence_penalty: float = 0.0
    proper_score: float = 0.0
    exploration_bonus: float = 0.0 
    total: float = 0.0
    
    def calculate_total(self):
        """Calculate total reward from components."""
        self.total = (
            self.terminal_reward +
            self.step_cost +
            self.info_gain_bonus +
            self.confidence_penalty +
            self.proper_score +
            self.exploration_bonus 
        )
        return self.total


class RewardCalculator:
    """
    Sophisticated reward calculator for EAGER agent.
    Implements multi-component reward system with confidence gating.
    """
    
    def __init__(
        self,
        frozen_evaluator: Optional[Any] = None,
        use_proper_scoring: bool = True,
        verbose: bool = False,
        enable_curriculum: bool = True
    ):
        """
        Initialize reward calculator.
        
        Args:
            frozen_evaluator: Frozen model for unbiased reward calculation
            use_proper_scoring: Include proper scoring rule component
            verbose: Log detailed reward breakdowns
            enable_curriculum: Enable curriculum learning for confidence
        """
        self.frozen_evaluator = frozen_evaluator
        self.use_proper_scoring = use_proper_scoring
        self.verbose = verbose
        self.enable_curriculum = enable_curriculum
        self.total_steps = 0  # Track total steps for curriculum
        
        # Action costs mapping
        self.action_costs = {
            0: NEXT_ACTION_COST,      # NEXT
            1: FOCUS_ACTION_COST,      # FOCUS
            2: AUGMENT_ACTION_COST,    # AUGMENT
            3: 0.0,                    # STOP_REAL (terminal)
            4: 0.0                     # STOP_FAKE (terminal)
        }
        
        logger.info("Initialized reward calculator")
    
    def calculate_step_reward(
        self,
        action: int,
        uncertainty_before: float,
        uncertainty_after: float,
        frames_analyzed: int,
        confidence: float
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for non-terminal actions.
        
        Args:
            action: Action taken (0=NEXT, 1=FOCUS, 2=AUGMENT)
            uncertainty_before: Uncertainty before action
            uncertainty_after: Uncertainty after action
            frames_analyzed: Number of frames analyzed so far
            confidence: Current confidence level
            
        Returns:
            total_reward: Total step reward
            components: Breakdown of reward components
        """
        components = RewardComponents()
        
        # Step cost based on action
        components.step_cost = self.action_costs[action]
        
        # Information gain bonus
        if action in [1, 2]:  # FOCUS or AUGMENT
            uncertainty_reduction = uncertainty_before - uncertainty_after
            
            if uncertainty_reduction > INFO_GAIN_THRESHOLD:
                # Reward proportional to uncertainty reduction
                components.info_gain_bonus = min(
                    uncertainty_reduction * 2.0,
                    MAX_INFO_GAIN_BONUS
                )
                
                if self.verbose:
                    logger.info(f"Info gain bonus: {components.info_gain_bonus:.3f} "
                               f"(reduction: {uncertainty_reduction:.3f})")
        
        # Step 3.2: Add exploration bonus based on frame coverage as per PDF
        # Reward exploring more frames to avoid premature decisions
        from src.config import FRAMES_PER_VIDEO
        frame_coverage_ratio = frames_analyzed / FRAMES_PER_VIDEO
        
        # Give bonus for exploring more frames (up to 50% coverage)
        if frame_coverage_ratio < 0.5:
            components.exploration_bonus = 0.1 * (1.0 - frame_coverage_ratio * 2)  
            
            if self.verbose:
                logger.info(f"Exploration bonus: {components.exploration_bonus:.3f} "
                           f"(coverage: {frame_coverage_ratio:.2f})")
        
        # Calculate total
        components.calculate_total()
        
        return components.total, components
    
    def calculate_terminal_reward(
        self,
        prediction: int,
        true_label: int,
        confidence: float,
        episode_length: int,
        frames_analyzed: int
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for terminal classification actions.
        
        Args:
            prediction: Agent's prediction (0=real, 1=fake)
            true_label: Ground truth label (0=real, 1=fake)
            confidence: Classification confidence
            episode_length: Total steps taken in episode
            frames_analyzed: Number of frames actually analyzed
            
        Returns:
            total_reward: Total terminal reward
            components: Breakdown of reward components
        """
        components = RewardComponents()
        
        # Check if prediction is correct
        correct = (prediction == true_label)
        
        if correct:
            # Base reward for correct classification
            components.terminal_reward = CORRECT_CLASSIFICATION_REWARD
            
            
            # Increase reward to encourage more confident correct decisions
            success_bonus = min(confidence * 2.0, 1.0)  # Bonus proportional to confidence
            components.terminal_reward += success_bonus
            
            # Early decision bonus (if not timed out)
            if episode_length < MAX_EPISODE_STEPS:
                components.terminal_reward += EARLY_DECISION_BONUS
                
                # Additional bonus for optimal timing
                if episode_length <= OPTIMAL_DECISION_STEPS:
                    timing_bonus = (OPTIMAL_DECISION_STEPS - episode_length) * 0.1
                    components.terminal_reward += timing_bonus
                    if self.verbose:
                        logger.info(f"Optimal timing bonus: {timing_bonus:.3f}")
            
            # Efficiency bonus based on frames analyzed
            efficiency_bonus = max(0, (30 - frames_analyzed) * 0.05)
            components.terminal_reward += efficiency_bonus
            
            if self.verbose:
                logger.info(f"Correct classification! Success bonus: {success_bonus:.3f}, "
                           f"Total bonus: {components.terminal_reward - CORRECT_CLASSIFICATION_REWARD:.3f}")
        else:
            # Base penalty for incorrect classification
            components.terminal_reward = INCORRECT_CLASSIFICATION_PENALTY
            
            # Still give small early decision bonus to encourage attempting
            if episode_length < MAX_EPISODE_STEPS:
                components.terminal_reward += EARLY_DECISION_BONUS * 0.3
            
            if self.verbose:
                logger.info(f"Incorrect classification! Net penalty: {components.terminal_reward:.3f}")
        
        # Proper scoring rule component
        if self.use_proper_scoring:
            components.proper_score = self._calculate_proper_score(
                correct, confidence
            )
        
        # Calculate total
        components.calculate_total()
        
        return components.total, components
    
    def calculate_no_decision_penalty(
        self,
        episode_length: int,
        frames_analyzed: int,
        final_confidence: float
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate penalty when agent doesn't make a decision.
        
        Args:
            episode_length: Total steps taken
            frames_analyzed: Number of frames analyzed
            final_confidence: Final confidence level
            
        Returns:
            total_reward: Total penalty
            components: Breakdown of reward components
        """
        components = RewardComponents()
        
        # Base penalty for not deciding
        components.terminal_reward = NO_DECISION_PENALTY
        
        # Additional penalty if confidence was high but still didn't decide
        if final_confidence > CONFIDENCE_THRESHOLD:
            components.confidence_penalty = -5.0
            if self.verbose:
                logger.info(f"High confidence ({final_confidence:.3f}) but no decision!")
        
        # Calculate total
        components.calculate_total()
        
        return components.total, components
    
    def apply_confidence_gate(
        self,
        action: int,
        confidence: float,
        frames_analyzed: int
    ) -> Tuple[int, float]:
        """
        Apply confidence gating mechanism to prevent premature decisions.
        
        Args:
            action: Proposed action
            confidence: Current confidence level
            frames_analyzed: Number of frames analyzed
            
        Returns:
            modified_action: Potentially modified action
            gate_penalty: Penalty for gate violation
        """
        gate_penalty = 0.0
        modified_action = action
        
        # Check if trying to make terminal decision
        if action in [3, 4]:  # STOP_REAL or STOP_FAKE
            
            # Check minimum frames requirement
            if frames_analyzed < MIN_FRAMES_BEFORE_DECISION:
                modified_action = 0  # Force NEXT action
                gate_penalty = CONFIDENCE_GATE_PENALTY
                
                if self.verbose:
                    logger.warning(f"Blocked early decision at frame {frames_analyzed}")
            
            # Check confidence threshold (with curriculum learning)
            current_threshold = self.get_current_confidence_threshold()
            if confidence < current_threshold:
                modified_action = 0  # Force NEXT action
                gate_penalty = CONFIDENCE_GATE_PENALTY
                
                if self.verbose:
                    logger.warning(f"Blocked low-confidence decision: {confidence:.3f} < {current_threshold:.3f}")
        
        return modified_action, gate_penalty
    
    def get_current_confidence_threshold(self) -> float:
        """
        Get current confidence threshold with curriculum learning.
        
        Returns:
            Current confidence threshold
        """
        if not self.enable_curriculum:
            return CONFIDENCE_THRESHOLD
        
        # Linear curriculum from start to end threshold
        progress = min(1.0, self.total_steps / CONFIDENCE_CURRICULUM_STEPS)
        threshold = CONFIDENCE_CURRICULUM_START + \
                   (CONFIDENCE_CURRICULUM_END - CONFIDENCE_CURRICULUM_START) * progress
        
        return threshold
    
    def update_curriculum_progress(self, steps: int):
        """Update curriculum learning progress."""
        self.total_steps += steps
    
    def _calculate_proper_score(
        self,
        correct: bool,
        confidence: float
    ) -> float:
        """
        Calculate proper scoring rule component (log-likelihood).
        
        Args:
            correct: Whether prediction was correct
            confidence: Prediction confidence
            
        Returns:
            Proper score component
        """
        # Ensure confidence is in valid range
        confidence = max(0.01, min(0.99, confidence))
        
        if correct:
            # Reward high confidence on correct predictions
            proper_score = np.log(confidence)
        else:
            # Heavily penalize high confidence on wrong predictions
            proper_score = np.log(1 - confidence)
        
        # Scale to reasonable range
        proper_score *= 2.0
        
        return proper_score
    
    def calculate_cumulative_reward(
        self,
        episode_rewards: list,
        gamma: float = 0.99
    ) -> float:
        """
        Calculate discounted cumulative reward for an episode.
        
        Args:
            episode_rewards: List of step rewards
            gamma: Discount factor
            
        Returns:
            Discounted cumulative reward
        """
        cumulative = 0.0
        discount = 1.0
        
        for reward in episode_rewards:
            cumulative += discount * reward
            discount *= gamma
        
        return cumulative
    
    def get_reward_statistics(
        self,
        reward_history: list
    ) -> Dict[str, float]:
        """
        Calculate statistics from reward history.
        
        Args:
            reward_history: List of episode rewards
            
        Returns:
            Dictionary of statistics
        """
        if not reward_history:
            return {}
        
        rewards = np.array(reward_history)
        
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        }


class AdaptiveRewardShaper:
    """
    Adaptive reward shaping to improve training stability.
    Adjusts reward components based on training progress.
    """
    
    def __init__(
        self,
        initial_exploration_bonus: float = 0.1,
        decay_rate: float = 0.999
    ):
        """
        Initialize adaptive reward shaper.
        
        Args:
            initial_exploration_bonus: Initial bonus for exploration
            decay_rate: Decay rate for exploration bonus
        """
        self.exploration_bonus = initial_exploration_bonus
        self.decay_rate = decay_rate
        self.episode_count = 0
    
    def shape_reward(
        self,
        base_reward: float,
        action: int,
        episode_num: int
    ) -> float:
        """
        Apply adaptive reward shaping.
        
        Args:
            base_reward: Base reward from calculator
            action: Action taken
            episode_num: Current episode number
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        # Add exploration bonus for analysis actions early in training
        if action in [1, 2]:  # FOCUS or AUGMENT
            current_bonus = self.exploration_bonus * (self.decay_rate ** episode_num)
            shaped_reward += current_bonus
        
        return shaped_reward
    
    def update(self):
        """Update shaping parameters after each episode."""
        self.episode_count += 1


if __name__ == "__main__":
    # Test reward calculation
    logger.info("Testing reward system...")
    
    # Create reward calculator
    calculator = RewardCalculator(verbose=True)
    
    # Test step reward
    print("\n=== Testing step rewards ===")
    
    # NEXT action
    reward, components = calculator.calculate_step_reward(
        action=0,
        uncertainty_before=0.5,
        uncertainty_after=0.5,
        frames_analyzed=10,
        confidence=0.6
    )
    print(f"NEXT action reward: {reward:.3f}")
    
    # FOCUS action with information gain
    reward, components = calculator.calculate_step_reward(
        action=1,
        uncertainty_before=0.5,
        uncertainty_after=0.3,
        frames_analyzed=10,
        confidence=0.7
    )
    print(f"FOCUS action reward: {reward:.3f} (info gain: {components.info_gain_bonus:.3f})")
    
    # Test terminal rewards
    print("\n=== Testing terminal rewards ===")
    
    # Correct classification
    reward, components = calculator.calculate_terminal_reward(
        prediction=1,
        true_label=1,
        confidence=0.9,
        episode_length=20,
        frames_analyzed=15
    )
    print(f"Correct classification reward: {reward:.3f}")
    
    # Incorrect classification
    reward, components = calculator.calculate_terminal_reward(
        prediction=0,
        true_label=1,
        confidence=0.9,
        episode_length=30,
        frames_analyzed=25
    )
    print(f"Incorrect classification reward: {reward:.3f}")
    
    # Test confidence gating
    print("\n=== Testing confidence gating ===")
    
    # Try early decision
    action, penalty = calculator.apply_confidence_gate(
        action=3,  # STOP_REAL
        confidence=0.9,
        frames_analyzed=3  # Too early
    )
    print(f"Early decision: action={action}, penalty={penalty:.3f}")
    
    # Try low-confidence decision
    action, penalty = calculator.apply_confidence_gate(
        action=4,  # STOP_FAKE
        confidence=0.6,  # Too low
        frames_analyzed=10
    )
    print(f"Low-confidence decision: action={action}, penalty={penalty:.3f}")
    
    print("\nReward system tests completed successfully!")