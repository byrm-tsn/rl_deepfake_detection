"""
EAGER Policy Network implementation.
PPO-LSTM agent for strategic deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
import logging

from src.config import (
    VISION_EMBEDDING_DIM, LSTM_EFFECTIVE_DIM,
    POLICY_HIDDEN_DIM, VALUE_HIDDEN_DIM, POLICY_DROPOUT,
    NUM_ACTIONS, STATE_DIM, DEVICE
)

logger = logging.getLogger(__name__)


class EagerStateEncoder(nn.Module):
    """
    Encodes the multi-component state into a unified representation.
    """
    
    def __init__(
        self,
        frame_features_dim: int = VISION_EMBEDDING_DIM,
        temporal_memory_dim: int = LSTM_EFFECTIVE_DIM,
        output_dim: int = 512
    ):
        """
        Initialize state encoder.
        
        Args:
            frame_features_dim: Dimension of frame features
            temporal_memory_dim: Dimension of temporal memory
            output_dim: Output encoding dimension
        """
        super().__init__()
        
        # Calculate total input dimension
        # frame_features + temporal_memory + position(1) + uncertainty(1)
        input_dim = frame_features_dim + temporal_memory_dim + 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(POLICY_DROPOUT)
        )
        
        self.output_dim = output_dim
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observation dictionary into state vector.
        
        Args:
            observation: Dictionary with state components
            
        Returns:
            Encoded state tensor
        """
        # Import dimension validator 
        from src.utils import dimension_validator
        
        # Extract components
        frame_features = observation['frame_features']
        temporal_memory = observation['temporal_memory']
        frame_position = observation['frame_position']
        uncertainty = observation['uncertainty']
        
        # Handle batch dimension
        if frame_features.dim() == 1:
            frame_features = frame_features.unsqueeze(0)
            temporal_memory = temporal_memory.unsqueeze(0)
            frame_position = frame_position.unsqueeze(0)
            uncertainty = uncertainty.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = frame_features.shape[0]
        
        # Add dimension validation for state components
        dimension_validator(
            (batch_size, VISION_EMBEDDING_DIM),
            frame_features,
            "frame_features in EagerStateEncoder"
        )
        
        dimension_validator(
            (batch_size, LSTM_EFFECTIVE_DIM),
            temporal_memory,
            "temporal_memory in EagerStateEncoder"
        )
        
        dimension_validator(
            (batch_size, 1),
            frame_position,
            "frame_position in EagerStateEncoder"
        )
        
        dimension_validator(
            (batch_size, 1),
            uncertainty,
            "uncertainty in EagerStateEncoder"
        )
        
        # Concatenate all components
        state = torch.cat([
            frame_features,
            temporal_memory,
            frame_position,
            uncertainty
        ], dim=-1)
        
        # Validate concatenated state dimension
        expected_state_dim = VISION_EMBEDDING_DIM + LSTM_EFFECTIVE_DIM + 2
        dimension_validator(
            (batch_size, expected_state_dim),
            state,
            "concatenated_state in EagerStateEncoder"
        )
        
        # Encode
        encoded = self.encoder(state)
        
        return encoded


class EagerPolicy(nn.Module):
    """
    EAGER policy network with separate actor and critic heads.
    """
    
    def __init__(
        self,
        state_encoder: EagerStateEncoder,
        action_dim: int = NUM_ACTIONS,
        policy_hidden_dim: int = POLICY_HIDDEN_DIM,
        value_hidden_dim: int = VALUE_HIDDEN_DIM,
        dropout: float = POLICY_DROPOUT
    ):
        """
        Initialize EAGER policy.
        
        Args:
            state_encoder: State encoding network
            action_dim: Number of actions
            policy_hidden_dim: Hidden dimension for policy network
            value_hidden_dim: Hidden dimension for value network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_encoder = state_encoder
        encoded_dim = state_encoder.output_dim
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(encoded_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Actor head (policy)
        self.policy_net = nn.Sequential(
            nn.Linear(256, policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_hidden_dim, action_dim)
        )
        
        # Critic head (value)
        self.value_net = nn.Sequential(
            nn.Linear(256, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
        
        logger.info("Initialized EAGER policy network")
    
    def forward(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            observation: State observation dictionary
            deterministic: Use deterministic action selection
            
        Returns:
            action_logits: Action logits
            value: State value estimate
            action: Selected action
        """
        # Encode state
        encoded_state = self.state_encoder(observation)
        
        # Shared features
        shared_features = self.shared_net(encoded_state)
        
        # Get action logits and value
        action_logits = self.policy_net(shared_features)
        value = self.value_net(shared_features)
        
        # Action selection
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action_logits, value.squeeze(-1), action
    
    def get_action_probs(
        self,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get action probability distribution.
        
        Args:
            observation: State observation
            
        Returns:
            Action probabilities
        """
        encoded_state = self.state_encoder(observation)
        shared_features = self.shared_net(encoded_state)
        action_logits = self.policy_net(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs
    
    def get_value(
        self,
        observation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            observation: State observation
            
        Returns:
            State value
        """
        encoded_state = self.state_encoder(observation)
        shared_features = self.shared_net(encoded_state)
        value = self.value_net(shared_features)
        
        return value.squeeze(-1)


class EagerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Stable Baselines3 integration.
    """
    
    def __init__(self, observation_space: gym.Space):
        """
        Initialize feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
        """
        # Initialize with dummy features dimension (will be overridden)
        super().__init__(observation_space, features_dim=1)
        
        # Create state encoder
        self.state_encoder = EagerStateEncoder()
        
        # Update features dimension
        self._features_dim = self.state_encoder.output_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Extracted features
        """
        # Convert tensor observations to dictionary format
        return self.state_encoder(observations)


class EagerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for EAGER agent.
    Integrates with Stable Baselines3 PPO.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        **kwargs
    ):
        """
        Initialize EAGER actor-critic policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            **kwargs: Additional arguments
        """
        # Use custom feature extractor
        kwargs["features_extractor_class"] = EagerFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {}
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )
        
        logger.info("Initialized EAGER Actor-Critic Policy for PPO")


class EagerAgent:
    """
    Complete EAGER agent with training capabilities.
    """
    
    def __init__(
        self,
        env,
        vision_backbone,
        temporal_memory,
        classifier_head,
        learning_rate: float = 3e-4,
        device: str = DEVICE
    ):
        """
        Initialize EAGER agent.
        
        Args:
            env: DeepfakeEnv environment
            vision_backbone: Vision feature extractor
            temporal_memory: LSTM network
            classifier_head: Classification head
            learning_rate: Learning rate
            device: Computing device
        """
        self.env = env
        self.vision_backbone = vision_backbone
        self.temporal_memory = temporal_memory
        self.classifier_head = classifier_head
        self.device = device
        
        # Create state encoder and policy
        self.state_encoder = EagerStateEncoder().to(device)
        self.policy = EagerPolicy(self.state_encoder).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        logger.info("Initialized EAGER Agent")
    
    def act(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> int:
        """
        Select action given observation.
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic policy
            
        Returns:
            Selected action
        """
        # Import dimension validator
        from src.utils import dimension_validator
        
        # Convert observation to tensors
        obs_tensors = {
            key: torch.from_numpy(val).float().to(self.device)
            for key, val in observation.items()
        }
        
        # Validate observation dimensions before processing
        dimension_validator(
            (VISION_EMBEDDING_DIM,),
            obs_tensors['frame_features'],
            "frame_features in EagerAgent.act"
        )
        
        dimension_validator(
            (LSTM_EFFECTIVE_DIM,),
            obs_tensors['temporal_memory'],
            "temporal_memory in EagerAgent.act"
        )
        
        # Get action from policy
        with torch.no_grad():
            _, _, action = self.policy(obs_tensors, deterministic)
        
        return action.item()
    
    def compute_returns(
        self,
        rewards: List[float],
        gamma: float = 0.99
    ) -> List[float]:
        """
        Compute discounted returns.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            List of returns
        """
        returns = []
        running_return = 0
        
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        
        return returns
    
    def train_step(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Perform one training step on collected trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Training metrics
        """
        # Prepare batch data
        observations = []
        actions = []
        returns = []
        
        for traj in trajectories:
            observations.extend(traj['observations'])
            actions.extend(traj['actions'])
            returns.extend(self.compute_returns(traj['rewards']))
        
        # Convert to tensors
        # Compute loss and update
        
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0
        }
        
        return metrics
    
    def save(self, path: str):
        """Save agent model."""
        torch.save({
            'state_encoder': self.state_encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        logger.info(f"Saved agent to {path}")
    
    def load(self, path: str):
        """Load agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded agent from {path}")


if __name__ == "__main__":
    # Test agent components
    import gymnasium as gym
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    logger.info("Testing EAGER Agent components...")  
    observation = {
        'frame_features': torch.randn(VISION_EMBEDDING_DIM),
        'temporal_memory': torch.randn(LSTM_EFFECTIVE_DIM),
        'frame_position': torch.tensor([0.5]),
        'uncertainty': torch.tensor([0.3])
    }
    
    # Test state encoder
    encoder = EagerStateEncoder()
    encoded = encoder(observation)
    logger.info(f"Encoded state shape: {encoded.shape}")
    
    # Test policy network
    policy = EagerPolicy(encoder)
    action_logits, value, action = policy(observation, deterministic=False)
    logger.info(f"Action logits shape: {action_logits.shape}")
    logger.info(f"Value: {value.item():.3f}")
    logger.info(f"Selected action: {action.item()}")
    
    # Test action probabilities
    action_probs = policy.get_action_probs(observation)
    logger.info(f"Action probabilities: {action_probs.squeeze().tolist()}")
    
    print("Agent tests completed successfully!")