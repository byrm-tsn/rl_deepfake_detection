"""
GRPO (Group Relative Policy Optimization) Trainer for Phase 3 Fine-tuning
Implements GRPO using TorchRL for improved policy optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import warnings
from tqdm import tqdm

from torchrl.envs import ParallelEnv, EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict

from src.config import *
from src.deepfake_env import DeepfakeEnv
from src.utils import set_random_seeds

class GRPOBuffer:
    """Custom buffer for GRPO that groups trajectories for relative reward computation"""
    
    def __init__(self, buffer_size: int = 10000, group_size: int = 8):
        self.buffer = deque(maxlen=buffer_size)
        self.group_size = group_size
        
    def add(self, trajectory: Dict[str, torch.Tensor]):
        self.buffer.append(trajectory)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample_groups(self, num_groups: int) -> List[List[Dict]]:
        """Sample groups of trajectories for relative reward computation"""
        required_size = self.group_size * num_groups
        if len(self.buffer) < required_size:
            # If not enough data, return as many groups as possible
            available_groups = len(self.buffer) // self.group_size
            if available_groups == 0:
                return []
            num_groups = available_groups
            required_size = self.group_size * num_groups
        
        indices = np.random.choice(len(self.buffer), 
                                 size=required_size, 
                                 replace=False)
        groups = []
        for i in range(num_groups):
            group_indices = indices[i * self.group_size:(i + 1) * self.group_size]
            group = [self.buffer[idx] for idx in group_indices]
            groups.append(group)
        return groups

class GRPOLoss(nn.Module):
    """GRPO loss implementation with relative reward normalization"""
    
    def __init__(self, 
                 ppo_model: nn.Module,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 group_size: int = 8):
        super().__init__()
        self.ppo_model = ppo_model
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.group_size = group_size
        self.gamma = 0.99  # Add gamma as attribute
        
    def compute_relative_advantages(self, trajectories: List[Dict]) -> List[torch.Tensor]:
        """
        Compute TRUE GRPO advantages with correct group-relative ranking.
        
        GRPO ranks trajectories within each group and applies ranking-based
        scaling to advantages. Better trajectories get positive scaling,
        worse trajectories get negative scaling.
        """
        # Step 1: Compute returns and advantages for each trajectory
        trajectory_returns = []
        trajectory_advantages = []
        
        for i, traj in enumerate(trajectories):
            rewards = traj['rewards']
            values = traj['values']
            dones = traj['dones']
            
            # Ensure values is 1D (squeeze out extra dimensions)
            if values.dim() > 1:
                values = values.squeeze(-1)
            
            # Compute discounted returns for this trajectory
            returns = torch.zeros_like(rewards)
            running_return = 0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0
                running_return = rewards[t] + self.gamma * running_return
                returns[t] = running_return
            
            # Store total trajectory return (use discounted return for fair ranking)
            # Use the final return value which represents the discounted sum
            total_return = float(returns[0])  # First timestep has full discounted return
            trajectory_returns.append(total_return)
            
            # Store the returns for later use
            traj['returns'] = returns
            
            # Compute advantages (returns - values)
            advantages = returns - values
            trajectory_advantages.append(advantages)
        
        # Step 2: Apply GRPO group-relative ranking
        group_size = self.group_size
        num_trajectories = len(trajectories)
        
        normalized_advantages = []
        
        # Process trajectories in groups
        for group_start in range(0, num_trajectories, group_size):
            group_end = min(group_start + group_size, num_trajectories)
            group_indices = list(range(group_start, group_end))
            group_returns = trajectory_returns[group_start:group_end]
            current_group_size = len(group_indices)
            
            if current_group_size < 2:
                # Can't rank single trajectory, use original advantages
                for i in group_indices:
                    normalized_advantages.append(trajectory_advantages[i])
                logging.debug(f"Group {group_start//group_size}: Single trajectory, using original advantages")
                continue
            
            # GRPO Core: Rank trajectories by their returns
            # Create list of (index, return) pairs and sort by return
            indexed_returns = [(i, trajectory_returns[i]) for i in group_indices]
            indexed_returns.sort(key=lambda x: x[1], reverse=True)  # Sort by return, descending
            
            # Create ranking multipliers
            # Linear ranking from +1 (best) to -1 (worst)
            ranking_multipliers = {}
            group_info = []
            for rank, (traj_idx, traj_return) in enumerate(indexed_returns):
                # Linear interpolation from 1.0 to -1.0
                if current_group_size > 1:
                    multiplier = 1.0 - (2.0 * rank) / (current_group_size - 1)
                else:
                    multiplier = 0.0
                ranking_multipliers[traj_idx] = multiplier
                group_info.append((rank+1, traj_idx, traj_return, multiplier))
            
            # Log group ranking (important for monitoring GRPO behavior)
            logging.info(f"GRPO Group {group_start//group_size} Ranking:")
            for rank, idx, ret, mult in group_info:
                sign = "+" if mult >= 0 else ""
                logging.info(f"  Rank {rank}/{current_group_size}: Traj {idx} (return={ret:.2f}) → multiplier={sign}{mult:.3f}")
            
            # Apply ranking multipliers to advantages
            for i in group_indices:
                multiplier = ranking_multipliers[i]
                original_advantages = trajectory_advantages[i]
                
                # GRPO: Scale advantages by ranking multiplier
                # All timesteps in a trajectory get the same multiplier
                scaled_advantages = original_advantages * multiplier
                
                # Clip advantages to prevent extreme values (but preserve ranking signal!)
                # Do NOT normalize again as it destroys the GRPO ranking effect
                scaled_advantages = torch.clamp(scaled_advantages, min=-5.0, max=5.0)
                
                normalized_advantages.append(scaled_advantages)
            
            # Track group statistics for monitoring (optional logging)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                group_mean = np.mean(group_returns)
                group_std = np.std(group_returns)
                logging.debug(f"Group {group_start//group_size}: mean return={group_mean:.2f}, std={group_std:.2f}")
        
        return normalized_advantages
    
    def compute_relative_advantages_flat(self, rewards: torch.Tensor, 
                                        values: torch.Tensor,
                                        dones: torch.Tensor,
                                        gamma: float = 0.99,
                                        gae_lambda: float = 0.95) -> torch.Tensor:
        """Fallback: Compute advantages for flattened data (used in forward pass)"""
        # Simple GAE computation for concatenated timesteps
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute returns
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Advantages are returns - values
        advantages = returns - values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages
    
    def forward(self, batch: TensorDict) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss with gradient clipping safeguards"""
        # Extract data - all should be 1D tensors for concatenated batch
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        rewards = batch["rewards"]
        values = batch["values"]
        dones = batch["dones"]
        
        # Debug logging
        logging.debug(f"GRPOLoss forward: states shape={states.shape}, device={states.device}")
        
        # Check if we have pre-computed advantages (from trajectory normalization)
        if "advantages" in batch:
            advantages = batch["advantages"]
            returns = batch["returns"] if "returns" in batch else advantages + values
        else:
            # Fallback to simple normalization for flattened data
            advantages = self.compute_relative_advantages_flat(rewards, values, dones, 
                                                              gamma=self.gamma, 
                                                              gae_lambda=0.95)
            returns = advantages + values
        
        # Compute current policy outputs WITH gradients
        #where we need gradients for backprop
        logging.debug(f"Computing policy forward pass...")
        action_logits, state_values = self.ppo_model.forward(states)
        logging.debug(f"Forward pass complete: logits shape={action_logits.shape}")
        
        # Ensure state_values is 1D
        if state_values.dim() > 1:
            state_values = state_values.squeeze(-1)
        
        # Ensure 1D tensors
        if actions.dim() > 1:
            actions = actions.squeeze()
        if old_log_probs.dim() > 1:
            old_log_probs = old_log_probs.squeeze()
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        if values.dim() > 1:
            values = values.squeeze()
        if dones.dim() > 1:
            dones = dones.squeeze()
        if state_values.dim() > 1:
            state_values = state_values.squeeze()
        if advantages.dim() > 1:
            advantages = advantages.squeeze()
        if returns.dim() > 1:
            returns = returns.squeeze()
        
        # Compute log probabilities
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # Ensure actions are long type for indexing
        if actions.dtype != torch.long:
            actions = actions.long()
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Add gradient safeguards
        ratio = torch.clamp(ratio, min=1e-8, max=100)
        
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss with clipping
        value_pred_clipped = values + torch.clamp(state_values - values, 
                                                  -self.clip_epsilon, 
                                                  self.clip_epsilon)
        value_loss = torch.max((state_values - returns) ** 2,
                               (value_pred_clipped - returns) ** 2).mean()
        
        # Total loss with NaN checking
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning("NaN/Inf detected in loss, returning zero gradient")
            total_loss = torch.tensor(0.0, requires_grad=True, device=total_loss.device)
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "mean_advantage": advantages.mean().detach(),
            "mean_return": returns.mean().detach()
        }

class PPOModelWrapper(nn.Module):
    """Wrapper to make SB3 PPO model compatible with GRPO training"""
    
    def __init__(self, sb3_ppo_model):
        super().__init__()
        self.sb3_model = sb3_ppo_model
        self.policy = sb3_ppo_model.policy
        
    def forward(self, states):
        """Forward pass through policy network"""
        # Convert flat tensor back to dict format if needed
        if isinstance(states, torch.Tensor):
            # Handle both 1D (single obs) and 2D (batch) tensors
            if states.dim() == 1:
                states = states.unsqueeze(0)  # Add batch dimension
            
            # Get the device from the model
            device = next(self.policy.parameters()).device
            
            # Ensure states are on the correct device
            if states.device != device:
                states = states.to(device)
            
            # Reconstruct observation dict from flat tensor using config
            batch_size = states.shape[0]
            obs_dict = {}
            idx = 0
            
            # Extract components based on configuration
            for key, dim in OBSERVATION_COMPONENTS.items():
                obs_dict[key] = states[:, idx:idx+dim]
                idx += dim
            
            states = obs_dict
        
        features = self.policy.extract_features(states)
        
        if hasattr(self.policy, 'mlp_extractor'):
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
            action_logits = self.policy.action_net(latent_pi)
            values = self.policy.value_net(latent_vf)
        else:
            action_logits = self.policy.action_net(features)
            values = self.policy.value_net(features)
        
        return action_logits, values.squeeze(-1)
    
    def get_action_and_value(self, obs):
        """Get action, value, and log prob for given observation"""
        with torch.no_grad():
            # Convert flat tensor to dict for SB3 policy
            if isinstance(obs, torch.Tensor):
                # Convert to numpy dict for SB3's predict method
                obs_np = obs.cpu().numpy()
                if obs_np.ndim == 1:
                    obs_np = obs_np.reshape(1, -1)
                
                obs_dict_np = {}
                idx = 0
                
                # Extract components using configuration
                for key, dim in OBSERVATION_COMPONENTS.items():
                    obs_dict_np[key] = obs_np[:, idx:idx+dim]
                    idx += dim
                
                # Ensure all arrays have batch dimension
                for key in obs_dict_np:
                    if obs_dict_np[key].ndim == 1:
                        obs_dict_np[key] = obs_dict_np[key].reshape(1, -1)
                
                # Get action using SB3's predict
                actions, _ = self.sb3_model.predict(obs_dict_np, deterministic=False)
                
                # Convert dict back to torch for value and log_prob calculation
                obs_dict_torch = {k: torch.tensor(v, device=obs.device, dtype=torch.float32) 
                                 for k, v in obs_dict_np.items()}
            else:
                # Already in correct format
                actions, _ = self.sb3_model.predict(obs, deterministic=False)
                obs_dict_torch = obs
            
            # Get values and log probs
            features = self.policy.extract_features(obs_dict_torch)
            
            if hasattr(self.policy, 'mlp_extractor'):
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                action_logits = self.policy.action_net(latent_pi)
                values = self.policy.value_net(latent_vf)
            else:
                action_logits = self.policy.action_net(features)
                values = self.policy.value_net(features)
            
            # Calculate log prob
            dist = torch.distributions.Categorical(logits=action_logits)
            
            if isinstance(actions, np.ndarray):
                actions_tensor = torch.tensor(actions, device=values.device, dtype=torch.long)
            else:
                actions_tensor = actions
                
            log_prob = dist.log_prob(actions_tensor.squeeze())
        
        return actions_tensor, values.squeeze(-1), log_prob
    
    def parameters(self):
        """Return policy parameters for optimization"""
        return self.policy.parameters()
    
    def state_dict(self):
        """Return state dict of the policy"""
        return self.policy.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the policy"""
        self.policy.load_state_dict(state_dict)
    
    def eval(self):
        """Set policy to eval mode"""
        self.policy.eval()
        return self
    
    def train(self):
        """Set policy to train mode"""
        self.policy.train()
        return self

class Phase3GRPOTrainer:
    """Phase 3 GRPO fine-tuning trainer"""
    
    def __init__(self,
                 ppo_model,  # Can be SB3 PPO or nn.Module
                 train_env: DeepfakeEnv,
                 val_env: DeepfakeEnv,
                 device: str = "cuda",
                 learning_rate: float = 5e-5,
                 group_size: int = 8,
                 buffer_size: int = 10000,
                 gradient_clip: float = 0.5):
        
        self.device = torch.device(device)
        
        # Wrap SB3 model if needed
        if hasattr(ppo_model, 'policy'):  # SB3 PPO model
            self.ppo_model = PPOModelWrapper(ppo_model).to(self.device)
        else:  # Already a nn.Module
            self.ppo_model = ppo_model.to(self.device)
        
        self.train_env = train_env
        self.val_env = val_env
        
        # GRPO components
        self.buffer = GRPOBuffer(buffer_size=buffer_size, group_size=group_size)
        self.grpo_loss = GRPOLoss(
            ppo_model=self.ppo_model,
            clip_epsilon=PPO_CLIP_RANGE,
            entropy_coef=PPO_ENT_COEF * 1.5,  
            value_coef=PPO_VF_COEF,
            group_size=group_size
        )
        
        # Optimizer with lower learning rate for fine-tuning
        self.optimizer = AdamW(self.ppo_model.parameters(), 
                              lr=learning_rate,
                              weight_decay=1e-5)  # Reduced weight decay
        
        # Linear warmup then cosine annealing scheduler
        self.warmup_iterations = 5
        self.base_lr = learning_rate
        self.current_iteration = 0
        
        # Start with lower LR for warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.1
        
        # Cosine annealing scheduler (will be used after warmup)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-7)
        
        # Gradient clipping value
        self.gradient_clip = gradient_clip
        
        # Metrics tracking
        self.training_metrics = {
            "episode_rewards": deque(maxlen=100),
            "episode_lengths": deque(maxlen=100),
            "policy_losses": deque(maxlen=100),
            "value_losses": deque(maxlen=100),
            "entropies": deque(maxlen=100),
            "gradient_norms": deque(maxlen=100)
        }
        
        logging.info(f"Initialized Phase 3 GRPO Trainer")
        logging.info(f"  Learning Rate: {learning_rate}")
        logging.info(f"  Group Size: {group_size}")
        logging.info(f"  Buffer Size: {buffer_size}")
        logging.info(f"  Gradient Clip: {gradient_clip}")
    
    def collect_trajectories(self, num_episodes: int) -> List[Dict]:
        """Collect trajectories for GRPO training"""
        trajectories = []
        
        for ep_idx in range(num_episodes):
            obs_info = self.train_env.reset()
            # Handle both tuple return (obs, info) and single obs return
            if isinstance(obs_info, tuple):
                obs, _ = obs_info
            else:
                obs = obs_info
            
            # Extract and flatten observation from dict if needed
            if isinstance(obs, dict):
                # The observation dict contains: frame_features, temporal_memory, frame_position, uncertainty
                # We need to concatenate them into a flat array matching STATE_DIM
                obs_components = []
                if 'frame_features' in obs:
                    obs_components.append(obs['frame_features'].flatten())
                if 'temporal_memory' in obs:
                    obs_components.append(obs['temporal_memory'].flatten())
                if 'frame_position' in obs:
                    obs_components.append(obs['frame_position'].flatten())
                if 'uncertainty' in obs:
                    obs_components.append(obs['uncertainty'].flatten())
                
                if obs_components:
                    obs_array = np.concatenate(obs_components)
                else:
                    raise ValueError(f"Could not extract observation components from dict: {obs.keys()}")
            else:
                obs_array = obs
            
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
                "values": [],
                "dones": []
            }
            
            done = False
            while not done:
                # Convert observation to tensor
                if isinstance(obs_array, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor(np.array(obs_array)).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, value, log_prob = self.ppo_model.get_action_and_value(obs_tensor)
                
                trajectory["states"].append(obs_array)
                trajectory["actions"].append(action.cpu().numpy())
                trajectory["log_probs"].append(log_prob.cpu().numpy())
                trajectory["values"].append(value.cpu().numpy())
                
                # Step environment
                step_result = self.train_env.step(action.cpu().numpy()[0])
                
                # Handle different return formats (4 or 5 values)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                # Extract and flatten observation from dict if needed
                if isinstance(obs, dict):
                    obs_components = []
                    if 'frame_features' in obs:
                        obs_components.append(obs['frame_features'].flatten())
                    if 'temporal_memory' in obs:
                        obs_components.append(obs['temporal_memory'].flatten())
                    if 'frame_position' in obs:
                        obs_components.append(obs['frame_position'].flatten())
                    if 'uncertainty' in obs:
                        obs_components.append(obs['uncertainty'].flatten())
                    
                    if obs_components:
                        obs_array = np.concatenate(obs_components)
                    else:
                        obs_array = obs.get('observation', obs)  # Fallback
                else:
                    obs_array = obs
                
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
            
            # Convert to tensors with dimension checking
            for key in trajectory:
                if key == "states":
                    # Stack states to create 2D tensor [num_timesteps, feature_dim]
                    states_array = np.stack(trajectory[key])
                    trajectory[key] = torch.tensor(states_array, dtype=torch.float32)
                    # Debug: check dimensions
                    if trajectory[key].dim() != 2 or trajectory[key].shape[1] != STATE_DIM:
                        logging.error(f"States tensor has wrong shape: {trajectory[key].shape}, expected [?, {STATE_DIM}]")
                else:
                    trajectory[key] = torch.tensor(np.array(trajectory[key]), 
                                                  dtype=torch.float32)
            
            # Calculate episode metrics
            episode_reward = float(torch.sum(trajectory["rewards"]))
            episode_length = len(trajectory["rewards"])
            
            # Debug logging
            logging.info(f"Episode {ep_idx+1}: states shape={trajectory['states'].shape}, "
                        f"rewards shape={trajectory['rewards'].shape}")
            
            # Track metrics
            self.training_metrics["episode_rewards"].append(episode_reward)
            self.training_metrics["episode_lengths"].append(episode_length)
            
            # Log every 4th episode for visibility
            if (ep_idx + 1) % 4 == 0:
                recent_rewards = list(self.training_metrics["episode_rewards"])[-4:]
                avg_recent = np.mean(recent_rewards)
                logging.info(f"  Episodes {ep_idx-2}-{ep_idx+1}: "
                           f"Rewards {[f'{r:+.2f}' for r in recent_rewards]}, "
                           f"Avg: {avg_recent:+.3f}")
            
            trajectories.append(trajectory)
            self.buffer.add(trajectory)
        
        return trajectories
    
    def train_step(self, batch: TensorDict) -> Dict[str, float]:
        """Single GRPO training step with gradient safeguards"""
        self.optimizer.zero_grad()
        
        # Forward pass
        losses = self.grpo_loss(batch)
        total_loss = losses["total_loss"]
        
        # Backward with gradient clipping
        total_loss.backward()
        
        # Gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ppo_model.parameters(), 
            self.gradient_clip
        )
        
        # Check for NaN gradients
        has_nan = False
        for param in self.ppo_model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    break
        
        if has_nan:
            logging.warning("NaN/Inf gradient detected, skipping update")
            self.optimizer.zero_grad()
            return {k: 0.0 for k in losses.keys()}
        
        # Optimizer step
        self.optimizer.step()
        
        # Track metrics
        self.training_metrics["policy_losses"].append(losses["policy_loss"].item())
        self.training_metrics["value_losses"].append(losses["value_loss"].item())
        self.training_metrics["entropies"].append(losses["entropy"].item())
        self.training_metrics["gradient_norms"].append(grad_norm.item())
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy using direct PyTorch inference without SB3"""
        self.ppo_model.eval()
        
        total_rewards = []
        total_lengths = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs_info = self.val_env.reset()
                # Handle both tuple return (obs, info) and single obs return
                if isinstance(obs_info, tuple):
                    obs, _ = obs_info
                else:
                    obs = obs_info
                
                # Extract and flatten observation from dict if needed
                if isinstance(obs, dict):
                    obs_components = []
                    if 'frame_features' in obs:
                        obs_components.append(obs['frame_features'].flatten())
                    if 'temporal_memory' in obs:
                        obs_components.append(obs['temporal_memory'].flatten())
                    if 'frame_position' in obs:
                        obs_components.append(obs['frame_position'].flatten())
                    if 'uncertainty' in obs:
                        obs_components.append(obs['uncertainty'].flatten())
                    
                    if obs_components:
                        obs_array = np.concatenate(obs_components)
                    else:
                        obs_array = obs.get('observation', obs)  
                else:
                    obs_array = obs
                
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    # Convert observation to tensor
                    if isinstance(obs_array, np.ndarray):
                        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
                    else:
                        obs_tensor = torch.FloatTensor(np.array(obs_array)).to(self.device)
                    
                    # Ensure proper batch dimension
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    # Direct forward pass through the model (bypassing SB3 conversion)
                    # Use the underlying policy network directly
                    if hasattr(self.ppo_model, 'forward'):
                        # If it's our wrapper, use forward directly
                        action_logits, values = self.ppo_model.forward(obs_tensor)
                    else:
                        # For raw network, just get logits
                        action_logits = self.ppo_model(obs_tensor)
                        values = torch.zeros(1)  # Placeholder if no value head
                    
                    # Sample action from distribution
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    
                    # Step environment
                    step_result = self.val_env.step(action.cpu().numpy()[0])
                    
                    # Handle different return formats
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    
                    # Extract observation array for next iteration
                    if isinstance(obs, dict):
                        # Same extraction as at the beginning of episode
                        obs_components = []
                        if 'frame_features' in obs:
                            obs_components.append(obs['frame_features'].flatten())
                        if 'temporal_memory' in obs:
                            obs_components.append(obs['temporal_memory'].flatten())
                        if 'frame_position' in obs:
                            obs_components.append(obs['frame_position'].flatten())
                        if 'uncertainty' in obs:
                            obs_components.append(obs['uncertainty'].flatten())
                        
                        if obs_components:
                            obs_array = np.concatenate(obs_components)
                        else:
                            obs_array = obs.get('observation', obs)  # Fallback
                    else:
                        obs_array = obs
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Check if episode ended with a decision
                    if done:
                        # The environment provides final_prediction and true_label
                        if "final_prediction" in info and "true_label" in info:
                            if info["final_prediction"] is not None:
                                total_predictions += 1
                                if info["final_prediction"] == info["true_label"]:
                                    correct_predictions += 1
                        # Also check for the older 'correct' field for compatibility
                        elif "correct" in info:
                            total_predictions += 1
                            if info["correct"]:
                                correct_predictions += 1
                
                total_rewards.append(episode_reward)
                total_lengths.append(episode_length)
        
        self.ppo_model.train()
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_length": np.mean(total_lengths),
            "accuracy": accuracy
        }
    
    def train(self, 
             num_iterations: int = 100,
             episodes_per_iteration: int = 16,
             updates_per_iteration: int = 10,
             eval_freq: int = 10,
             save_freq: int = 20,
             checkpoint_dir: Optional[Path] = None) -> nn.Module:
        """Main GRPO training loop"""
        
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR / "phase3_grpo"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Starting Phase 3 GRPO Fine-tuning")
        logging.info(f"  Iterations: {num_iterations}")
        logging.info(f"  Episodes per iteration: {episodes_per_iteration}")
        logging.info(f"  Updates per iteration: {updates_per_iteration}")
        
        best_reward = -float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 10  # Stop if no improvement for 10 iterations
        baseline_reward = None  # Track initial performance
        
        for iteration in tqdm(range(num_iterations), desc="GRPO Training"):
            # Collect new trajectories
            trajectories = self.collect_trajectories(episodes_per_iteration)
            
            # Sample groups for GRPO updates
            groups = self.buffer.sample_groups(num_groups=updates_per_iteration)
            
            if not groups:
                logging.info(f"Iteration {iteration}: Buffer size {len(self.buffer)}, "
                           f"need {self.grpo_loss.group_size * updates_per_iteration} for full update")
                continue
            
            if len(groups) < updates_per_iteration:
                logging.info(f"Iteration {iteration}: Performing {len(groups)} updates "
                           f"instead of {updates_per_iteration} (buffer building up)")
            
            # Perform GRPO updates
            iteration_losses = []
            for update_idx, group in enumerate(groups):
                logging.info(f"  Starting update {update_idx + 1}/{len(groups)}...")
                # Prepare batch from group
                logging.info(f"    Preparing batch from {len(group)} trajectories...")
                batch = self.prepare_batch(group)
                logging.info(f"    Batch prepared: {batch.batch_size[0]} timesteps")
                logging.info(f"    Running training step...")
                losses = self.train_step(batch)
                logging.info(f"    Update {update_idx + 1} complete: policy_loss={losses.get('policy_loss', 0):.4f}")
                iteration_losses.append(losses)
            
            # Learning rate scheduling with warmup
            self.current_iteration = iteration
            if iteration < self.warmup_iterations:
                # Linear warmup
                warmup_factor = (iteration + 1) / self.warmup_iterations
                current_lr = self.base_lr * (0.1 + 0.9 * warmup_factor)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                logging.info(f"Warmup: Setting LR to {current_lr:.2e}")
            else:
                # Use cosine annealing after warmup
                self.scheduler.step()
            
            # Calculate iteration metrics
            if iteration_losses:
                avg_losses = {
                    k: np.mean([loss[k] for loss in iteration_losses])
                    for k in iteration_losses[0].keys()
                }
                
                # Calculate rolling averages for smoother display
                recent_rewards = list(self.training_metrics["episode_rewards"])[-20:]
                recent_lengths = list(self.training_metrics["episode_lengths"])[-20:]
                
                # Update progress bar with detailed metrics
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_length = np.mean(recent_lengths) if recent_lengths else 0
                
                # Color coding for terminal output
                reward_color = "\033[92m" if avg_reward > 0 else "\033[91m"  # Green if positive, red if negative
                reset_color = "\033[0m"
                
                # Detailed logging every iteration
                logging.info(
                    f"\n{'='*80}\n"
                    f"📊 Iteration {iteration}/{num_iterations} | "
                    f"Buffer: {len(self.buffer)} trajectories\n"
                    f"{'='*80}\n"
                    f"📈 Performance Metrics:\n"
                    f"   • Avg Reward: {reward_color}{avg_reward:+.4f}{reset_color} "
                    f"(last 20 episodes)\n"
                    f"   • Avg Episode Length: {avg_length:.1f} frames\n"
                    f"   • Total Episodes Collected: {len(self.training_metrics['episode_rewards'])}\n"
                    f"\n📉 Loss Metrics:\n"
                    f"   • Policy Loss: {avg_losses['policy_loss']:.4f}\n"
                    f"   • Value Loss: {avg_losses['value_loss']:.4f}\n"
                    f"   • Entropy: {avg_losses['entropy']:.4f} "
                    f"({'good' if avg_losses['entropy'] > 0.01 else 'low exploration'})\n"
                    f"   • Mean Advantage: {avg_losses['mean_advantage']:+.4f}\n"
                    f"   • Mean Return: {avg_losses['mean_return']:.4f}\n"
                    f"\n🔧 Training Stats:\n"
                    f"   • Gradient Norm: {np.mean(list(self.training_metrics['gradient_norms'])[-10:]):.4f}\n"
                    f"   • Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
                    f"   • Updates Performed: {len(groups)}/{updates_per_iteration}\n"
                    f"{'='*80}\n"
                )
            
            # Track baseline performance
            if iteration == 0 and recent_rewards:
                baseline_reward = np.mean(recent_rewards)
                logging.info(f"📊 Baseline reward: {baseline_reward:.4f}")
            
            # Evaluation
            if iteration % eval_freq == 0 and iteration > 0:
                logging.info("\n🔍 Running evaluation...")
                eval_metrics = self.evaluate()
                
                # Early stopping check
                current_reward = eval_metrics['mean_reward']
                if baseline_reward is not None:
                    if current_reward < baseline_reward * 0.95:  # 5% degradation threshold
                        early_stopping_counter += 1
                        logging.warning(f"⚠️ Performance degraded! Counter: {early_stopping_counter}/{early_stopping_patience}")
                        if early_stopping_counter >= early_stopping_patience:
                            logging.warning(f"🛑 Early stopping triggered! Performance degraded for {early_stopping_patience} iterations")
                            break
                    else:
                        early_stopping_counter = 0  # Reset counter if performance is good
                
                # Determine if this is the best model
                is_best = eval_metrics['mean_reward'] > best_reward
                best_indicator = " 🏆 NEW BEST!" if is_best else ""
                
                logging.info(
                    f"\n{'='*80}\n"
                    f"✨ EVALUATION RESULTS (Iteration {iteration}){best_indicator}\n"
                    f"{'='*80}\n"
                    f"   • Mean Reward: {eval_metrics['mean_reward']:+.4f} "
                    f"(σ={eval_metrics['std_reward']:.4f})\n"
                    f"   • Detection Accuracy: {eval_metrics['accuracy']*100:.2f}%\n"
                    f"   • Mean Episode Length: {eval_metrics['mean_length']:.1f} frames\n"
                    f"   • Previous Best: {best_reward:+.4f}\n"
                    f"{'='*80}\n"
                )
                
                # Save best model
                if eval_metrics['mean_reward'] > best_reward:
                    best_reward = eval_metrics['mean_reward']
                    self.save_checkpoint(checkpoint_dir / "best_model.pth", 
                                       iteration, eval_metrics)
            
            # Regular checkpoint
            if iteration % save_freq == 0:
                self.save_checkpoint(checkpoint_dir / f"checkpoint_{iteration}.pth",
                                   iteration, {})
        
        # Final save
        self.save_checkpoint(checkpoint_dir / "final_model.pth", 
                           num_iterations, {})
        
        # Final evaluation
        logging.info("\n🎯 Running final evaluation...")
        final_metrics = self.evaluate(num_episodes=20)
        
        # Training summary
        all_rewards = list(self.training_metrics["episode_rewards"])
        all_lengths = list(self.training_metrics["episode_lengths"])
        
        logging.info(
            f"\n{'='*80}\n"
            f"🎉 GRPO TRAINING COMPLETED!\n"
            f"{'='*80}\n"
            f"📊 Final Statistics:\n"
            f"   • Total Episodes: {len(all_rewards)}\n"
            f"   • Best Reward Achieved: {best_reward:+.4f}\n"
            f"   • Final Reward: {final_metrics['mean_reward']:+.4f} "
            f"(σ={final_metrics['std_reward']:.4f})\n"
            f"   • Final Accuracy: {final_metrics['accuracy']*100:.2f}%\n"
            f"\n📈 Training Progress:\n"
            f"   • Starting Avg Reward: {np.mean(all_rewards[:20]) if len(all_rewards) >= 20 else 'N/A':+.4f}\n"
            f"   • Ending Avg Reward: {np.mean(all_rewards[-20:]):+.4f}\n"
            f"   • Improvement: {(np.mean(all_rewards[-20:]) - np.mean(all_rewards[:20])):+.4f}\n"
            f"\n💾 Model Checkpoints:\n"
            f"   • Best Model: {checkpoint_dir / 'best_model.pth'}\n"
            f"   • Final Model: {checkpoint_dir / 'final_model.pth'}\n"
            f"{'='*80}\n"
        )
        
        return self.ppo_model
    
    def prepare_batch(self, group: List[Dict]) -> TensorDict:
        """Prepare batch from trajectory group with GRPO advantages"""
        # Compute trajectory-level advantages for this group
        normalized_advantages = self.grpo_loss.compute_relative_advantages(group)
        
        concatenated_data = {}
        
        # Debug: log trajectory shapes
        total_timesteps = sum(len(traj['rewards']) for traj in group)
        logging.info(f"DEBUG: Preparing batch from {len(group)} trajectories, total timesteps: {total_timesteps}")
        
        # Concatenate all trajectories along the time dimension
        for key in group[0].keys():
            # Collect all data for this key across trajectories
            all_data = []
            for i, traj in enumerate(group):
                data = traj[key]
                
                # Handle states specially - they should be 2D [timesteps, features]
                if key == 'states':
                    # States should already be 2D from collect_trajectories: [num_timesteps, feature_dim]
                    if data.dim() != 2:
                        logging.error(f"Trajectory {i}: States tensor has unexpected dimension: {data.dim()}, shape: {data.shape}")
                        if data.dim() == 1:
                            # Try to recover
                            data = data.view(-1, STATE_DIM)
                        elif data.dim() == 3:
                            data = data.view(-1, data.shape[-1])
                    # Keep as 2D for states
                else:
                    # For other keys (actions, rewards, etc.), ensure 1D
                    data = data.flatten()
                
                all_data.append(data)
            
            # Concatenate along time dimension
            # For states (2D), concatenate along dim=0 to get [total_timesteps, feature_dim]
            # For others (1D), concatenate along dim=0 to get [total_timesteps]
            concatenated_data[key] = torch.cat(all_data, dim=0)
            
            # Log the shape for debugging
            if key == 'states':
                logging.info(f"DEBUG: Concatenated {key}: shape {concatenated_data[key].shape}")
        
        # Add the normalized advantages - these should be 1D
        normalized_advantages_flat = []
        for i, adv in enumerate(normalized_advantages):
            adv_flat = adv.flatten()
            logging.info(f"DEBUG: Normalized advantage {i} shape: {adv_flat.shape}")
            normalized_advantages_flat.append(adv_flat)
        concatenated_data["advantages"] = torch.cat(normalized_advantages_flat, dim=0)
        logging.info(f"DEBUG: Concatenated advantages shape: {concatenated_data['advantages'].shape}")
        
        if 'returns' in group[0]:
            all_returns = []
            for i, traj in enumerate(group):
                ret = traj['returns']
                ret_flat = ret.flatten()
                logging.info(f"DEBUG: Returns {i} shape: {ret_flat.shape}")
                all_returns.append(ret_flat)
            concatenated_data["returns"] = torch.cat(all_returns, dim=0)
            logging.info(f"DEBUG: Concatenated returns shape: {concatenated_data['returns'].shape}")
        
        # Move to device and log shapes
        batch_data = {}
        for k, v in concatenated_data.items():
            batch_data[k] = v.to(self.device)
            logging.info(f"DEBUG: batch_data['{k}'] shape: {batch_data[k].shape}")
        
        # Validate states shape - should be 2D: [total_timesteps, feature_dim]
        states = batch_data["states"]
        if states.dim() != 2:
            logging.error(f"Unexpected states dimension: {states.dim()}, shape: {states.shape}")
            if states.dim() == 1:
                # Try to recover by reshaping
                batch_data["states"] = states.view(-1, STATE_DIM)
                logging.warning(f"Reshaped states from {states.shape} to {batch_data['states'].shape}")
        elif states.shape[1] != STATE_DIM:
            logging.error(f"States feature dimension {states.shape[1]} doesn't match STATE_DIM {STATE_DIM}")
        
        # Create TensorDict with total number of timesteps
        total_timesteps = batch_data["states"].shape[0]
        logging.debug(f"Batch prepared with {total_timesteps} timesteps, states shape: {batch_data['states'].shape}")
        return TensorDict(batch_data, batch_size=[total_timesteps])
    
    def save_checkpoint(self, path: Path, iteration: int, metrics: Dict):
        """Save training checkpoint"""
        # Save the actual SB3 model state for compatibility
        if isinstance(self.ppo_model, PPOModelWrapper):
            # If it's our wrapper, save the underlying SB3 policy state
            model_state = self.ppo_model.policy.state_dict()
            model_type = "sb3_policy"
        else:
            # Otherwise save as is
            model_state = self.ppo_model.state_dict()
            model_type = "pytorch"
        
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": model_state,
            "model_type": model_type,  # Track what type of model this is
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "training_metrics": {
                k: list(v) for k, v in self.training_metrics.items()
            }
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path} (type: {model_type})")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint with compatibility handling"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different model types
        model_type = checkpoint.get("model_type", "pytorch")
        
        if model_type == "sb3_policy" and isinstance(self.ppo_model, PPOModelWrapper):
            # Loading SB3 policy state into wrapper
            self.ppo_model.policy.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(self.ppo_model, PPOModelWrapper):
            # Try to load into the wrapper's policy
            try:
                self.ppo_model.policy.load_state_dict(checkpoint["model_state_dict"])
            except:
                # Fallback to loading the whole wrapper
                self.ppo_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Direct PyTorch model
            self.ppo_model.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore metrics
        for k, v in checkpoint.get("training_metrics", {}).items():
            self.training_metrics[k] = deque(v, maxlen=100)
        
        logging.info(f"Checkpoint loaded from {path} (type: {model_type})")
        return checkpoint.get("iteration", 0)