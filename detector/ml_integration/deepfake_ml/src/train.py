"""
Multi-phase training pipeline for EAGER algorithm.
Implements supervised warm-start and PPO-LSTM reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AUTOCAST_AVAILABLE = True
    USE_NEW_AUTOCAST = True
except ImportError:
    try:
        
        from torch.cuda.amp import autocast, GradScaler
        AUTOCAST_AVAILABLE = True
        USE_NEW_AUTOCAST = False
    except ImportError:
       
        AUTOCAST_AVAILABLE = False
        USE_NEW_AUTOCAST = False
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
import json
import time
from datetime import datetime
from collections import deque

from stable_baselines3 import PPO
# Import necessary VecEnvs and seeding utility
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Import project modules from local source directory
from src.config import *
from src.data_loader import ProcessedVideoDataset, create_data_loaders
from src.feature_extractor import (
    VisionBackbone, TemporalMemory, ClassifierHead, 
    FrozenEvaluator, FeatureExtractorModule
)
from src.deepfake_env import DeepfakeEnv
from src.reward_system import RewardCalculator
from src.agent import EagerAgent, EagerActorCriticPolicy
from src.evaluation_metrics import PhaseEvaluator, RLEvaluator

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter for hard examples
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to batch.
    
    Args:
        x: Input batch (batch_size, num_frames, channels, height, width)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Augmented batch
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        return x, y, y, 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions
    if x.dim() == 5:  # Video data: (batch, frames, channels, height, width)
        _, num_frames, _, H, W = x.shape
    else:  # Image data: (batch, channels, height, width)
        _, _, H, W = x.shape
        num_frames = None
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    if num_frames is not None:
        # Apply to all frames in video
        mixed_x[:, :, :, bby1:bby2, bbx1:bbx2] = x[index, :, :, bby1:bby2, bbx1:bbx2]
    else:
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Phase1WarmStartTrainer:
    """
    Phase 1: Supervised warm-start training.
    Establishes baseline feature extraction and classification capabilities.
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractorModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = WARMSTART_LR,
        device: str = DEVICE
    ):
        """
        Initialize warm-start trainer.
        
        Args:
            feature_extractor: Complete feature extraction module
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Computing device
        """
        self.feature_extractor = feature_extractor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate  # Store for checkpoint saving
        
        # Optimizer with weight decay for regularization
        from src.config import (WARMSTART_WEIGHT_DECAY, WARMSTART_SCHEDULER_PATIENCE, 
                           WARMSTART_SCHEDULER_FACTOR, USE_FOCAL_LOSS, FOCAL_ALPHA, FOCAL_GAMMA,
                           USE_COSINE_ANNEALING, USE_REDUCE_ON_PLATEAU, USE_ADAMW, WARMSTART_MIN_LR,
                           WARMSTART_GRADIENT_ACCUMULATION, WARMSTART_EARLY_STOPPING_PATIENCE)
        
        # Use AdamW if configured, otherwise standard Adam
        if USE_ADAMW:
            self.optimizer = torch.optim.AdamW(
                self.feature_extractor.parameters(),
                lr=learning_rate,
                weight_decay=WARMSTART_WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=learning_rate,
                weight_decay=WARMSTART_WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Learning rate scheduler - Cosine Annealing with Warm Restarts or ReduceLROnPlateau
        if USE_COSINE_ANNEALING:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=COSINE_T0,  # Initial restart period
                T_mult=COSINE_T_MULT,  # Period multiplier after restart
                eta_min=WARMSTART_MIN_LR  # Minimum learning rate
            )
            self.scheduler_step_per_batch = True  # Step scheduler per batch
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Monitor validation accuracy
                factor=WARMSTART_SCHEDULER_FACTOR,
                patience=WARMSTART_SCHEDULER_PATIENCE,
                min_lr=WARMSTART_MIN_LR
            )
            self.scheduler_step_per_batch = False
        
        # Loss function - Hybrid loss strategy (CrossEntropy -> Focal Loss)
        from src.config import WARMSTART_LABEL_SMOOTHING
        
        # Store both loss functions for hybrid switching
        self.ce_criterion = nn.CrossEntropyLoss(label_smoothing=WARMSTART_LABEL_SMOOTHING)
        self.focal_criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        
        # Start with CrossEntropy, switch to Focal after epoch 10
        self.hybrid_loss_switch_epoch = 10
        self.criterion = self.ce_criterion  # Start with CE loss
        
        # Gradient accumulation settings
        self.gradient_accumulation_steps = WARMSTART_GRADIENT_ACCUMULATION
        self.effective_batch_size = self.train_loader.batch_size * self.gradient_accumulation_steps
        
        # Mixed precision training (FP16) for RTX 5090 optimization
        self.use_mixed_precision = USE_MIXED_PRECISION and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Mixed precision (FP16) training enabled for RTX 5090 optimization")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
        # Early stopping parameters from config
        self.early_stopping_patience = WARMSTART_EARLY_STOPPING_PATIENCE
        self.early_stopping_min_delta = 0.001  # 0.1% improvement threshold for accuracy
        self.epochs_without_improvement = 0
       
        
        # Initialize evaluator for Phase 1
        self.evaluator = PhaseEvaluator(
            phase_name="phase1_warmstart",
            save_dir=PROJECT_ROOT / "logs"
        )
        
        logger.info("Initialized Phase 1 Warm-Start Trainer")
        
        # Model architecture verification
        self._verify_model_architecture()
    
    def _verify_model_architecture(self):
        """Verify model architecture and print parameter counts."""
        logger.info("="*80)
        logger.info("MODEL ARCHITECTURE VERIFICATION")
        logger.info("="*80)
        
        total_params = 0
        trainable_params = 0
        
        # Count parameters by component
        components = {
            'Vision Backbone': self.feature_extractor.vision_backbone,
            'Temporal Memory (LSTM)': self.feature_extractor.temporal_memory,
            'Classifier Head': self.feature_extractor.classifier_head
        }
        
        if hasattr(self.feature_extractor, 'attention_pooling') and self.feature_extractor.attention_pooling:
            components['Attention Pooling'] = self.feature_extractor.attention_pooling
        
        for name, module in components.items():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += params
            trainable_params += trainable
            logger.info(f"{name:25s}: {params:,} params ({trainable:,} trainable)")
        
        logger.info(f"{'Total':25s}: {total_params:,} params ({trainable_params:,} trainable)")
        
        # Verify dimensions
        dummy_input = torch.randn(1, 50, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.feature_extractor(dummy_input)
        
        logger.info("\nDimension Flow Verification:")
        logger.info(f"  Input: {dummy_input.shape}")
        logger.info(f"  Output logits: {output['logits'].shape}")
        logger.info(f"  Output probs: {output['probs'].shape}")
        logger.info(f"  Temporal memory: {output['temporal_memory'].shape}")
        logger.info("="*80)
    
    def calculate_ece(self, confidences, predictions, labels, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def _mixup_data(self, x, y, alpha=0.2):
        """Apply MixUp augmentation to batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Calculate MixUp loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train_epoch(self, epoch: int = 0) -> Tuple[float, float]:
        """
        Train for one epoch with CutMix augmentation and gradient accumulation.
        
        Args:
            epoch: Current epoch number (for warmup)
            
        Returns:
            Average training loss and accuracy
        """
        self.feature_extractor.train()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        batch_losses = []
        accumulated_loss = 0.0
        
        # Debug: Track training predictions
        train_pred_real = 0
        train_pred_fake = 0
        train_real_labels = 0
        train_fake_labels = 0
        
        from src.config import WARMSTART_CUTMIX_ALPHA, WARMUP_EPOCHS, WARMUP_FACTOR
        use_cutmix = WARMSTART_CUTMIX_ALPHA > 0
        
        # Learning rate warmup
        if epoch < WARMUP_EPOCHS:
            warmup_factor = WARMUP_FACTOR + (1.0 - WARMUP_FACTOR) * (epoch / WARMUP_EPOCHS)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
        
        progress_bar = tqdm(self.train_loader, desc="Training", ncols=120)
        for batch_idx, (frames, labels, video_ids) in enumerate(progress_bar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Apply CutMix augmentation if enabled (only during training)
            if use_cutmix and np.random.rand() < 0.5:  # Apply CutMix 50% of the time
                frames, labels_a, labels_b, lam = cutmix_data(frames, labels, WARMSTART_CUTMIX_ALPHA)
                cutmix_applied = True
            else:
                cutmix_applied = False
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                if USE_NEW_AUTOCAST:
                    
                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.feature_extractor(frames)
                        
                        # Calculate loss (with CutMix if applicable)
                        if cutmix_applied:
                            loss = lam * self.criterion(outputs['logits'], labels_a) + (1 - lam) * self.criterion(outputs['logits'], labels_b)
                        else:
                            loss = self.criterion(outputs['logits'], labels)
                else:
                   
                    with autocast(enabled=True):
                        outputs = self.feature_extractor(frames)
                        
                        # Calculate loss (with CutMix if applicable)
                        if cutmix_applied:
                            loss = lam * self.criterion(outputs['logits'], labels_a) + (1 - lam) * self.criterion(outputs['logits'], labels_b)
                        else:
                            loss = self.criterion(outputs['logits'], labels)
                    
                    # Don't normalize loss for backpropagation - only for logging
            else:
                outputs = self.feature_extractor(frames)
                
                # Calculate loss (with CutMix if applicable)
                if cutmix_applied:
                    loss = lam * self.criterion(outputs['logits'], labels_a) + (1 - lam) * self.criterion(outputs['logits'], labels_b)
                else:
                    loss = self.criterion(outputs['logits'], labels)
                
                # Don't normalize loss for backpropagation - only for logging
            
            # Accumulate raw loss 
            accumulated_loss += loss.item()
            
            # Calculate accuracy (skip for CutMix batches to avoid misleading metrics)
            predictions = torch.argmax(outputs['logits'], dim=1)
            if not cutmix_applied:
                batch_correct = (predictions == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
                batch_acc = batch_correct / labels.size(0)
                
                # Debug: Track predictions and labels distribution
                train_pred_real += (predictions == 0).sum().item()
                train_pred_fake += (predictions == 1).sum().item()
                train_real_labels += (labels == 0).sum().item()
                train_fake_labels += (labels == 1).sum().item()
            else:
                # Skip accuracy calculation for CutMix batches
                batch_acc = 0.0  
            
            # Backward pass with gradient accumulation and mixed precision
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Perform optimizer step after gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.feature_extractor.parameters(),
                        PPO_MAX_GRAD_NORM
                    )
                    # Step with scaled gradients
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.feature_extractor.parameters(),
                        PPO_MAX_GRAD_NORM
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step scheduler if using cosine annealing
                if self.scheduler_step_per_batch:
                    self.scheduler.step()
                
                # Track accumulated loss 
                total_loss += accumulated_loss
                batch_losses.append(accumulated_loss / self.gradient_accumulation_steps) 
                num_batches += 1
                accumulated_loss = 0.0
            
            # Update progress bar with detailed info
            if total > 0:
                running_acc = correct / total
                avg_loss = total_loss / max(num_batches, 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'acc': f'{running_acc:.2%}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Print detailed info every 50 batches
            if (batch_idx + 1) % 50 == 0 and num_batches > 0:
                logger.info(f"  Batch {batch_idx+1}/{len(self.train_loader)}: "
                           f"Loss={avg_loss:.4f}, Acc={running_acc:.2%}, "
                           f"LR={current_lr:.2e}")
        
        # Handle any remaining gradients
        if accumulated_loss > 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.feature_extractor.parameters(),
                    PPO_MAX_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.feature_extractor.parameters(),
                    PPO_MAX_GRAD_NORM
                )
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average loss per gradient step (divide by total gradient steps)
        avg_loss = total_loss / (max(num_batches, 1) * self.gradient_accumulation_steps)
        train_acc = correct / total if total > 0 else 0
        logger.info(f"  Training epoch complete: Avg Loss={avg_loss:.4f}, Accuracy={train_acc:.2%}")
        
        # Debug: Show training distribution
        if total > 0:
            logger.info(f"  === TRAINING DEBUG ===")
            logger.info(f"  Training predictions: Real={train_pred_real}/{total} ({100*train_pred_real/total:.1f}%), Fake={train_pred_fake}/{total} ({100*train_pred_fake/total:.1f}%)")
            logger.info(f"  Training labels: Real={train_real_labels}/{total} ({100*train_real_labels/total:.1f}%), Fake={train_fake_labels}/{total} ({100*train_fake_labels/total:.1f}%)")
        
        return avg_loss, train_acc
    
    def validate(self, save_predictions: bool = False, epoch: int = 0) -> Tuple[float, float, float]:
        """
        Validate on validation set with ECE calculation.
        
        Args:
            save_predictions: Whether to save predictions
            epoch: Current epoch number
            
        Returns:
            validation_loss: Average validation loss
            accuracy: Validation accuracy
            ece: Expected Calibration Error
        """
        self.feature_extractor.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        # Debug: Track predictions distribution
        pred_real_count = 0
        pred_fake_count = 0
        batch_losses = []
        logits_per_class = {0: [], 1: []}
        
        # For ECE calculation
        all_confidences = []
        all_predictions = []
        all_labels = []
        
        # Reset evaluator for this validation run if saving predictions
        if save_predictions:
            self.evaluator.reset_metrics()
        
        progress_bar = tqdm(self.val_loader, desc="Validating", ncols=120)
        with torch.no_grad():
            for batch_idx, (frames, labels, video_ids) in enumerate(progress_bar):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.feature_extractor(frames)
                loss = self.criterion(outputs['logits'], labels)
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                
                # Debug: Track what model is predicting
                pred_real_count += (predictions == 0).sum().item()
                pred_fake_count += (predictions == 1).sum().item()
                batch_losses.append(loss.item())
                
                # Track logits per class
                for logit, label in zip(outputs['logits'], labels):
                    logits_per_class[label.item()].append(logit.cpu().detach())
                
                # Add to evaluator if saving predictions
                if save_predictions:
                    confidences = torch.max(probabilities, dim=1)[0]
                    self.evaluator.add_batch(
                        predictions=predictions,
                        labels=labels,
                        probabilities=probabilities,
                        video_ids=video_ids,
                        confidences=confidences
                    )
                batch_correct = (predictions == labels).sum().item()
                correct += batch_correct
                total += labels.size(0)
                
                # Per-class accuracy
                for pred, label in zip(predictions, labels):
                    if label == 0:  # Real
                        real_total += 1
                        if pred == label:
                            real_correct += 1
                    else:  # Fake
                        fake_total += 1
                        if pred == label:
                            fake_correct += 1
                
                total_loss += loss.item()
                
                # Store for ECE calculation
                confidences = torch.max(probabilities, dim=1)[0]
                all_confidences.append(confidences.cpu())
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                
                # Update progress bar
                running_acc = correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{running_acc:.2%}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
        
        # Calculate ECE
        all_confidences = torch.cat(all_confidences)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        ece = self.calculate_ece(all_confidences, all_predictions, all_labels)
        
        logger.info(f"  Validation complete: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}, ECE={ece:.4f}")
        logger.info(f"    Real videos: {real_accuracy:.2%} ({real_correct}/{real_total})")
        logger.info(f"    Fake videos: {fake_accuracy:.2%} ({fake_correct}/{fake_total})")
        
        # Debug: Comprehensive analysis
        logger.info(f"\n  === VALIDATION DEBUG ===")
        logger.info(f"  Model prediction distribution:")
        logger.info(f"    Predicted Real: {pred_real_count}/{total} ({100*pred_real_count/total:.1f}%)")
        logger.info(f"    Predicted Fake: {pred_fake_count}/{total} ({100*pred_fake_count/total:.1f}%)")
        logger.info(f"  Ground truth distribution:")
        logger.info(f"    Actual Real: {real_total}/{total} ({100*real_total/total:.1f}%)")
        logger.info(f"    Actual Fake: {fake_total}/{total} ({100*fake_total/total:.1f}%)")
        
        # Loss statistics
        if batch_losses:
            logger.info(f"  Loss statistics:")
            logger.info(f"    Mean: {np.mean(batch_losses):.4f}, Std: {np.std(batch_losses):.4f}")
            logger.info(f"    Min: {np.min(batch_losses):.4f}, Max: {np.max(batch_losses):.4f}")
        
        # Logits analysis
        if logits_per_class[0] and logits_per_class[1]:
            real_logits = torch.stack(logits_per_class[0])
            fake_logits = torch.stack(logits_per_class[1])
            logger.info(f"  Logits analysis:")
            logger.info(f"    Real class logits - Mean: {real_logits.mean(0).tolist()}, Std: {real_logits.std(0).tolist()}")
            logger.info(f"    Fake class logits - Mean: {fake_logits.mean(0).tolist()}, Std: {fake_logits.std(0).tolist()}")
        
        # Check for prediction bias
        prediction_bias = abs(pred_real_count - pred_fake_count) / total
        if prediction_bias > 0.15:
            logger.warning(f"  ⚠️ Model shows prediction bias: {prediction_bias:.1%} toward {'REAL' if pred_real_count > pred_fake_count else 'FAKE'} class")
            logger.warning(f"  ⚠️ Model predicts {pred_real_count/total:.1%} as REAL but only {real_accuracy:.1%} accuracy on real videos")
        
        # Generate confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0 and save_predictions:
            logger.info(f"  Generating confusion matrix for epoch {epoch + 1}")
            self.evaluator.generate_confusion_matrix()
        
        return avg_loss, accuracy, ece
    
    def train(self, num_epochs: int = WARMSTART_EPOCHS) -> FrozenEvaluator:
        """
        Run complete warm-start training.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Frozen evaluator for RL phase
        """
        # Store num_epochs as instance variable for checkpoint saving
        self.num_epochs = num_epochs
        
        logger.info("="*80)
        logger.info(f"Starting Phase 1: Supervised Warm-Start Training")
        logger.info(f"Training for {num_epochs} epochs with learning rate {WARMSTART_LR}")
        logger.info(f"Dataset: {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val")
        
        # Get and display detailed class distribution
        train_dist = self.train_loader.dataset.get_class_distribution()
        val_dist = self.val_loader.dataset.get_class_distribution()
        
        logger.info("\nTraining Set Distribution:")
        logger.info(f"  Real: {train_dist['real']} videos ({train_dist['real']*50:,} frames)")
        logger.info(f"  Fake: {train_dist['fake']} videos ({train_dist['fake']*50:,} frames)")
        logger.info(f"  Real/Fake Ratio: {train_dist['real']/train_dist['fake']:.2f}" if train_dist['fake'] > 0 else "N/A")
        
        logger.info("\nValidation Set Distribution:")
        logger.info(f"  Real: {val_dist['real']} videos ({val_dist['real']*50:,} frames)")
        logger.info(f"  Fake: {val_dist['fake']} videos ({val_dist['fake']*50:,} frames)")
        logger.info(f"  Real/Fake Ratio: {val_dist['real']/val_dist['fake']:.2f}" if val_dist['fake'] > 0 else "N/A")
        
        logger.info("="*80)
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE 1 TRAINING - EPOCH {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Force flush logs to file (Windows fix)
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            
            
            # Train
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            train_time = time.time() - start_time
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            start_time = time.time()
            # Save predictions on last epoch or every 5 epochs
            save_preds = (epoch == num_epochs - 1) or ((epoch + 1) % 5 == 0)
            val_loss, val_acc, val_ece = self.validate(save_predictions=save_preds, epoch=epoch)
            val_time = time.time() - start_time
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Step learning rate scheduler (only for ReduceLROnPlateau)
            if not self.scheduler_step_per_batch:
                self.scheduler.step(val_acc)
            
            # Summary
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f} (time: {train_time:.1f}s)")
            logger.info(f"  Val Loss:   {val_loss:.4f} (time: {val_time:.1f}s)")
            logger.info(f"  Val Acc:    {val_acc:.2%}")
            
            # Save checkpoint every 5 epochs for recovery
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"warmstart_epoch_{epoch+1}.pth", epoch=epoch+1)
                logger.info(f"  💾 Checkpoint saved: warmstart_epoch_{epoch+1}.pth")
            
            # Save best model and check early stopping based on ACCURACY
            if val_acc > self.best_val_acc + self.early_stopping_min_delta:
                self.best_val_acc = val_acc
                self.save_checkpoint(f"warmstart_best.pth", epoch=epoch+1)
                logger.info(f"  🎯 New best model! Accuracy improved to {val_acc:.2%}")
                # Reset early stopping counter
                self.epochs_without_improvement = 0
            else:
                logger.info(f"  Current best: {self.best_val_acc:.2%}")
                # Increment early stopping counter
                self.epochs_without_improvement += 1
                logger.info(f"  Epochs without improvement: {self.epochs_without_improvement}/{self.early_stopping_patience}")
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"\n⚠️  Early stopping triggered! No improvement for {self.early_stopping_patience} epochs")
                    logger.info(f"  Loading best model from checkpoint...")
                    # Load best model
                    checkpoint = torch.load(CHECKPOINT_DIR / "warmstart_best.pth")
                    self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                    break
        
        # Generate comprehensive evaluation report for Phase 1
        logger.info("\nGenerating Phase 1 evaluation report...")
        
        # Validate on final model and collect all predictions
        val_loss, val_acc, val_ece = self.validate(save_predictions=True)
        
        # Create metric history for plotting
        metric_history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }
        
        # Generate full evaluation report
        phase1_metrics = self.evaluator.generate_full_report(metric_history)
        logger.info(f"Phase 1 evaluation metrics: {phase1_metrics}")
        
        # Create frozen evaluator
        frozen_evaluator = FrozenEvaluator(self.feature_extractor.classifier_head)
        
        logger.info(f"Phase 1 complete! Best accuracy: {self.best_val_acc:.4f}")
        return frozen_evaluator
    
    def save_checkpoint(self, filename: str, epoch: Optional[int] = None):
        """Save model checkpoint with metadata."""
        checkpoint_path = CHECKPOINT_DIR / filename
        checkpoint_data = {
            'model_state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'epoch': epoch if epoch is not None else len(self.train_losses),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.train_loader.batch_size,
                'unfreeze_layers': UNFREEZE_VIT_LAYERS,
                'use_attention_pooling': USE_ATTENTION_POOLING
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path} (epoch {checkpoint_data['epoch']})")

# Standalone environment creation function for multiprocessing (required for SubprocVecEnv)
def make_env(rank: int, dataset, vision_backbone, temporal_memory, classifier_head, reward_calculator, device, training, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = DeepfakeEnv(
            dataset=dataset,
            vision_backbone=vision_backbone,
            temporal_memory=temporal_memory,
            classifier_head=classifier_head,
            reward_calculator=reward_calculator,
            device=device,
            training=training
        )
        # Important: use a different seed for each environment
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init

class EagerTrainingCallback(BaseCallback):
    """
    Custom callback for EAGER training monitoring with enhanced verbosity and profiling.
    """
    
    def __init__(self, verbose: int = 0, reward_calculator=None, evaluator=None, log_freq=50, experiment_tracker=None):
        super().__init__(verbose)
        self.log_freq = log_freq # Frequency for console logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = {i: 0 for i in range(NUM_ACTIONS)}
        self.episode_count = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.last_log_time = time.time()
        self.reward_calculator = reward_calculator
        self.evaluator = evaluator
        self.experiment_tracker = experiment_tracker

        # For tracking recent confidence and uncertainty efficiently
        self.recent_confidences = deque(maxlen=log_freq)
        self.recent_uncertainties = deque(maxlen=log_freq)
        # Track performance metrics for monitoring
        self.recent_step_times = deque(maxlen=500)
        self.recent_mcdo_times = deque(maxlen=500)
        
        # Enhanced tracking for detailed evaluation
        self.episode_data = {
            'confidence_trajectory': [],
            'uncertainty_trajectory': [],
            'action_sequence': [],
            'frames_analyzed': 0,
            'frames_skipped': 0,
            'current_frame': 0
        }

    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        
        # Determine the number of parallel environments
        num_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1

        # Update curriculum learning progress based on the number of environments
        if self.reward_calculator:
            self.reward_calculator.update_curriculum_progress(num_envs)
        
        # Track action distribution
        if 'actions' in self.locals:
            actions = self.locals['actions']
            # Handle potential tensor/numpy array shapes from vectorized envs
            if isinstance(actions, (np.ndarray, torch.Tensor)):
                 actions = actions.flatten()
            for action in actions:
                action_int = int(action)
                self.action_counts[action_int] += 1
                # Track action sequence for current episode
                self.episode_data['action_sequence'].append(action_int)
                
                # Track frame analysis patterns
                if action_int in [0, 1]:  # NEXT or ANALYZE_CURRENT
                    self.episode_data['frames_analyzed'] += 1
                elif action_int == 2:  # SKIP_FORWARD
                    self.episode_data['frames_skipped'] += 1
        
        # Monitor Profiling Data (collected every step)
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                if 'profiling' in info:
                    self.recent_step_times.append(info['profiling']['total_step_ms'])
                    # Track the time specifically spent on uncertainty (MCDO)
                    self.recent_mcdo_times.append(info['profiling']['uncertainty_calc_ms'])

        
        # Track episode completions
        if 'dones' in self.locals and 'infos' in self.locals:
            dones = self.locals['dones']
            infos = self.locals['infos']
            
            # Handle vectorized environments
            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]
                    self.episode_count += 1
                    
                    # Track episode metrics (Stable Baselines Monitor wrapper adds 'episode' key)
                    if 'episode' in info:
                        episode_reward = info['episode']['r']
                        episode_length = info['episode']['l']
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)

                    # Track Confidence and Uncertainty (from DeepfakeEnv info)
                    final_confidence = info.get('current_confidence', 0.0)
                    final_uncertainty = info.get('current_uncertainty', 0.0)
                    self.recent_confidences.append(final_confidence)
                    self.recent_uncertainties.append(final_uncertainty)
                    
                    # Track trajectory data
                    self.episode_data['confidence_trajectory'].append(final_confidence)
                    self.episode_data['uncertainty_trajectory'].append(final_uncertainty)
                    self.episode_data['current_frame'] = info.get('current_frame_idx', 0)

                    # Log these metrics individually to TensorBoard for detailed graphs
                    self.logger.record("rollout/ep_final_confidence", final_confidence)
                    self.logger.record("rollout/ep_final_uncertainty", final_uncertainty)
                    
                    # Track accuracy if available
                    true_label = None
                    predicted_label = None
                    is_correct = False
                    
                    if 'true_label' in info and 'final_prediction' in info:
                        if info['final_prediction'] is not None:
                            self.total_predictions += 1
                            true_label = info['true_label']
                            predicted_label = info['final_prediction']
                            is_correct = (predicted_label == true_label)
                            
                            if is_correct:
                                self.correct_predictions += 1
                            
                            # Add to evaluator if available
                            if self.evaluator:
                                # ... (evaluator logging remains the same) ...
                                self.evaluator.add_batch(
                                    predictions=torch.tensor([predicted_label]),
                                    labels=torch.tensor([true_label]),
                                    probabilities=None,
                                    video_ids=[info.get('video_id', f"video_{self.episode_count}")]
                                )
                                
                                # Add enhanced episode data to evaluator
                                self.evaluator.correct_predictions.append(is_correct)
                                self.evaluator.true_labels.append(true_label)
                                self.evaluator.predicted_labels.append(predicted_label)
                                self.evaluator.final_confidences.append(final_confidence)
                                self.evaluator.final_uncertainties.append(final_uncertainty)
                                self.evaluator.decision_frames.append(self.episode_data['current_frame'])
                                self.evaluator.frames_analyzed.append(self.episode_data['frames_analyzed'])
                                self.evaluator.frames_skipped.append(self.episode_data['frames_skipped'])
                                self.evaluator.confidence_trajectories.append(self.episode_data['confidence_trajectory'][-50:])
                                self.evaluator.uncertainty_trajectories.append(self.episode_data['uncertainty_trajectory'][-50:])
                                self.evaluator.action_sequences.append(self.episode_data['action_sequence'][-50:])
                    
                    # Reset episode data for next episode
                    self.episode_data = {
                        'confidence_trajectory': [],
                        'uncertainty_trajectory': [],
                        'action_sequence': [],
                        'frames_analyzed': 0,
                        'frames_skipped': 0,
                        'current_frame': 0
                    }
                    
                    # Log metrics to experiment tracker for plotting
                    if self.experiment_tracker and self.episode_count % 10 == 0:  # Log every 10 episodes
                        current_accuracy = (self.correct_predictions / self.total_predictions 
                                          if self.total_predictions > 0 else 0)
                        
                        # Calculate rolling averages
                        window = min(100, len(self.episode_rewards))
                        avg_reward = np.mean(self.episode_rewards[-window:]) if self.episode_rewards else 0
                        avg_length = np.mean(self.episode_lengths[-window:]) if self.episode_lengths else 0
                        
                        self.experiment_tracker.log_metrics({
                            'episode_reward': avg_reward,
                            'episode_length': avg_length,
                            'episode_accuracy': current_accuracy,
                            'avg_confidence': np.mean(self.recent_confidences) if self.recent_confidences else 0,
                            'avg_uncertainty': np.mean(self.recent_uncertainties) if self.recent_uncertainties else 0
                        }, step=self.episode_count)
                    
                    # Log detailed stats periodically
                    if self.episode_count % self.log_freq == 0:
                        self._log_detailed_stats()
        
        return True
    
    def _log_detailed_stats(self):
        """Log detailed statistics including confidence, uncertainty, threshold, and profiling."""
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        eps_per_sec = self.log_freq / time_elapsed if time_elapsed > 0 else 0
        
        accuracy = (self.correct_predictions / self.total_predictions * 100 
                    if self.total_predictions > 0 else 0)
        
        # Calculate average confidence/uncertainty
        avg_conf = np.mean(self.recent_confidences) if self.recent_confidences else 0
        avg_uncert = np.mean(self.recent_uncertainties) if self.recent_uncertainties else 0
        
        # Profiling Stats (Average over the longer window)
        avg_step_ms = np.mean(self.recent_step_times) if self.recent_step_times else 0
        avg_mcdo_ms = np.mean(self.recent_mcdo_times) if self.recent_mcdo_times else 0

        # Enhanced Logging
        log_info = f"  Ep {self.episode_count} | Acc: {accuracy:.1f}% ({self.correct_predictions}/{self.total_predictions}) | Conf: {avg_conf:.3f} | Uncert: {avg_uncert:.3f}"

        # Log curriculum progress (Confidence Threshold)
        if self.reward_calculator and self.reward_calculator.enable_curriculum:
            curr_threshold = self.reward_calculator.get_current_confidence_threshold()
            log_info += f" | Threshold: {curr_threshold:.3f}"
            # Log threshold to TensorBoard
            self.logger.record("train/confidence_threshold", curr_threshold)

        
        # Add profiling and speed
        log_info += f" | Speed: {eps_per_sec:.1f} eps/s | Avg Step: {avg_step_ms:.1f}ms (MCDO: {avg_mcdo_ms:.1f}ms)"
        logger.info(log_info)
        
        self.last_log_time = current_time
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Log action distribution
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            action_dist_str = ", ".join([
                f"{ACTION_NAMES[i]}: {count/total_actions:.1%}"
                for i, count in self.action_counts.items()
            ])
            logger.info(f"  Rollout Action distribution: {action_dist_str}")
            
        
        # Track RL-specific metrics in evaluator
        if self.evaluator and self.episode_rewards:
            # Calculate action distribution
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                action_dist = {
                    ACTION_NAMES[i]: count / total_actions
                    for i, count in self.action_counts.items()
                }
                
                # Add episode metrics
                if self.episode_rewards:
                    # Use a larger window for more stable averages
                    window_size = 100
                    avg_reward = np.mean(self.episode_rewards[-window_size:])
                    avg_length = np.mean(self.episode_lengths[-window_size:]) if self.episode_lengths else 0
                    
                    # Get current confidence threshold if using curriculum
                    conf_threshold = None
                    if self.reward_calculator and self.reward_calculator.enable_curriculum:
                        conf_threshold = self.reward_calculator.get_current_confidence_threshold()
                    
                    # Add enhanced metrics to evaluator
                    self.evaluator.add_episode(
                        reward=avg_reward,
                        length=int(avg_length),
                        action_dist=action_dist,
                        confidence_threshold=conf_threshold,
                        # Enhanced metrics from episode data
                        confidence_trajectory=self.episode_data['confidence_trajectory'][-100:] if self.episode_data['confidence_trajectory'] else None,
                        uncertainty_trajectory=self.episode_data['uncertainty_trajectory'][-100:] if self.episode_data['uncertainty_trajectory'] else None,
                        action_sequence=self.episode_data['action_sequence'][-100:] if self.episode_data['action_sequence'] else None,
                        decision_frame=self.episode_data['current_frame'],
                        frames_analyzed_count=self.episode_data['frames_analyzed'],
                        frames_skipped_count=self.episode_data['frames_skipped'],
                        final_confidence=final_confidence if 'final_confidence' in locals() else None,
                        final_uncertainty=final_uncertainty if 'final_uncertainty' in locals() else None,
                        correct_prediction=self.correct_predictions > 0,  # Simplified for now
                        true_label=None,  # Will be updated when we have the info
                        predicted_label=None  # Will be updated when we have the info
                    )
        
        # Reset counters
        self.action_counts = {i: 0 for i in range(NUM_ACTIONS)}


class Phase2RLTrainer:
    """
    Phase 2: PPO-LSTM Reinforcement Learning Training.
    Trains agent to make strategic decisions about evidence gathering.
    """
    
    def __init__(
        self,
        train_dataset: ProcessedVideoDataset,
        val_dataset: ProcessedVideoDataset,
        vision_backbone: VisionBackbone,
        temporal_memory: TemporalMemory,
        classifier_head: ClassifierHead,
        frozen_evaluator: FrozenEvaluator,
        device: str = DEVICE
    ):
        self.device = device
        
        # Create reward calculator
        self.reward_calculator = RewardCalculator(
            frozen_evaluator=frozen_evaluator,
            verbose=False
        )
        
        # Store components for environment creation
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Ensure components are on the correct device BEFORE multiprocessing starts
        self.vision_backbone = vision_backbone.to(device)
        self.temporal_memory = temporal_memory.to(device)
        self.classifier_head = classifier_head.to(device)
        
        # Create parallel training environments
        if NUM_PARALLEL_ENVS > 1:
            logger.info(f"Creating {NUM_PARALLEL_ENVS} parallel environments using SubprocVecEnv...")

            # Create list of environment creation functions using the standalone helper
            env_fns = [
                make_env(
                    rank=i,
                    dataset=self.train_dataset,
                    vision_backbone=self.vision_backbone,
                    temporal_memory=self.temporal_memory,
                    classifier_head=self.classifier_head,
                    reward_calculator=self.reward_calculator,
                    device=self.device,
                    training=True,
                    seed=SEED # Pass the base seed
                )
                for i in range(NUM_PARALLEL_ENVS)
            ]
            
          
            try:
                logger.info("Attempting to initialize SubprocVecEnv with 'spawn' method...")
                # Environment initialization may take time due to process spawning
                self.train_env = SubprocVecEnv(env_fns, start_method='spawn')
                logger.info("Successfully initialized SubprocVecEnv.")
            except Exception as e:
                # Fallback if multiprocessing fails
                logger.warning(f"SubprocVecEnv failed initialization (Error: {e}). Falling back to DummyVecEnv (slower). Ensure your main script is protected by if __name__ == '__main__':")
                self.train_env = DummyVecEnv(env_fns)

        else:
            # Single environment (using the helper for consistency and proper Monitor wrapping)
            logger.info("Creating single training environment...")
            # Wrap in DummyVecEnv as SB3 expects a vectorized environment
            env_fn = make_env(
                rank=0, dataset=self.train_dataset, vision_backbone=self.vision_backbone,
                temporal_memory=self.temporal_memory, classifier_head=self.classifier_head,
                reward_calculator=self.reward_calculator, device=self.device, training=True, seed=SEED
            )
            self.train_env = DummyVecEnv([env_fn])
        
        # Create validation environment (single, wrapped in DummyVecEnv for EvalCallback)
        val_env_fn = make_env(
            rank=0, dataset=self.val_dataset, vision_backbone=self.vision_backbone,
            temporal_memory=self.temporal_memory, classifier_head=self.classifier_head,
            reward_calculator=self.reward_calculator, device=self.device, training=False, seed=SEED+999
        )
        self.val_env = DummyVecEnv([val_env_fn])
        
        # Initialize RL evaluator for Phase 2
        self.evaluator = RLEvaluator(
            phase_name="phase2_rl",
            save_dir=LOG_DIR 
        )
        
        logger.info("Initialized Phase 2 RL Trainer")
    
    def create_ppo_model(self) -> PPO:
        # Create PPO model with custom policy
        model = PPO(
            policy=EagerActorCriticPolicy,
            env=self.train_env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            tensorboard_log=str(TENSORBOARD_DIR),
            device=self.device,
            verbose=1
        )
        
        return model
    
    def train(
        self,
        total_timesteps: int = TOTAL_TIMESTEPS,
        save_freq: int = SAVE_FREQ,
        eval_freq: int = EVAL_FREQ,
        experiment_tracker=None
    ):
        """
        Run PPO training.
        
        Args:
            total_timesteps: Total training timesteps
            save_freq: Checkpoint save frequency
            eval_freq: Evaluation frequency
            experiment_tracker: Optional experiment tracker for metrics logging
        """
        logger.info("="*80)
        logger.info(f"Starting Phase 2: PPO-LSTM Reinforcement Learning")
        logger.info(f"Training for {total_timesteps} timesteps")
        logger.info(f"Learning rate: {PPO_LEARNING_RATE}, Batch size: {PPO_BATCH_SIZE}")
        
        # Calculate effective frequencies for vectorized environments
        # SB3 callbacks operate based on the number of steps per environment
        num_envs = self.train_env.num_envs if hasattr(self.train_env, 'num_envs') else 1
        effective_save_freq = max(1, save_freq // num_envs)
        effective_eval_freq = max(1, eval_freq // num_envs)

        if num_envs > 1:
            logger.info(f"Using {num_envs} parallel environments")
            logger.info(f"Dataset: {len(self.train_dataset)} training videos")
            logger.info(f"Effective Save Freq: {effective_save_freq} steps/env (Total approx: {save_freq})")
            logger.info(f"Effective Eval Freq: {effective_eval_freq} steps/env (Total approx: {eval_freq})")
        else:
             # Access dataset through Monitor wrapper if single env
            logger.info(f"Environment: {len(self.train_env.env.dataset)} training videos")
        logger.info("="*80)
        
        # Create PPO model
        model = self.create_ppo_model()
        
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=effective_save_freq,
            save_path=str(CHECKPOINT_DIR),
            name_prefix="ppo_eager"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.val_env,
            best_model_save_path=str(FINAL_MODEL_DIR),
            log_path=str(TRAINING_LOG_DIR),
            eval_freq=effective_eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Custom callback with reward calculator for curriculum tracking and evaluator
        eager_callback = EagerTrainingCallback(
            verbose=1,
            reward_calculator=self.reward_calculator,
            evaluator=self.evaluator,
            log_freq=50, # Log to console every 50 episodes
            experiment_tracker=experiment_tracker
        )
        callbacks.append(eager_callback)
        
        # Combine callbacks
        callback = CallbackList(callbacks)
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(FINAL_MODEL_DIR / "ppo_eager_final")
        
        # Generate comprehensive evaluation report for Phase 2
        logger.info("\nGenerating Phase 2 RL evaluation report...")
        
        # Evaluate final performance
        self._evaluate_rl_performance(model)
        
        # Generate RL-specific plots
        self.evaluator.generate_rl_plots()
        
        # Generate full evaluation report
        phase2_metrics = self.evaluator.generate_full_report()
        logger.info(f"Phase 2 evaluation metrics: {phase2_metrics}")
        
        logger.info("Phase 2 RL training complete!")
        
        return model
    
    def _evaluate_rl_performance(self, model):
        """Evaluate RL model performance on validation set."""
        logger.info("Evaluating RL model on validation set...")
        
        # Reset evaluator for final evaluation
        self.evaluator.reset_metrics()
        
        # Run evaluation episodes
        num_eval_episodes = 100
        correct = 0
        total = 0
        
        for i in range(num_eval_episodes):
            obs = self.val_env.reset()  # DummyVecEnv returns only observation
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # DummyVecEnv returns: obs, rewards, dones, infos (4 values)
                obs, rewards, dones, infos = self.val_env.step(action)
                done = dones[0]  # Get first (and only) environment's done status
                info = infos[0]  # Get first (and only) environment's info
                
                if done and 'true_label' in info and 'final_prediction' in info:
                    if info['final_prediction'] is not None:
                        total += 1
                        if info['final_prediction'] == info['true_label']:
                            correct += 1
                        
                        # Add to evaluator
                        self.evaluator.add_batch(
                            predictions=torch.tensor([info['final_prediction']]),
                            labels=torch.tensor([info['true_label']]),
                            probabilities=None,
                            video_ids=[f"eval_video_{i}"]
                        )
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"RL Model Validation Accuracy: {accuracy:.2%} ({correct}/{total})")


def run_complete_training():
    """
    Run complete multi-phase EAGER training pipeline.
    """
    logger.info("="*50)
    logger.info("Starting EAGER Multi-Phase Training")
    logger.info("="*50)
    
    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=WARMSTART_BATCH_SIZE
    )
    
    # Get datasets for RL
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Initialize models
    logger.info("Initializing models...")
    feature_extractor = FeatureExtractorModule().to(DEVICE)
    
    # ========== PHASE 1: Supervised Warm-Start ==========
    logger.info("\n" + "="*50)
    logger.info("PHASE 1: Supervised Warm-Start")
    logger.info("="*50)
    
    phase1_trainer = Phase1WarmStartTrainer(
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    frozen_evaluator = phase1_trainer.train(num_epochs=WARMSTART_EPOCHS)
    
    # ========== PHASE 2: PPO-LSTM RL Training ==========
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: PPO-LSTM Reinforcement Learning")
    logger.info("="*50)
    
    # Extract components for RL
    vision_backbone = feature_extractor.vision_backbone
    temporal_memory = feature_extractor.temporal_memory
    classifier_head = feature_extractor.classifier_head
    
    # Freeze vision backbone for RL
    vision_backbone.eval()
    for param in vision_backbone.parameters():
        param.requires_grad = False
    
    phase2_trainer = Phase2RLTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vision_backbone=vision_backbone,
        temporal_memory=temporal_memory,
        classifier_head=classifier_head,
        frozen_evaluator=frozen_evaluator
    )
    
    ppo_model = phase2_trainer.train(
        total_timesteps=TOTAL_TIMESTEPS,
        experiment_tracker=None  
    )
    
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    
    # Save training metadata
    metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'config': get_config(),
        'phase1_best_acc': phase1_trainer.best_val_acc,
        'total_timesteps': TOTAL_TIMESTEPS
    }
    
    with open(FINAL_MODEL_DIR / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training metadata saved to {FINAL_MODEL_DIR}")
    
    return ppo_model, frozen_evaluator


if __name__ == "__main__":
    # Setup logging only if not already configured
    if not logging.getLogger().handlers:
        # Create file handler that flushes immediately (Windows fix)
        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
        
        # Setup logging with flush handler
        log_file = TRAINING_LOG_DIR / 'training.log'
        file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
    
    # Run training
    model, evaluator = run_complete_training()