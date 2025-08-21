"""
Feature extraction module using Vision Transformer backbone.
Includes temporal memory network with LSTM for sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import timm
import logging
import math

# Import configuration and modules from project structure
from src.config import (
    VISION_BACKBONE, VISION_EMBEDDING_DIM, FREEZE_BACKBONE,
    VIT_MODEL_NAME,
    USE_DINOV3, DINOV3_WEIGHTS_PATH,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT, 
    LSTM_BIDIRECTIONAL, LSTM_EFFECTIVE_DIM,
    CLASSIFIER_HIDDEN_DIM, CLASSIFIER_DROPOUT,
    FRAMES_PER_VIDEO, FRAME_SIZE, DEVICE,
    UNFREEZE_VIT_LAYERS, USE_ATTENTION_POOLING,
    ATTENTION_HEADS, ATTENTION_DIM,
    CLASSIFIER_LAYERS, USE_RESIDUAL_CLASSIFIER
)

logger = logging.getLogger(__name__)


class VisionBackbone(nn.Module):
    """
    Vision Transformer backbone for frame feature extraction.
    Uses pre-trained ViT model with the classification head removed.
    """
    
    def __init__(
        self,
        model_name: str = VISION_BACKBONE,
        pretrained: bool = True,
        freeze: bool = FREEZE_BACKBONE,
        unfreeze_layers: int = UNFREEZE_VIT_LAYERS
    ):
        """
        Initialize Vision Transformer backbone with partial fine-tuning support.
        
        Args:
            model_name: Name of the ViT model from timm
            pretrained: Use pre-trained weights
            freeze: Freeze backbone parameters during training
            unfreeze_layers: Number of transformer blocks to unfreeze from the end
        """
        super().__init__()
        
        # Choose between DINOv3 and timm ViT
        if USE_DINOV3:
            # Use DINOv3 instead of timm
            logger.info("Using DINOv3 ViT-B/16 as vision backbone")
            self._load_dinov3()
        else:
            # Load pre-trained Vision Transformer from timm
            logger.info(f"Using timm {model_name} as vision backbone")
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
        
        # Get output dimension
        if USE_DINOV3:
            # DINOv3 ViT-B/16 always outputs 768
            self.output_dim = 768
        else:
            # Get from timm model
            self.output_dim = self.backbone.num_features
        
        assert self.output_dim == VISION_EMBEDDING_DIM, \
            f"Expected output dim {VISION_EMBEDDING_DIM}, got {self.output_dim}"
        
        # Partial freezing if requested
        if freeze:
            self._partial_freeze(unfreeze_layers)
        
        # ImageNet normalization - proper means and stds for pre-trained ViT
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        model_type = "DINOv3 ViT-B/16" if USE_DINOV3 else model_name
        logger.info(f"Initialized {model_type} backbone with {unfreeze_layers} unfrozen layers")
    
    def _load_dinov3(self):
        """Load DINOv3 model instead of timm."""
        import sys
        from pathlib import Path
        
        # Add DINOv3 repo to path
        dinov3_path = Path(__file__).parent.parent / "dinov3_repo"
        if dinov3_path.exists():
            sys.path.insert(0, str(dinov3_path))
        else:
            raise FileNotFoundError(f"DINOv3 repository not found at {dinov3_path}")
        
        # Check if weights exist
        if not DINOV3_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"DINOv3 weights not found at {DINOV3_WEIGHTS_PATH}. "
                "Download from: https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth"
            )
        
        try:
            # Load DINOv3 model with weights
            self.backbone = torch.hub.load(
                str(dinov3_path),
                'dinov3_vitb16',
                source='local',
                weights=str(DINOV3_WEIGHTS_PATH)
            )
            logger.info("Successfully loaded DINOv3 ViT-B/16 with pretrained weights")
        except Exception as e:
            logger.warning(f"Failed to load DINOv3 with torch.hub: {e}")
            logger.info("Attempting manual loading...")
            
            # Fallback: Load architecture and weights separately
            self.backbone = torch.hub.load(
                str(dinov3_path),
                'dinov3_vitb16',
                source='local',
                pretrained=False
            )
            
            # Load weights manually
            checkpoint = torch.load(DINOV3_WEIGHTS_PATH, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("Manually loaded DINOv3 weights")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")
    
    def _partial_freeze(self, unfreeze_layers: int):
        """Freeze backbone except for last N transformer blocks."""
        # First freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Then unfreeze last N transformer blocks
        if hasattr(self.backbone, 'blocks') and unfreeze_layers > 0:
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - unfreeze_layers)
            for i in range(start_idx, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            logger.info(f"Unfrozen last {unfreeze_layers} transformer blocks")
        
        # Also unfreeze norm layer for better fine-tuning
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        
        # For DINOv3, also check for 'norm_layer' attribute
        if USE_DINOV3 and hasattr(self.backbone, 'norm_layer'):
            for param in self.backbone.norm_layer.parameters():
                param.requires_grad = True
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            frames: Tensor of shape (batch_size, num_frames, 3, 224, 224)
                   or (num_frames, 3, 224, 224) for single video
            
        Returns:
            Features of shape (batch_size, num_frames, 768) or (num_frames, 768)
        """
        # Handle both batched and single video inputs
        if frames.dim() == 4:  # Single video
            frames = frames.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_frames, C, H, W = frames.shape
        
        # Reshape for batch processing
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        
        # Normalize with ImageNet statistics (ensure mean/std are on same device)
        frames_flat = (frames_flat - self.mean.to(frames_flat.device)) / self.std.to(frames_flat.device)
        
        # Extract features
        with torch.no_grad() if self.backbone.training == False else torch.enable_grad():
            features_flat = self.backbone(frames_flat)
            
            # Handle DINOv3 output format if needed
            if USE_DINOV3 and isinstance(features_flat, tuple):
                # DINOv3 might return tuple, take first element
                features_flat = features_flat[0]
            
            # Ensure we have the right shape
            if features_flat.dim() > 2:
                # Pool if needed (shouldn't happen with ViT, but safety check)
                features_flat = features_flat.mean(dim=1)
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        features = features_flat.view(batch_size, num_frames, -1)
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features


class TemporalMemory(nn.Module):
    """
    Bidirectional LSTM network for temporal sequence modeling.
    Processes frame features to capture temporal dependencies.
    """
    
    def __init__(
        self,
        input_size: int = VISION_EMBEDDING_DIM,
        hidden_size: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = LSTM_BIDIRECTIONAL
    ):
        """
        Initialize LSTM temporal memory.
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM network
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Add dropout after LSTM output 
        from src.config import WARMSTART_LSTM_DROPOUT
        self.lstm_dropout = nn.Dropout(WARMSTART_LSTM_DROPOUT) 
        
        # Output dimension
        self.output_dim = hidden_size * self.num_directions
        
        logger.info(f"Initialized LSTM (hidden={hidden_size}, layers={num_layers}, "
                   f"bidirectional={bidirectional})")
    
    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence of frame features.
        
        Args:
            features: Frame features (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            output: LSTM output (batch_size, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final hidden and cell states
        """
        if hidden is None:
            hidden = self.init_hidden(features.size(0), features.device)
        
        output, (h_n, c_n) = self.lstm(features, hidden)
        
        # Apply dropout after LSTM output 
        if self.training:
            output = self.lstm_dropout(output)
        
        return output, (h_n, c_n)
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            (h_0, c_0): Initial hidden and cell states
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros_like(h_0)
        
        return (h_0, c_0)
    
    def get_final_hidden(
        self,
        hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract final hidden state for use as temporal memory.
        
        Args:
            hidden_states: (h_n, c_n) from LSTM
            
        Returns:
            Final hidden state of shape (batch_size, hidden_size * num_directions)
        """
        h_n, _ = hidden_states
        
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2, :, :]  # Last layer, forward
            h_backward = h_n[-1, :, :]  # Last layer, backward
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            # Just take the last layer's hidden state
            final_hidden = h_n[-1, :, :]
        
        return final_hidden


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention mechanism for temporal feature aggregation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = ATTENTION_DIM,
        num_heads: int = ATTENTION_HEADS,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
        
        Returns:
            Pooled features of shape (batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply multi-head attention
        attended, _ = self.attention(query, x, x, key_padding_mask=mask)
        
        # Squeeze sequence dimension and project
        attended = attended.squeeze(1)
        output = self.output_proj(attended)
        
        return output


class ClassifierHead(nn.Module):
    """
    Enhanced classification head with deeper architecture and residual connections.
    Includes confidence estimation through entropy calculation.
    """
    
    def __init__(
        self,
        input_size: int = VISION_EMBEDDING_DIM + LSTM_EFFECTIVE_DIM, 
        hidden_size: int = CLASSIFIER_HIDDEN_DIM,
        dropout: float = CLASSIFIER_DROPOUT,
        num_layers: int = CLASSIFIER_LAYERS,
        use_residual: bool = USE_RESIDUAL_CLASSIFIER
    ):
        """
        Initialize enhanced classifier head with residual connections.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            dropout: Dropout probability
            num_layers: Number of classifier layers
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Build deeper classifier with optional residual connections
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 2)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Residual scaling factor
        self.residual_scale = 1.0 / math.sqrt(num_layers) if use_residual else 1.0
        
    def enable_mc_dropout(self):
        """
        Enable dropout layers during evaluation mode for Monte Carlo Dropout.
        This forces dropout layers (which might be in .eval() mode) to switch to .train() mode.
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify video as real or fake with residual connections.
        
        Args:
            features: Temporal memory features
            
        Returns:
            logits: Classification logits
            probs: Classification probabilities
            confidence: Confidence score
        """
        # Initial projection
        x = self.input_proj(features)
        x = F.relu(x)
        
        # Pass through hidden layers with residual connections
        for i in range(0, len(self.hidden_layers), 4):  # Process 4 components at a time
            if self.use_residual and i > 0:  # Skip residual for first layer
                residual = x
                # Apply layer block
                x = self.hidden_layers[i](x)  # Linear
                x = self.hidden_layers[i+1](x)  # LayerNorm
                x = self.hidden_layers[i+2](x)  # ReLU
                x = self.hidden_layers[i+3](x)  # Dropout
                # Add residual connection
                x = x + residual * self.residual_scale
            else:
                # Standard forward pass
                x = self.hidden_layers[i](x)
                x = self.hidden_layers[i+1](x)
                x = self.hidden_layers[i+2](x)
                x = self.hidden_layers[i+3](x)
        
        # Final classification
        logits = self.output_layer(x)
        
        # Apply temperature scaling for calibration
        calibrated_logits = logits / self.temperature
        probs = F.softmax(calibrated_logits, dim=-1)
        
        # Calculate confidence as maximum probability
        confidence = torch.max(probs, dim=-1)[0]
        
        # Return original logits for loss calculation, calibrated probs, and confidence
        return logits, probs, confidence


class FrozenEvaluator(nn.Module):
    """
    Frozen classifier for unbiased reward calculation during RL training.
    Copies the structure from ClassifierHead but with frozen parameters.
    """
    
    def __init__(self, classifier_head: ClassifierHead):
        """
        Initialize frozen evaluator from trained classifier.
        
        Args:
            classifier_head: Trained classifier to freeze
        """
        super().__init__()
        
        # Deep copy the entire classifier head structure
        import copy
        self.input_proj = copy.deepcopy(classifier_head.input_proj)
        self.hidden_layers = copy.deepcopy(classifier_head.hidden_layers)
        self.output_layer = copy.deepcopy(classifier_head.output_layer)
        self.temperature = copy.deepcopy(classifier_head.temperature)
        self.use_residual = classifier_head.use_residual
        self.residual_scale = classifier_head.residual_scale
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        logger.info("Created frozen evaluator for reward calculation")
        
    def enable_mc_dropout(self):
        """
        Enable dropout layers during evaluation mode for Monte Carlo Dropout.
        This forces dropout layers (which might be in .eval() mode) to switch to .train() mode.
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                # Even though weights are frozen, dropout needs train mode to be stochastic.
                m.train()
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate video classification for reward calculation.
        
        Args:
            features: Temporal memory features
            
        Returns:
            logits: Classification logits
            probs: Classification probabilities
            confidence: Confidence score
        """
        with torch.no_grad():
            # Initial projection
            x = self.input_proj(features)
            x = F.relu(x)
            
            # Pass through hidden layers with residual connections
            for i in range(0, len(self.hidden_layers), 4):  # Process 4 components at a time
                if self.use_residual and i > 0:  # Skip residual for first layer
                    residual = x
                    # Apply layer block
                    x = self.hidden_layers[i](x)  # Linear
                    x = self.hidden_layers[i+1](x)  # LayerNorm
                    x = self.hidden_layers[i+2](x)  # ReLU
                    x = self.hidden_layers[i+3](x)  # Dropout
                    # Add residual connection
                    x = x + residual * self.residual_scale
                else:
                    # Standard forward pass
                    x = self.hidden_layers[i](x)
                    x = self.hidden_layers[i+1](x)
                    x = self.hidden_layers[i+2](x)
                    x = self.hidden_layers[i+3](x)
            
            # Final classification
            logits = self.output_layer(x)
            
            # Apply temperature scaling for calibration
            calibrated_logits = logits / self.temperature
            probs = F.softmax(calibrated_logits, dim=-1)
            
            # Calculate confidence as maximum probability
            confidence = torch.max(probs, dim=-1)[0]
        
        return logits, probs, confidence


class FeatureExtractorModule(nn.Module):
    """
    Complete feature extraction module combining vision backbone, temporal memory,
    and attention pooling mechanism.
    """
    
    def __init__(self):
        """Initialize complete feature extraction pipeline with attention pooling."""
        super().__init__()
        
        self.vision_backbone = VisionBackbone()
        self.temporal_memory = TemporalMemory()
        
        # Add attention pooling if configured
        if USE_ATTENTION_POOLING:
            self.attention_pooling = MultiHeadAttentionPooling(
                input_dim=LSTM_EFFECTIVE_DIM,
                hidden_dim=ATTENTION_DIM,  # Should be 1024 per config
                num_heads=ATTENTION_HEADS
            )
            # Classifier input: last frame features (768) + attention pooled LSTM (1024) = 1792
            classifier_input_size = VISION_EMBEDDING_DIM + ATTENTION_DIM
        else:
            self.attention_pooling = None
            # Without attention: frame features (768) + LSTM final hidden (1024) = 1792
            classifier_input_size = VISION_EMBEDDING_DIM + LSTM_EFFECTIVE_DIM
        
        # Initialize classifier with correct input size (1792)
        self.classifier_head = ClassifierHead(input_size=classifier_input_size)
        
    def forward(
        self,
        frames: torch.Tensor,
        return_all_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from video frames with attention pooling.
        
        Args:
            frames: Video frames (batch_size, num_frames, 3, 224, 224)
            return_all_features: Return intermediate features
            
        Returns:
            Dictionary containing:
                - frame_features: Per-frame visual features
                - temporal_features: LSTM output features
                - temporal_memory: Final LSTM hidden state or attention-pooled features
                - logits: Classification logits
                - probs: Classification probabilities
                - confidence: Confidence scores
        """
        # Extract visual features
        frame_features = self.vision_backbone(frames)
        
        # Process through LSTM
        temporal_features, (h_n, c_n) = self.temporal_memory(frame_features)
        
        # Apply attention pooling if configured
        if self.attention_pooling is not None:
            # Use attention to aggregate temporal features
            temporal_aggregated = self.attention_pooling(temporal_features)
            
            # Get current frame features (use last processed frame)
            current_frame_features = frame_features[:, -1, :]  # Last frame in sequence
            
            # Combine frame and attention-pooled temporal features
            combined_features = torch.cat([current_frame_features, temporal_aggregated], dim=-1)
        else:
            # Original approach: use final LSTM hidden state
            temporal_memory = self.temporal_memory.get_final_hidden((h_n, c_n))
            
            # Get current frame features (use last processed frame)
            current_frame_features = frame_features[:, -1, :]  # Last frame in sequence
            
            # Combine frame and temporal features for classification
            combined_features = torch.cat([current_frame_features, temporal_memory], dim=-1)
            temporal_aggregated = temporal_memory
        
        # Classification
        logits, probs, confidence = self.classifier_head(combined_features)
        
        results = {
            'logits': logits,
            'probs': probs,
            'confidence': confidence,
            'temporal_memory': temporal_aggregated
        }
        
        if return_all_features:
            results.update({
                'frame_features': frame_features,
                'temporal_features': temporal_features,
                'lstm_hidden': h_n,
                'lstm_cell': c_n
            })
        
        return results


if __name__ == "__main__":
    # Test feature extraction
    logger.info("Testing feature extraction modules...")
    
    # Create dummy input
    batch_size = 2
    frames = torch.randn(batch_size, FRAMES_PER_VIDEO, 3, FRAME_SIZE, FRAME_SIZE)
    frames = frames.to(DEVICE)
    
    # Test vision backbone
    vision_backbone = VisionBackbone().to(DEVICE)
    frame_features = vision_backbone(frames)
    logger.info(f"Frame features shape: {frame_features.shape}")
    assert frame_features.shape == (batch_size, FRAMES_PER_VIDEO, VISION_EMBEDDING_DIM)
    
    # Test temporal memory
    temporal_memory = TemporalMemory().to(DEVICE)
    temporal_features, hidden = temporal_memory(frame_features)
    logger.info(f"Temporal features shape: {temporal_features.shape}")
    assert temporal_features.shape == (batch_size, FRAMES_PER_VIDEO, LSTM_EFFECTIVE_DIM)
    
    # Test classifier
    final_hidden = temporal_memory.get_final_hidden(hidden)
    classifier = ClassifierHead().to(DEVICE)
    logits, probs, confidence = classifier(final_hidden)
    logger.info(f"Logits shape: {logits.shape}, Confidence shape: {confidence.shape}")
    
    # Test complete module
    feature_extractor = FeatureExtractorModule().to(DEVICE)
    results = feature_extractor(frames, return_all_features=True)
    logger.info(f"Complete feature extraction successful!")
    logger.info(f"Output keys: {results.keys()}")
    
    print("Feature extraction tests completed successfully!")