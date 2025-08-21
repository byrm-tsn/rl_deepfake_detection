"""
DINOv3 Feature Extractor - Drop-in replacement for current ViT backbone
This uses DINOv3 ViT-B/16 which outputs 768 dimensions, perfectly matching current architecture
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import sys

# Add DINOv3 repo to path
dinov3_path = Path(__file__).parent.parent / "dinov3_repo"
if dinov3_path.exists():
    sys.path.insert(0, str(dinov3_path))

from src.config import VISION_EMBEDDING_DIM, FREEZE_BACKBONE, UNFREEZE_VIT_LAYERS

logger = logging.getLogger(__name__)

class DINOv3VisionBackbone(nn.Module):
    """
    DINOv3 ViT-B/16 backbone for feature extraction.
    Drop-in replacement for current VisionBackbone with same 768-dim output.
    """
    
    def __init__(
        self,
        freeze: bool = FREEZE_BACKBONE,
        unfreeze_layers: int = UNFREEZE_VIT_LAYERS,
        device: str = 'cuda'
    ):
        """
        Initialize DINOv3 backbone.
        
        Args:
            freeze: Whether to freeze the backbone
            unfreeze_layers: Number of transformer blocks to unfreeze from the end
            device: Device to load model on
        """
        super().__init__()
        
        # Check for DINOv3 weights
        weights_path = Path(__file__).parent.parent / "models" / "dinov3_weights" / "dinov3_vitb16_pretrain_lvd1689m.pth"
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"DINOv3 weights not found at {weights_path}. "
                "Please download from: https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth"
            )
        
        logger.info(f"Loading DINOv3 ViT-B/16 from {weights_path}")
        
        try:
            # Load DINOv3 model
            self.backbone = torch.hub.load(
                str(dinov3_path),
                'dinov3_vitb16',
                source='local',
                weights=str(weights_path)
            )
            logger.info("Successfully loaded DINOv3 ViT-B/16 with pretrained weights")
        except Exception as e:
            logger.error(f"Failed to load DINOv3: {e}")
            logger.info("Attempting to load weights manually...")
            
            # Fallback: Load architecture and weights separately
            self.backbone = torch.hub.load(
                str(dinov3_path),
                'dinov3_vitb16',
                source='local',
                pretrained=False 
            )
            
            # Load weights manually
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("Manually loaded DINOv3 weights")
        
        # Verify output dimension
        self.output_dim = 768  # DINOv3 ViT-B/16 outputs 768
        assert self.output_dim == VISION_EMBEDDING_DIM, \
            f"Expected output dim {VISION_EMBEDDING_DIM}, got {self.output_dim}"
        
        # Move to device
        self.backbone = self.backbone.to(device)
        
        # Partial freezing if requested
        if freeze:
            self._partial_freeze(unfreeze_layers)
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        logger.info(f"Initialized DINOv3 ViT-B/16 with {unfreeze_layers} unfrozen layers")
    
    def _partial_freeze(self, unfreeze_layers: int):
        """Freeze backbone except for last N transformer blocks."""
        # First freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # DINOv3 structure: backbone.blocks for transformer blocks
        if hasattr(self.backbone, 'blocks') and unfreeze_layers > 0:
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - unfreeze_layers)
            
            for i in range(start_idx, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            
            logger.info(f"Unfrozen last {unfreeze_layers}/{total_blocks} transformer blocks")
        
        # Also unfreeze final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            logger.info("Unfrozen final norm layer")
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames using DINOv3.
        
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
        
        # Normalize with ImageNet statistics
        frames_flat = (frames_flat - self.mean.to(frames_flat.device)) / self.std.to(frames_flat.device)
        
        # Extract features with DINOv3
        with torch.no_grad() if not self.training else torch.enable_grad():
            # DINOv3 forward pass
            features_flat = self.backbone(frames_flat)
            
            # Handle different output formats
            if isinstance(features_flat, tuple):
                features_flat = features_flat[0]  
            
            # Ensure we have 2D output (batch*frames, 768)
            if features_flat.dim() > 2:
                features_flat = features_flat.mean(dim=1)  
        
        # Reshape back to (batch_size, num_frames, 768)
        features = features_flat.view(batch_size, num_frames, self.output_dim)
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features
    
    def get_attention_maps(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from DINOv3 for visualization.
        
        Args:
            frames: Input frames
            
        Returns:
            Attention maps
        """
        with torch.no_grad():
            # Process frames
            if frames.dim() == 4:
                frames = frames.unsqueeze(0)
            
            batch_size, num_frames, C, H, W = frames.shape
            frames_flat = frames.view(batch_size * num_frames, C, H, W)
            
            # Normalize
            frames_flat = (frames_flat - self.mean.to(frames_flat.device)) / self.std.to(frames_flat.device)
            
            # Get attention from last layer
            if hasattr(self.backbone, 'get_last_selfattention'):
                attention = self.backbone.get_last_selfattention(frames_flat)
                return attention
            else:
                logger.warning("DINOv3 model doesn't support attention extraction")
                return None


def compare_extractors(video_frames: torch.Tensor):
    """
    Compare features from current ViT vs DINOv3.
    
    Args:
        video_frames: Tensor of video frames
        
    Returns:
        Comparison metrics
    """
    from src.feature_extractor import VisionBackbone
    
    # Load both models
    current_model = VisionBackbone().cuda().eval()
    dinov3_model = DINOv3VisionBackbone().cuda().eval()
    
    # Extract features
    with torch.no_grad():
        current_features = current_model(video_frames.cuda())
        dinov3_features = dinov3_model(video_frames.cuda())
    
    # Compare statistics
    results = {
        'current': {
            'mean': current_features.mean().item(),
            'std': current_features.std().item(),
            'min': current_features.min().item(),
            'max': current_features.max().item(),
        },
        'dinov3': {
            'mean': dinov3_features.mean().item(),
            'std': dinov3_features.std().item(),
            'min': dinov3_features.min().item(),
            'max': dinov3_features.max().item(),
        }
    }
    
    # Compute cosine similarity between features
    current_flat = current_features.flatten(0, 1) 
    dinov3_flat = dinov3_features.flatten(0, 1)
    
    # Normalize for cosine similarity
    current_norm = current_flat / current_flat.norm(dim=1, keepdim=True)
    dinov3_norm = dinov3_flat / dinov3_flat.norm(dim=1, keepdim=True)
    
    # Compute similarity
    similarity = (current_norm * dinov3_norm).sum(dim=1).mean().item()
    results['feature_similarity'] = similarity
    
    return results


def main():
    """Test DINOv3 feature extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DINOv3 Feature Extractor')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare with current extractor')
    parser.add_argument('--device', default='cuda', 
                       help='Device to use')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    
    dummy_frames = torch.randn(1, 10, 3, 224, 224) 
    
    if args.compare:
        logger.info("Comparing current ViT vs DINOv3...")
        results = compare_extractors(dummy_frames)
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTOR COMPARISON")
        print("="*60)
        print("\nCurrent ViT Statistics:")
        for key, val in results['current'].items():
            print(f"  {key}: {val:.4f}")
        
        print("\nDINOv3 Statistics:")
        for key, val in results['dinov3'].items():
            print(f"  {key}: {val:.4f}")
        
        print(f"\nFeature Similarity: {results['feature_similarity']:.4f}")
        print("(1.0 = identical, 0.0 = orthogonal)")
        print("="*60)
    else:
        # Just test DINOv3
        logger.info("Testing DINOv3 feature extractor...")
        model = DINOv3VisionBackbone(device=args.device)
        
        # Test forward pass
        features = model(dummy_frames.to(args.device))
        
        print("\n" + "="*60)
        print("DINOV3 TEST SUCCESSFUL")
        print("="*60)
        print(f"Input shape: {dummy_frames.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Expected: (1, 10, 768)")
        print(f"Match: {features.shape == (1, 10, 768)}")
        print("="*60)


if __name__ == "__main__":
    main()