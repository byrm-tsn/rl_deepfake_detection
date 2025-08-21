"""
Data loading and preprocessing module for EAGER algorithm.
Handles loading of preprocessed video frames and metadata management.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from tqdm import tqdm
import logging
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
# GPU-accelerated image decoding
try:
    import torchvision.io as tvio
    GPU_DECODE_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_DECODE_AVAILABLE = False

from src.config import (
    PROCESSED_DATASET_PATH, METADATA_PROCESSED,
    FRAMES_PER_VIDEO, FRAME_SIZE, CHANNELS,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DEVICE,
    USE_MIXED_PRECISION
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Data augmentation class for training data as specified in PDF.
    Applies random rotation, affine shear, and Gaussian noise.
    """
    
    def __init__(self, training: bool = True):
        """Initialize augmentation parameters."""
        self.training = training
        
    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to frame tensor.
        
        Args:
            img_tensor: Frame tensor of shape (C, H, W)
            
        Returns:
            Augmented frame tensor
        """
        if not self.training:
            return img_tensor
            
        # Convert to PIL image for transforms
        img_pil = TF.to_pil_image(img_tensor)
        
        # Apply augmentations with 50% probability each
        
        # 1. Random rotation between -15 and +15 degrees (50% probability)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img_pil = TF.rotate(img_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # 2. Random affine shear of 0.1 (50% probability)
        if random.random() < 0.5:
            shear_degrees = random.uniform(-10, 10)  # 0.1 radians ≈ 5.7 degrees
            img_pil = TF.affine(
                img_pil,
                angle=0,
                translate=(0, 0),
                scale=1.0,
                shear=[shear_degrees, 0],
                interpolation=TF.InterpolationMode.BILINEAR
            )
        
        # Convert back to tensor
        img_tensor = TF.to_tensor(img_pil)
        
        # 3. Random Gaussian noise with std 0.01 (50% probability)
        if random.random() < 0.5:
            noise = torch.randn_like(img_tensor) * 0.01
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        # Random horizontal flip (50% probability)
        if random.random() < 0.5:
            img_tensor = TF.hflip(img_tensor)
        
        # Random brightness/contrast jitter
        if random.random() < 0.5:
            img_tensor = TF.adjust_brightness(img_tensor, random.uniform(0.9, 1.1))
            img_tensor = TF.adjust_contrast(img_tensor, random.uniform(0.9, 1.1))
        
        return img_tensor


class ProcessedVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed video frames.
    Each video consists of exactly 50 preprocessed face frames.
    """
    
    def __init__(
        self, 
        split: str = "train",
        transform: Optional[Any] = None,
        validate_frames: bool = True,
        min_confidence: float = 0.7
    ):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional torchvision transforms
            validate_frames: Whether to validate frame integrity
            min_confidence: Minimum face detection confidence
        """
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.transform = transform
        self.validate_frames = validate_frames
        self.min_confidence = min_confidence
        
        # Load metadata
        self.metadata_path = METADATA_PROCESSED
        self.data_root = PROCESSED_DATASET_PATH / split
        
        # Load and process metadata
        self._load_metadata()
        
        # Validate dataset integrity
        if validate_frames:
            self._validate_dataset()
        
        logger.info(f"Loaded {len(self.video_ids)} videos for {split} split")
    
    def _load_metadata(self):
        """Load and process metadata CSV file."""
        if not self.metadata_path.exists():
            # If metadata doesn't exist, scan directory structure
            logger.warning(f"Metadata file not found at {self.metadata_path}")
            logger.info("Scanning directory structure instead...")
            self._scan_directory_structure()
            return
        
        # Load metadata CSV
        df = pd.read_csv(self.metadata_path)
        
        # Filter by split
        df_split = df[df['split'] == self.split]
        
        # Group by video_id
        grouped = df_split.groupby('video_id')
        
        # Create video entries
        self.videos = []
        self.video_ids = []
        self.labels = []
        
        for video_id, group in grouped:
            # Ensure exactly 50 frames per video
            if len(group) != FRAMES_PER_VIDEO:
                logger.warning(f"Video {video_id} has {len(group)} frames, expected {FRAMES_PER_VIDEO}")
                continue
            
            # Sort by frame_number to ensure correct order
            group = group.sort_values('frame_number')
            
            # Check minimum confidence if available
            if 'detection_confidence' in group.columns:
                if group['detection_confidence'].min() < self.min_confidence:
                    logger.warning(f"Video {video_id} has low confidence frames")
                    continue
            
            # Get label (0=real, 1=fake)
            label = 1 if group.iloc[0]['label'] == 'fake' else 0
            
            # Store video information with absolute paths
            # Convert relative paths from metadata to absolute paths
            frame_paths = []
            for rel_path in group['frame_path'].tolist():
                # Convert backslashes to forward slashes and create absolute path
                rel_path = rel_path.replace('\\', '/')
                abs_path = PROCESSED_DATASET_PATH / rel_path
                frame_paths.append(str(abs_path))
            
            self.videos.append({
                'video_id': video_id,
                'frames': frame_paths,
                'label': label
            })
            
            self.video_ids.append(video_id)
            self.labels.append(label)
    
    def _scan_directory_structure(self):
        """Scan directory structure when metadata is not available."""
        self.videos = []
        self.video_ids = []
        self.labels = []
        
        # Scan fake and real directories
        for label_name in ['fake', 'real']:
            label_dir = self.data_root / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue
            
            label = 1 if label_name == 'fake' else 0
            
            # Each subdirectory is a video
            for video_dir in label_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                # Get all frame files
                frame_files = sorted(video_dir.glob("*.png"))
                
                if len(frame_files) != FRAMES_PER_VIDEO:
                    logger.warning(f"Video {video_dir.name} has {len(frame_files)} frames")
                    continue
                
                self.videos.append({
                    'video_id': video_dir.name,
                    'frames': [str(f) for f in frame_files],
                    'label': label
                })
                
                self.video_ids.append(video_dir.name)
                self.labels.append(label)
    
    def _validate_dataset(self):
        """Validate dataset integrity."""
        logger.info(f"Validating {len(self.videos)} videos...")
        
        invalid_videos = []
        for i, video in enumerate(tqdm(self.videos, desc="Validating")):
            # Check if all frame files exist
            for frame_path in video['frames']:
                if not Path(frame_path).exists():
                    logger.warning(f"Frame not found: {frame_path}")
                    invalid_videos.append(i)
                    break
        
        # Remove invalid videos
        for i in reversed(invalid_videos):
            del self.videos[i]
            del self.video_ids[i]
            del self.labels[i]
        
        if invalid_videos:
            logger.warning(f"Removed {len(invalid_videos)} invalid videos")
    
    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load a video and its label.
        
        Args:
            idx: Video index
            
        Returns:
            frames_tensor: Tensor of shape (50, 3, 224, 224)
            label: 0 for real, 1 for fake
            video_id: Video identifier
        """
        video_info = self.videos[idx]
        video_id = video_info['video_id']
        label = video_info['label']
        frame_paths = video_info['frames']
        
        # Load all frames
        frames = []
        for frame_path in frame_paths:
            frame = self._load_frame(frame_path)
            frames.append(frame)
        
        # Stack frames: (50, 3, 224, 224)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label, video_id
    
    def _load_frame(self, frame_path: str) -> torch.Tensor:
        """
        Load and preprocess a single frame (GPU-accelerated when available).
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Preprocessed frame tensor of shape (3, 224, 224)
        """
        # Try GPU-accelerated decoding first 
        if GPU_DECODE_AVAILABLE and USE_MIXED_PRECISION:
            try:
                # Decode image directly to GPU tensor
                img_tensor = tvio.read_image(frame_path, mode=tvio.ImageReadMode.RGB)
                img_tensor = img_tensor.float() / 255.0
                
                # Resize if necessary using GPU
                if img_tensor.shape[1:] != (FRAME_SIZE, FRAME_SIZE):
                    img_tensor = TF.resize(img_tensor, [FRAME_SIZE, FRAME_SIZE])
                
                # Apply additional transforms if provided
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                
                return img_tensor
            except Exception as e:
                # Fallback to CPU decoding if GPU decoding fails
                logger.debug(f"GPU decode failed for {frame_path}, using CPU: {e}")
        
        # CPU decoding fallback
        img = cv2.imread(frame_path)
        if img is None:
            logger.error(f"Failed to load frame: {frame_path}")
            # Return black frame as fallback
            img = np.zeros((FRAME_SIZE, FRAME_SIZE, CHANNELS), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary
        if img.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
        
        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # Apply additional transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of real vs fake videos."""
        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        return {
            'real': real_count,
            'fake': fake_count,
            'total': len(self.labels)
        }


def create_data_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    validate: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of parallel workers
        pin_memory: Pin memory for faster GPU transfer
        validate: Whether to validate dataset integrity
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create augmentation transform for training
    train_transform = DataAugmentation(training=True)
    val_transform = DataAugmentation(training=False)  
    
    # Create datasets
    train_dataset = ProcessedVideoDataset(
        split="train",
        transform=train_transform,
        validate_frames=validate
    )
    
    val_dataset = ProcessedVideoDataset(
        split="val",
        transform=val_transform,
        validate_frames=validate
    )
    
    test_dataset = ProcessedVideoDataset(
        split="test",
        transform=val_transform,
        validate_frames=validate
    )
    
    # Print detailed dataset statistics
    logger.info("="*80)
    logger.info("DATASET STATISTICS:")
    logger.info("="*80)
    
    for name, dataset in [("Train", train_dataset), 
                          ("Val", val_dataset), 
                          ("Test", test_dataset)]:
        dist = dataset.get_class_distribution()
        real_ratio = (dist['real'] / dist['total'] * 100) if dist['total'] > 0 else 0
        fake_ratio = (dist['fake'] / dist['total'] * 100) if dist['total'] > 0 else 0
        
        logger.info(f"\n{name} Dataset:")
        logger.info(f"  Total Videos: {dist['total']}")
        logger.info(f"  Real Videos: {dist['real']} ({real_ratio:.1f}%)")
        logger.info(f"  Fake Videos: {dist['fake']} ({fake_ratio:.1f}%)")
        logger.info(f"  Total Frames: {dist['total'] * FRAMES_PER_VIDEO:,}")
        logger.info(f"  Real Frames: {dist['real'] * FRAMES_PER_VIDEO:,}")
        logger.info(f"  Fake Frames: {dist['fake'] * FRAMES_PER_VIDEO:,}")
        
        if real_ratio < 40 or fake_ratio < 40:
            logger.warning(f"  ⚠️ Class imbalance detected! Real: {real_ratio:.1f}%, Fake: {fake_ratio:.1f}%")
    
    # Create weighted sampler for balanced class sampling
    # Calculate class weights (inverse frequency)
    train_dist = train_dataset.get_class_distribution()
    real_count = train_dist['real']
    fake_count = train_dist['fake']
    total_count = train_dist['total']
    
    # Compute class weights (inverse frequency)
    class_weights = {
        0: total_count / (2 * real_count) if real_count > 0 else 1.0,  
        1: total_count / (2 * fake_count) if fake_count > 0 else 1.0   
    }
    
    logger.info(f"\nClass Balancing Weights:")
    logger.info(f"  Real class weight: {class_weights[0]:.3f}")
    logger.info(f"  Fake class weight: {class_weights[1]:.3f}")
    logger.info("="*80)
    
    # Create sample weights for each video in the dataset
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    
    # Create WeightedRandomSampler for balanced batches
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Data leakage detection - verify no overlap between splits
    if validate:
        logger.info("="*80)
        logger.info("VERIFYING DATA INTEGRITY - CHECKING FOR DATA LEAKAGE")
        logger.info("="*80)
        
        # Get video IDs from each split
        train_ids = set(train_dataset.video_ids)
        val_ids = set(val_dataset.video_ids)
        test_ids = set(test_dataset.video_ids)
        
        # Check for overlaps
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        
        if train_val_overlap:
            logger.error(f"⚠️ DATA LEAKAGE DETECTED: {len(train_val_overlap)} videos in both train and val!")
            logger.error(f"Overlapping videos: {list(train_val_overlap)[:5]}...")
        else:
            logger.info("✅ No overlap between train and validation sets")
            
        if train_test_overlap:
            logger.error(f"⚠️ DATA LEAKAGE DETECTED: {len(train_test_overlap)} videos in both train and test!")
        else:
            logger.info("✅ No overlap between train and test sets")
            
        if val_test_overlap:
            logger.error(f"⚠️ DATA LEAKAGE DETECTED: {len(val_test_overlap)} videos in both val and test!")
        else:
            logger.info("✅ No overlap between validation and test sets")
        
        # Log unique counts
        logger.info(f"\nUnique video counts:")
        logger.info(f"  Training: {len(train_ids)} unique videos")
        logger.info(f"  Validation: {len(val_ids)} unique videos")
        logger.info(f"  Test: {len(test_ids)} unique videos")
        logger.info("="*80)
    
    return train_loader, val_loader, test_loader


def validate_dimensions(frames: torch.Tensor) -> bool:
    """
    Validate tensor dimensions match expected format.
    
    Args:
        frames: Input tensor
        
    Returns:
        True if dimensions are valid
    """
    expected_shape = (BATCH_SIZE, FRAMES_PER_VIDEO, CHANNELS, FRAME_SIZE, FRAME_SIZE)
    
    if frames.dim() == 4:  # Single video
        expected_shape = (FRAMES_PER_VIDEO, CHANNELS, FRAME_SIZE, FRAME_SIZE)
    
    if frames.shape != expected_shape[:len(frames.shape)]:
        logger.error(f"Dimension mismatch. Expected {expected_shape}, got {frames.shape}")
        return False
    
    return True


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing data loader...")
    
    # Create a sample dataset
    dataset = ProcessedVideoDataset(split="train", validate_frames=False)
    
    if len(dataset) > 0:
        # Load first video
        frames, label, video_id = dataset[0]
        
        logger.info(f"Loaded video: {video_id}")
        logger.info(f"Frames shape: {frames.shape}")
        logger.info(f"Label: {'fake' if label == 1 else 'real'}")
        
        # Validate dimensions
        assert validate_dimensions(frames), "Dimension validation failed"
        
        logger.info("Data loader test completed successfully!")
    else:
        logger.warning("No videos found in dataset. Please add data to the processed_data directory.")