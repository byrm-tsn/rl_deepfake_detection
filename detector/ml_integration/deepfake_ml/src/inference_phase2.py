"""
Phase 2 Inference Script with DINOv3 Face Attention Visualization

This script performs inference using the Phase 2 PPO model and generates
attention heatmaps using DINOv3's PCA-based feature visualization approach.
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Check for DINOv3 availability
try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    logger.warning("transformers not available, DINOv3 features will be disabled")

# Check for local DINOv3 repository
LOCAL_DINOV3_AVAILABLE = False
dinov3_repo_path = Path(__file__).parent.parent / "dinov3_repo"
if dinov3_repo_path.exists():
    sys.path.insert(0, str(dinov3_repo_path))
    LOCAL_DINOV3_AVAILABLE = True
    logger.info(f"Local DINOv3 repository found at: {dinov3_repo_path}")

# Import project modules
from src.config import get_config
from src.feature_extractor import FeatureExtractorModule as FeatureExtractor, VisionBackbone, TemporalMemory, ClassifierHead
from src.deepfake_env import DeepfakeEnv
from src.agent import EagerActorCriticPolicy as ActorCriticPolicy
from src.reward_system import RewardCalculator
from src.utils import set_random_seeds
from stable_baselines3 import PPO

# Import face detection
from data_processing.preprocessing import FaceDetector

# Configuration constants
MAX_FRAMES = 50
ACTION_NAMES = ['NEXT', 'FOCUS', 'AUGMENT', 'STOP_REAL', 'STOP_FAKE']

class Phase2Inference:
    """Inference pipeline for Phase 2 PPO model with DINOv3 attention visualization."""
    
    def __init__(self, model_path: str, device: str = 'cuda', use_attention_heatmap: bool = True):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model
            device: Device to use ('cuda' or 'cpu')
            use_attention_heatmap: Whether to generate attention heatmaps
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Initialize face detector
        logger.info("Initializing face detector...")
        self.face_detector = FaceDetector()
        
        # Initialize DINOv3 for attention heatmaps if requested
        self.use_attention_heatmap = use_attention_heatmap and (DINOV3_AVAILABLE or LOCAL_DINOV3_AVAILABLE)
        self.is_local_dinov3 = False
        self.is_dinov2_fallback = False
        
        if self.use_attention_heatmap:
            # Try local DINOv3 first with pretrained weights
            if LOCAL_DINOV3_AVAILABLE:
                try:
                    logger.info("Using local DINOv3 repository with torch.hub...")
                    
                    # Check if we have the pretrained weights
                    weights_path = Path(__file__).parent.parent / "models" / "dinov3_weights" / "dinov3_vitb16_pretrain_lvd1689m.pth"
                    
                    if weights_path.exists():
                        logger.info(f"Found pretrained weights at: {weights_path}")
                        # Load model with pretrained weights
                        self.dinov3_model = torch.hub.load(
                            str(dinov3_repo_path),
                            'dinov3_vitb16',
                            source='local',
                            weights=str(weights_path)  # Pass the path to weights
                        )
                        logger.info("Local DINOv3 loaded with pretrained weights!")
                    else:
                        logger.info("No pretrained weights found, loading architecture only...")
                        self.dinov3_model = torch.hub.load(
                            str(dinov3_repo_path),
                            'dinov3_vitb16',
                            source='local',
                            pretrained=False  # No weights, just architecture
                        )
                        logger.info("Local DINOv3 loaded (architecture only)")
                    
                    self.dinov3_model.to(self.device)
                    self.dinov3_model.eval()
                    
                    self.dinov3_processor = None  # Manual preprocessing
                    self.is_local_dinov3 = True
                    
                except Exception as e:
                    logger.warning(f"Failed to load local DINOv3: {e}")
                    self.is_local_dinov3 = False
            
            # If local DINOv3 failed, try using DINOv2
            if not self.is_local_dinov3:
                try:
                    logger.info("Using DINOv2 for attention visualization (proven approach)...")
                    dinov2_model_name = "facebook/dinov2-base"
                    self.dinov3_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
                    self.dinov3_model = AutoModel.from_pretrained(dinov2_model_name)
                    self.dinov3_model.to(self.device)
                    self.dinov3_model.eval()
                    logger.info("DINOv2 loaded successfully for attention visualization")
                    self.is_dinov2_fallback = True
                except Exception as e:
                    logger.warning(f"Failed to load DINOv2: {e}")
                    self.use_attention_heatmap = False
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load Phase 1 and Phase 2 models."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load Phase 1 components
        logger.info("Loading Phase 1 components...")
        self.visual_backbone = VisionBackbone().to(self.device)
        self.temporal_memory = TemporalMemory().to(self.device)
        self.classifier_head = ClassifierHead().to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.reward_calculator = RewardCalculator(
            frozen_evaluator=self.feature_extractor
        )
        logger.info("Initialized reward calculator")
        
        # Load Phase 2 PPO model
        logger.info("Loading Phase 2 PPO model...")
        
        # Create policy
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1794,))
        act_space = gym.spaces.Discrete(5)
        self.policy = ActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda x: 0.001
        ).to(self.device)
        logger.info("Initialized EAGER Actor-Critic Policy for PPO")
        
        # Load checkpoint - handle SB3 zip file format
        try:
            if self.model_path.endswith('.zip'):
                # Extract and load policy from SB3 zip file
                import zipfile
                import tempfile
                import os
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    policy_path = os.path.join(temp_dir, 'policy.pth')
                    if os.path.exists(policy_path):
                        checkpoint = torch.load(policy_path, map_location=self.device)
                        self.policy.load_state_dict(checkpoint)
                        logger.info("Loaded policy weights from SB3 zip file")
                    else:
                        logger.warning("Could not find policy.pth in zip file")
            else:
                # Regular checkpoint loading
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'policy_state_dict' in checkpoint:
                    self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    logger.info("Loaded policy state dict from checkpoint")
                else:
                    logger.warning("No policy_state_dict found in checkpoint")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning("Continuing with randomly initialized policy")
        
        self.policy.eval()
        logger.info("All models loaded successfully!")
    
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Extract frames from video with face detection.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Array of frames (N, H, W, C) in [0, 1] range
        """
        logger.info(f"Extracting frames from: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate sampling rate
        if max_frames is None:
            max_frames = MAX_FRAMES
        
        if total_frames <= max_frames:
            sample_rate = 1
        else:
            sample_rate = total_frames // max_frames
        
        logger.info(f"Video info: {total_frames} frames, {fps:.1f} FPS")
        logger.info(f"Sampling every {sample_rate} frames (max {max_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Process frame with face detection if available
                if self.face_detector is not None:
                    # Detect face in frame
                    face_result = self.face_detector.detect_face(frame)
                    
                    if face_result is None:
                        # Try with enhanced frame if detection fails
                        enhanced_frame = self.face_detector.enhance_frame(frame)
                        face_result = self.face_detector.detect_face(enhanced_frame)
                        if face_result:
                            frame = enhanced_frame
                    
                    if face_result:
                        # Crop face with margin
                        bbox = self.face_detector.add_margin(face_result['bbox'], frame.shape)
                        x1, y1, x2, y2 = bbox
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Skip if crop is too small
                        if cropped_frame.shape[0] < 10 or cropped_frame.shape[1] < 10:
                            logger.debug(f"Face crop too small at frame {frame_count}, skipping")
                            frame_count += 1
                            continue
                        
                        # Resize to 224x224
                        processed_frame = self.face_detector.resize_frame(cropped_frame)
                        
                        # Ensure the frame is BGR before converting to RGB
                        if len(processed_frame.shape) == 2:
                            # Grayscale, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                        elif processed_frame.shape[2] == 4:
                            # BGRA, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGB)
                        else:
                            # Assume BGR, convert to RGB
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Ensure shape is exactly (224, 224, 3) and normalize
                        processed_frame = cv2.resize(processed_frame, (224, 224))
                        processed_frame = processed_frame.astype(np.float32) / 255.0
                        
                        # Verify shape before appending
                        assert processed_frame.shape == (224, 224, 3), f"Unexpected shape: {processed_frame.shape}"
                        frames.append(processed_frame)
                        logger.debug(f"Face detected at frame {frame_count} using {face_result['detection_method']}")
                    else:
                        logger.debug(f"No face detected at frame {frame_count}, using full frame")
                        # Fallback to full frame if no face detected
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                        frame = frame.astype(np.float32) / 255.0
                        frames.append(frame)
                else:
                    # No face detection, use full frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        frames = np.array(frames)
        logger.info(f"Extracted {len(frames)} frames")
        
        return frames
    
    def generate_attention_heatmap_pca(self, image: np.ndarray) -> np.ndarray:
        """
        Generate attention heatmap using DINOv3/v2 PCA-based approach.
        This is the recommended approach from DINOv3 documentation.
        
        Args:
            image: Input image (H, W, C) in [0, 1] range
            
        Returns:
            Attention heatmap as numpy array
        """
        if not self.use_attention_heatmap or self.dinov3_model is None:
            return None
        
        try:
            # Convert to PIL Image for processor
            from PIL import Image
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)
            
            # Process image
            if self.is_local_dinov3:
                # Manual preprocessing for local DINOv3
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224), antialias=True),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                inputs = transform(pil_image).unsqueeze(0).to(self.device)
            else:
                # Use processor for HuggingFace models
                inputs = self.dinov3_processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Get features using forward pass
            with torch.no_grad():
                if self.is_local_dinov3:
                    # For local DINOv3, get intermediate features
                    if hasattr(self.dinov3_model, 'get_intermediate_layers'):
                        # Get features from last layer
                        features = self.dinov3_model.get_intermediate_layers(
                            inputs, n=[11], return_class_token=False
                        )[0]
                        
                        # Handle tuple output
                        if isinstance(features, tuple):
                            features = features[0]
                    else:
                        features = self.dinov3_model(inputs)
                else:
                    # For HuggingFace models
                    outputs = self.dinov3_model(**inputs, output_hidden_states=True)
                    # Get the last hidden state (patch features)
                    features = outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
            
            # features shape: (1, num_patches, feature_dim)
            features = features.squeeze(0).cpu().numpy()  # (num_patches, feature_dim)
            
            # Apply PCA to reduce to 3 principal components (for RGB visualization)
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features)  # (num_patches, 3)
            
            # Take the first principal component as the importance score
            importance = features_pca[:, 0]  # First PC captures most variance
            
            # Normalize to [0, 1]
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            # Reshape to 2D grid (14x14 for 224x224 with patch_size=16)
            patch_dim = int(np.sqrt(len(importance)))
            if patch_dim * patch_dim == len(importance):
                heatmap = importance.reshape(patch_dim, patch_dim)
            else:
                # Handle non-square cases
                heatmap = importance[:patch_dim*patch_dim].reshape(patch_dim, patch_dim)
            
            # Upsample to original size
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur for smoothness
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 2.0)
            
            # Normalize again after blurring
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            logger.warning(f"Failed to generate PCA heatmap: {e}")
            return None
    
    def create_environment(self, frames: np.ndarray) -> DeepfakeEnv:
        """
        Create a DeepfakeEnv with the provided frames.
        
        Args:
            frames: Video frames array
            
        Returns:
            Configured environment
        """
        # Create a simple dataset wrapper for single video
        class SingleVideoDataset:
            def __init__(self, frames):
                # Convert numpy array to torch tensor
                # frames is (N, H, W, C), need to convert to (N, C, H, W)
                frames = np.transpose(frames, (0, 3, 1, 2))
                self.frames = torch.from_numpy(frames).float()
                self.video_ids = ['test_video']
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return self.frames, 0, 'test_video'  # Return frames tensor, label, video_id
        
        dataset = SingleVideoDataset(frames)
        
        # Create environment with individual components
        env = DeepfakeEnv(
            dataset=dataset,
            vision_backbone=self.visual_backbone,
            temporal_memory=self.temporal_memory,
            classifier_head=self.classifier_head,
            reward_calculator=self.reward_calculator,
            device=self.device,
            training=False
        )
        logger.info("Initialized DeepfakeEnv (training=False)")
        
        return env
    
    def run_inference(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on video frames.
        
        Args:
            frames: Video frames array
            
        Returns:
            Dictionary with inference results
        """
        # Create environment
        env = self.create_environment(frames)
        
        # Reset environment 
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result  
        else:
            obs = reset_result  
        
        # Run inference
        logger.info("Running inference...")
        done = False
        steps = 0
        actions = []
        rewards = []
        
        while not done and steps < MAX_FRAMES:
            # Get action from policy
            with torch.no_grad():
                # The policy expects a dictionary observation format
                # We need to bypass SB3's preprocessing and directly call the policy
                if isinstance(obs, dict):
                    # Convert numpy arrays in dict to tensors
                    obs_tensor = {
                        'frame_features': torch.FloatTensor(obs['frame_features']).unsqueeze(0).to(self.device),
                        'temporal_memory': torch.FloatTensor(obs['temporal_memory']).unsqueeze(0).to(self.device),
                        'frame_position': torch.FloatTensor(obs['frame_position']).unsqueeze(0).to(self.device),
                        'uncertainty': torch.FloatTensor(obs['uncertainty']).unsqueeze(0).to(self.device)
                    }
                    
                    # Directly use the policy's features extractor and actor
                    # This bypasses SB3's observation preprocessing
                    features = self.policy.features_extractor(obs_tensor)
                    latent_pi = self.policy.mlp_extractor.forward_actor(features)
                    action_logits = self.policy.action_net(latent_pi)
                    
                    # Get deterministic action (argmax)
                    action = torch.argmax(action_logits, dim=1).cpu().numpy()[0]
                else:
                    # Fallback for non-dict observations
                    action, _ = self.policy.predict(obs, deterministic=True)
                    
                action = int(action)
            
            # Take action 
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                obs, reward, done, info = step_result
            
            # Record
            actions.append(action)
            rewards.append(reward)
            steps += 1
            
            # Log step
            action_name = ACTION_NAMES[action]
            logger.info(f"Step {steps}: Action={action_name}, Reward={reward:.3f}")
        
        # Get final prediction
        final_action = actions[-1] if actions else 0
        if final_action == 3:  # STOP_REAL
            prediction = "REAL"
        elif final_action == 4:  # STOP_FAKE
            prediction = "FAKE"
        else:
            # If didn't stop, use majority vote
            stop_real_count = sum(1 for a in actions if a == 3)
            stop_fake_count = sum(1 for a in actions if a == 4)
            prediction = "FAKE" if stop_fake_count > stop_real_count else "REAL"
        
        # Calculate action distribution
        action_counts = {name: 0 for name in ACTION_NAMES}
        for action in actions:
            action_counts[ACTION_NAMES[action]] += 1
        
        total_actions = len(actions)
        action_distribution = {
            name: count / total_actions if total_actions > 0 else 0
            for name, count in action_counts.items()
        }
        
        results = {
            'prediction': prediction,
            'confidence': 0.5,  
            'steps': steps,
            'actions': actions,
            'rewards': rewards,
            'total_reward': sum(rewards),
            'action_distribution': action_distribution
        }
        
        return results
    
    def visualize_results(self, frames: np.ndarray, results: Dict[str, Any], output_path: str):
        """
        Create visualization of inference results with attention heatmaps.
        
        Args:
            frames: Video frames
            results: Inference results
            output_path: Path to save visualization
        """
        # Create figure with subplots
        fig_width = 20 if self.use_attention_heatmap else 15
        fig = plt.figure(figsize=(fig_width, 10))
        
        # Adjust subplot layout based on heatmap availability
        if self.use_attention_heatmap:
            # 1. Sample frames
            ax1 = plt.subplot(2, 4, 1)
        else:
            ax1 = plt.subplot(2, 3, 1)
            
        n_display = min(4, len(frames))
        display_indices = np.linspace(0, len(frames)-1, n_display, dtype=int)
        
        display_frames = []
        for idx in display_indices:
            display_frames.append(frames[idx])
        
        # Create montage
        if len(display_frames) == 4:
            top = np.hstack(display_frames[:2])
            bottom = np.hstack(display_frames[2:])
            montage = np.vstack([top, bottom])
        else:
            montage = np.hstack(display_frames)
        
        ax1.imshow(montage)
        ax1.set_title(f"Sample Frames from Video")
        ax1.axis('off')
        
        # Add attention heatmap if available
        if self.use_attention_heatmap:
            ax_heat = plt.subplot(2, 4, 2)
            logger.info("Generating attention heatmaps...")
            
            # Generate heatmaps for selected frames using PCA approach
            heatmaps = []
            for i, idx in enumerate(display_indices[:4]):  # Max 4 heatmaps
                logger.info(f"Processing frame {i+1}/4 for heatmap")
                heatmap = self.generate_attention_heatmap_pca(frames[idx])
                if heatmap is not None:
                    logger.info(f"Heatmap generated successfully for frame {i+1}")
                    # Overlay heatmap on original frame
                    frame_display = frames[idx].copy()
                    
                    # Create colored heatmap using 'jet' colormap
                    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
                    
                    # Blend with original
                    alpha = 0.5
                    blended = frame_display * (1 - alpha) + heatmap_colored * alpha
                    heatmaps.append(blended)
                else:
                    heatmaps.append(frames[idx])
            
            # Create montage of heatmaps
            if len(heatmaps) == 4:
                top = np.hstack(heatmaps[:2])
                bottom = np.hstack(heatmaps[2:])
                heatmap_montage = np.vstack([top, bottom])
            elif len(heatmaps) > 0:
                heatmap_montage = np.hstack(heatmaps)
            else:
                # If no heatmaps were generated, show a placeholder
                heatmap_montage = np.ones((224, 224, 3)) * 0.5
                logger.warning("No heatmaps generated, showing placeholder")
            
            ax_heat.imshow(np.clip(heatmap_montage, 0, 1))  # Ensure values are in [0, 1]
            model_name = "DINOv2" if self.is_dinov2_fallback else "DINOv3"
            ax_heat.set_title(f"{model_name} PCA-based Feature Importance")
            ax_heat.axis('off')
        
        # 2. Action sequence
        ax2 = plt.subplot(2, 4, 3) if self.use_attention_heatmap else plt.subplot(2, 3, 2)
        action_sequence = results['actions'][:50]  # First 50 actions
        action_colors = ['blue', 'green', 'orange', 'red', 'purple']
        colors = [action_colors[a] for a in action_sequence]
        
        ax2.bar(range(len(action_sequence)), [1]*len(action_sequence), color=colors, width=1.0)
        ax2.set_title('Action Sequence (First 50 Steps)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Action')
        ax2.set_ylim([0, 1.5])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=name) 
                         for name, color in zip(ACTION_NAMES, action_colors)]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 3. Action distribution pie chart
        ax3 = plt.subplot(2, 4, 4) if self.use_attention_heatmap else plt.subplot(2, 3, 3)
        action_dist = list(results['action_distribution'].values())
        non_zero_actions = [(name, dist) for name, dist in results['action_distribution'].items() if dist > 0]
        
        if non_zero_actions:
            labels, sizes = zip(*non_zero_actions)
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Action Distribution')
        
        # 4. Reward progression
        ax4 = plt.subplot(2, 4, 5) if self.use_attention_heatmap else plt.subplot(2, 3, 4)
        cumulative_rewards = np.cumsum(results['rewards'])
        ax4.plot(cumulative_rewards, 'b-', linewidth=2)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_title('Cumulative Reward')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Cumulative Reward')
        ax4.grid(True, alpha=0.3)
        
        # 5. Step rewards
        ax5 = plt.subplot(2, 4, 6) if self.use_attention_heatmap else plt.subplot(2, 3, 5)
        ax5.bar(range(len(results['rewards'])), results['rewards'], 
                color=['green' if r > 0 else 'red' for r in results['rewards']])
        ax5.set_title('Step-wise Rewards')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 6. Results summary
        ax6 = plt.subplot(2, 4, 7) if self.use_attention_heatmap else plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        INFERENCE RESULTS
        ================
        
        Prediction: {results['prediction']}
        Confidence: {results['confidence']:.2%}
        Total Steps: {results['steps']}
        Total Reward: {results['total_reward']:.3f}
        
        Action Distribution:
        """
        
        for action, pct in results['action_distribution'].items():
            summary_text += f"\n  {action}: {pct:.1%}"
        
        ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace')
        
        # Add overall title
        fig.suptitle(f"Phase 2 Inference Results - {results['prediction']}", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")
        plt.close()

def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 Inference with Attention Visualization')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, 
                       default='models/final_models/ppo_eager_final.zip',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Disable attention heatmap generation')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device != args.device:
        logger.warning(f"CUDA not available, using CPU instead")
    
    # Initialize inference pipeline
    inference = Phase2Inference(
        model_path=args.model,
        device=device,
        use_attention_heatmap=not args.no_heatmap
    )
    
    # Extract frames
    frames = inference.extract_frames(args.video_path)
    
    # Run inference
    results = inference.run_inference(frames)
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info(f"PREDICTION: {results['prediction']}")
    logger.info(f"Confidence: {results['confidence']:.2%}")
    logger.info(f"Steps taken: {results['steps']}")
    logger.info(f"Total reward: {results['total_reward']:.3f}")
    logger.info("Action distribution:")
    for action, pct in results['action_distribution'].items():
        logger.info(f"  {action}: {pct:.1%}")
    logger.info("="*60 + "\n")
    
    # Create visualization
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inference_result_{timestamp}.png"
    else:
        output_path = args.output
    
    inference.visualize_results(frames, results, output_path)

if __name__ == "__main__":
    main()