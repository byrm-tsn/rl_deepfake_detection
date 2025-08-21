"""
Central configuration file for EAGER algorithm implementation.
All hyperparameters, paths, and system settings are centralized here.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent 
DATA_ROOT = PROJECT_ROOT / "data"

# =====================================================================
# GRPO CONFIGURATION (Phase 3)
# =====================================================================

GRPO_ITERATIONS = 50  # Number of GRPO iterations (reduced to prevent overfitting)
GRPO_GROUP_SIZE = 16  # Group size for relative reward computation 
GRPO_LEARNING_RATE = 5e-6  # Very low LR for fine-tuning 
GRPO_BUFFER_SIZE = 10000  # Experience buffer size
GRPO_EPISODES_PER_ITER = 32  # Episodes to collect per iteration
GRPO_UPDATES_PER_ITER = 2  # Updates per iteration
GRPO_GRADIENT_CLIP = 0.5  # Gradient clipping value

# =====================================================================
# DATA CONFIGURATION
# =====================================================================

# Data paths (must match user's directory structure)
DATA_DIR = DATA_ROOT / "processed_data"  
BALANCED_DATASET_PATH = DATA_ROOT / "balanced_dataset"
PROCESSED_DATASET_PATH = DATA_ROOT / "processed_data"
METADATA_BALANCED = BALANCED_DATASET_PATH / "metadata.csv"
METADATA_PROCESSED = PROCESSED_DATASET_PATH / "metadata.csv"

# Frame processing parameters
FRAME_SIZE = 224  
FRAMES_PER_VIDEO = 50  
CHANNELS = 3 

# Data loading parameters
BATCH_SIZE = 4  
NUM_WORKERS = 8
PIN_MEMORY = True  

# =====================================================================
# MODEL ARCHITECTURE PARAMETERS
# =====================================================================

# Vision backbone configuration
USE_DINOV3 = True 
VISION_BACKBONE = "vit_base_patch16_224.augreg_in21k_ft_in1k"  
VIT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"  
DINOV3_WEIGHTS_PATH = PROJECT_ROOT / "models" / "dinov3_weights" / "dinov3_vitb16_pretrain_lvd1689m.pth"
VISION_EMBEDDING_DIM = 768  # Output dimension from ViT (same for DINOv3 ViT-B)
FREEZE_BACKBONE = True  # Freeze during RL training

# LSTM temporal memory configuration
LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYERS = 3  
LSTM_DROPOUT = 0.2
LSTM_BIDIRECTIONAL = True
LSTM_EFFECTIVE_DIM = LSTM_HIDDEN_DIM * 2 

# Policy network configuration
POLICY_HIDDEN_DIM = 256
VALUE_HIDDEN_DIM = 256

POLICY_DROPOUT = 0.1

# Classifier head configuration
CLASSIFIER_HIDDEN_DIM = 512  
CLASSIFIER_DROPOUT = 0.3  

# State representation dimensions 
OBSERVATION_COMPONENTS = {
    'frame_features': VISION_EMBEDDING_DIM, 
    'temporal_memory': LSTM_EFFECTIVE_DIM,   
    'frame_position': 1,
    'uncertainty': 1
}

# Calculate total dimension from components
STATE_DIM = sum(OBSERVATION_COMPONENTS.values()) 
# Components: frame_features(768) + temporal_memory(1024) + position(1) + uncertainty(1)

# =====================================================================
# UNCERTAINTY ESTIMATION (Bayesian Approximation)
# =====================================================================

# Enable Bayesian uncertainty estimation using Monte Carlo Dropout (MCDO)
# If True, uncertainty is Normalized Predictive Entropy. If False, it is (1 - confidence).
USE_BAYESIAN_UNCERTAINTY = True 

# Number of Monte Carlo samples during inference. 
MC_SAMPLES = 20 

# =====================================================================
# REINFORCEMENT LEARNING PARAMETERS
# =====================================================================

# Action space
NUM_ACTIONS = 5
ACTION_NAMES = ["NEXT", "FOCUS", "AUGMENT", "STOP_REAL", "STOP_FAKE"]

# PPO hyperparameters
PPO_LEARNING_RATE = 1e-4  
PPO_N_STEPS = 2048  
PPO_BATCH_SIZE = 512  
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.02  
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Parallel environment settings 
NUM_PARALLEL_ENVS = 8  
USE_MIXED_PRECISION = True  

# Training configuration
TOTAL_TIMESTEPS = 1500000 
EVAL_FREQ = 20000 
SAVE_FREQ = 100000
LOG_INTERVAL = 100

# =====================================================================
# REWARD SYSTEM PARAMETERS
# =====================================================================

# Terminal rewards
CORRECT_CLASSIFICATION_REWARD = 15.0  
INCORRECT_CLASSIFICATION_PENALTY = -10.0  
NO_DECISION_PENALTY = -25.0  

# Step costs
NEXT_ACTION_COST = -0.5 
FOCUS_ACTION_COST = -0.55 
AUGMENT_ACTION_COST = -0.6  
EARLY_DECISION_BONUS = 2  

# Information gain bonus
MAX_INFO_GAIN_BONUS = 1.5  
UNCERTAINTY_THRESHOLD = 0.35  
INFO_GAIN_THRESHOLD = 0.025 

# Confidence gating
CONFIDENCE_THRESHOLD = 0.80 
MIN_FRAMES_BEFORE_DECISION = 7 
CONFIDENCE_GATE_PENALTY = -2 

# Curriculum learning for confidence 
CONFIDENCE_CURRICULUM_START = 0.55  
CONFIDENCE_CURRICULUM_END = 0.80   
CONFIDENCE_CURRICULUM_STEPS = 200000 

# Episode limits
MAX_EPISODE_STEPS = 75  
MAX_FRAMES = 50  
OPTIMAL_DECISION_STEPS = 25 


# Observation dimension for the 2-phase system
OBSERVATION_DIM = 1794  

# =====================================================================
# TRAINING PHASES
# =====================================================================

# Phase 1: Supervised warm-start 
WARMSTART_EPOCHS = 10 
WARMSTART_LR = 1e-5  
WARMSTART_MIN_LR = 1e-7  
WARMSTART_BATCH_SIZE = 16 
WARMSTART_GRADIENT_ACCUMULATION = 2 
WARMSTART_WEIGHT_DECAY = 5e-5  
WARMSTART_SCHEDULER_PATIENCE = 5  
WARMSTART_SCHEDULER_FACTOR = 0.5  
WARMSTART_EARLY_STOPPING_PATIENCE = 5 

# Enhanced regularization for overfitting
WARMSTART_LABEL_SMOOTHING = 0.0
WARMSTART_MIXUP_ALPHA = 0.0 
WARMSTART_CUTMIX_ALPHA = 0.0  
WARMSTART_LSTM_DROPOUT = 0.4 

# Learning rate scheduling
USE_COSINE_ANNEALING = False 
USE_REDUCE_ON_PLATEAU = True  
WARMUP_EPOCHS = 1 
WARMUP_FACTOR = 0.1  
USE_ADAMW = True  

# Focal Loss for class imbalance
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.65 
FOCAL_GAMMA = 2.0  

# Architecture improvements 
UNFREEZE_VIT_LAYERS = 4 
USE_ATTENTION_POOLING = True  
ATTENTION_HEADS = 8 
ATTENTION_DIM = 1024 
CLASSIFIER_LAYERS = 2  
USE_RESIDUAL_CLASSIFIER = True 

# Phase 2: RL training
RL_TRAINING_EPISODES = 20000

# =====================================================================
# PATHS AND DIRECTORIES
# =====================================================================

# Model storage
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINAL_MODEL_DIR = MODEL_DIR / "final_models"
LOG_DIR = PROJECT_ROOT / "logs" 
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
TRAINING_LOG_DIR = LOG_DIR / "training_logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, CHECKPOINT_DIR, FINAL_MODEL_DIR, 
                  LOG_DIR, TENSORBOARD_DIR, TRAINING_LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DEVICE CONFIGURATION
# =====================================================================

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DETERMINISTIC = True
SEED = 42

# Check GPU availability
if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using GPU: {GPU_NAME} with {GPU_MEMORY:.2f} GB memory")
else:
    print("WARNING: CUDA not available, using CPU. Training will be slow.")

# =====================================================================
# DATA AUGMENTATION PARAMETERS
# =====================================================================

# Augmentation for AUGMENT action
AUGMENT_BRIGHTNESS_RANGE = 0.2  
AUGMENT_JPEG_QUALITY_MIN = 70
AUGMENT_JPEG_QUALITY_MAX = 95
AUGMENT_ROTATION_RANGE = 5  

# Training data augmentation 
TRAIN_AUGMENTATION_PROB = 0.7 
TRAIN_HORIZONTAL_FLIP_PROB = 0.5  
TRAIN_COLOR_JITTER_STRENGTH = 0.2 
TRAIN_JPEG_COMPRESSION = True  
TRAIN_BRIGHTNESS_CONTRAST = True  

# =====================================================================
# LOGGING AND DEBUGGING
# =====================================================================

LOG_LEVEL = "INFO"
VERBOSE = True 
DEBUG_MODE = False
PROGRESS_BAR_WIDTH = 120  
SAVE_EPISODE_TRAJECTORIES = True
MAX_TRAJECTORIES_TO_SAVE = 100

# =====================================================================
# EVALUATION PARAMETERS
# =====================================================================

EVAL_EPISODES = 100
EVAL_DETERMINISTIC = True
GENERATE_CONFUSION_MATRIX = True
GENERATE_ROC_CURVE = True

# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================

EXPERIMENT_NAME = "eager_deepfake_detection_bayesian"
EXPERIMENT_DESCRIPTION = "EAGER algorithm for intelligent deepfake detection with Bayesian Uncertainty"

def get_config() -> Dict[str, Any]:
    """
    Returns all configuration parameters as a dictionary.
    Useful for logging and experiment tracking.
    """
    return {
        "experiment_name": EXPERIMENT_NAME,
        "vision_backbone": VISION_BACKBONE,
        "lstm_hidden_dim": LSTM_HIDDEN_DIM,
        "ppo_learning_rate": PPO_LEARNING_RATE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "reward_correct": CORRECT_CLASSIFICATION_REWARD,
        "reward_incorrect": INCORRECT_CLASSIFICATION_PENALTY,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "use_bayesian_uncertainty": USE_BAYESIAN_UNCERTAINTY,
        "mc_samples": MC_SAMPLES,
        "device": DEVICE,
        "seed": SEED,
    }

def validate_config():
    """
    Validates configuration parameters and checks system requirements.
    """
    # Check data directories exist
    if not BALANCED_DATASET_PATH.exists():
        print(f"WARNING: Balanced dataset path does not exist: {BALANCED_DATASET_PATH}")
    
    if not PROCESSED_DATASET_PATH.exists():
        print(f"WARNING: Processed dataset path does not exist: {PROCESSED_DATASET_PATH}")
    
    # Check GPU requirements
    if DEVICE == "cuda" and GPU_MEMORY < 24:
        print(f"WARNING: GPU memory ({GPU_MEMORY:.2f} GB) is less than recommended 24 GB")
    
    # Validate hyperparameters
    assert 0 < PPO_LEARNING_RATE < 1, "Invalid learning rate"
    assert 0 < CONFIDENCE_THRESHOLD <= 1, "Invalid confidence threshold"
    assert MIN_FRAMES_BEFORE_DECISION > 0, "Must analyze at least one frame"
    assert FRAMES_PER_VIDEO == 50, "Implementation expects exactly 50 frames per video"
    
    if USE_BAYESIAN_UNCERTAINTY:
        assert MC_SAMPLES > 1, "MC_SAMPLES must be greater than 1 for Bayesian uncertainty"

    print("Configuration validation completed.")

# Run validation on import
if __name__ != "__main__":
    validate_config()