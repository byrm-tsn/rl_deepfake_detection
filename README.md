# EAGER - Efficient Adaptive Gated Evidence Retrieval for Deepfake Detection using Reinforcement Learning

## Overview

EAGER is an advanced deepfake detection system that leverages reinforcement learning with DINOv3 vision transformer and custom Group Relative Policy Optimization (GRPO). The framework achieves high accuracy while processing only 20% of video frames, deployed through a Django web interface for real-time detection.


## Demo

Watch the demo video below to see the application in action:

https://github.com/user-attachments/assets/12063873-db85-49fc-8e0b-eb6f60f00c14


## Features

- **Intelligent Frame Selection**: Reinforcement learning agent analyzes only essential frames, reducing processing time to 10 seconds per video
- **User-Friendly Interface**: User-friendly Django application for video upload and analysis
- **Three-Phase Training**: CProgressive optimization through supervised warm-start, PPO-LSTM, and GRPO fine-tuning
- **Datasets Used**: FaceForensics++, Celeb-DF v2, Celeb-DF, DeeperForensics 1.0 and DFD (Google/Jigsaw) 
- **Attention Visualization**: DINOv3 heatmaps showing facial regions contributing to detection decisions
- **GPU Acceleration**: Utilizes PyTorch-GPU for faster processing
  

## Documentation

If you want to read my report, you can find it 

## Usage

- **Upload a video file through the web interface**
- **Wait for processing (approximately 10 seconds)**
- **View detection results with confidence scores**
- **Examine DINOv3 attention heatmaps showing detection reasoning**

## Model Architecture

- **Feature Extraction**: DINOv3 Vision Transformer (ViT-B/16)
- **Temporal Processing**: Bidirectional LSTM (3 layers, 512 hidden units)
- **Decision Making**: Reinforcement Learning Agent with PPO and GRPO
- **Uncertainty Estimation**: Bayesian inference via Monte Carlo Dropout

## Installation

### Prerequisites

- **Python 3.8+**
- **Django 4.2+**
- **PyTorch 2.0+**
- **TorchRL**
- **CUDA 11.8+**
- **cuDNN 8.6+**

### Technologies Used

[![My Skills](https://skillicons.dev/icons?i=vscode,github,django,js,html,css,git,opencv,py,sqlite,pytorch,sklearn)](https://skillicons.dev)


