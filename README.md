# clash-royale-card-tracker
CNN-based opponent card tracker for clash royale

## Overview

This project detects when opponents deploy troops in Clash Royale by:
1. **OpenCV Clock Detection** - Identifies deployment clocks (white center + small red fill)
2. **CNN Classification** - Classifies the deployed troop using a trained MobileNetV2 model
3. **Real-time Overlay** - Displays detections with persistent bounding boxes

**Current Accuracy:** ~85-90% (6 card classes)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional but recommended for training)
- CUDA 12.8+ (if using GPU)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/clash-royale-card-tracker
cd clash-royale-card-tracker

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```


## Usage

### 1. Train the Model

Trains a MobileNetV2 classifier on your collected troop images.
```bash
python train.py
```

**What it does:**
- Loads images from `data/train/` and `data/val/`
- Trains for 25 epochs
- Saves best model to `models/best_model_TIMESTAMP.pth`
- Generates training curves plot


### 2. Run Real-time Detection

Detects troop deployments in real-time from screen capture.
```bash
# With trained model (classification enabled)
python detector.py --model models/best_model_20250107_123456.pth

# Without model (clock detection only)
python detector.py

# Select game window interactively
python detector.py --model models/best_model_*.pth --select-region

# Disable saving detections
python detector.py --model models/best_model_*.pth --no-save
```

**Controls:**
- `SPACE` - Pause/Resume
- `S` - Manual screenshot save
- `D` - Toggle debug windows
- `Q` - Quit


## Dataset Collection Guide

### Image Requirements

**Per card class:** 100 images total
- **Training:** 56 gold + 24 normal (80 total)
- **Validation:** 14 gold + 6 normal (20 total)

**Timing:** Screenshot during **drop animation** (0.1-0.3s after deployment)
- Troop is elongated/stretched
- Clock is visible but mostly white with small red
- Troop is centered and clearly visible

**Folder structure:**
```
data/train/cannon/
  ├── gold/      # Gold cosmetic troops (56 images)
  └── normal/    # Normal color troops (24 images)
```

### Tips for Collection

1. **Use detector.py** to auto-save detections while playing
2. **Watch replays** for easier data collection
3. **Variety:** Different arenas, game states, angles
4. **Consistency:** Same framing (troop centered, clock at bottom)
5. **70/30 gold/normal split** reflects real player preferences


## Next Steps

### Short-term Improvements

1. **Switch to ResNet18** (better accuracy)
```python
   # In train.py, replace MobileNetV2:
   model = models.resnet18(pretrained=True)
   # Freeze early layers, train final layers
```
   **Expected gain:** +5-10% accuracy

2. **Collect More Data**
   - Target: 150-200 images per class
   - Focus on gold cosmetics (more common in practice)
   - **Expected gain:** +10-15% accuracy

3. **Optimize Transforms**
```python
   # Try direct resize (no random crop) for centered troops:
   transforms.Resize(224)  # Instead of Resize(256) + RandomCrop(224)
```

4. **Fine-tune Hyperparameters**
   - Increase epochs: 40 instead of 25
   - Try AdamW optimizer with weight decay
   - Experiment with learning rate schedules

### Future Features

- **Fireball/Log Detection** - Separate CV pipeline (orange/brown detection)
- **Card Cycle Counter** - Track when specific card returns to hand (4 placements)
- **Multi-model Ensemble** - Combine MobileNet + ResNet predictions
- **Evolution/Hero Variants** - Expand to handle special card variants

### Known Limitations

- **Not viable for elixir tracking** (requires 100% accuracy, CNN can't guarantee this)
- **Gold cosmetics** can confuse clock detection (addressed with white-center detection)
- **Fast troops** (skeletons, ice spirit) may move out of detection region quickly
- **Overlapping deployments** can cause missed classifications

---

## Technical Details

### Model Architecture
- **Backbone:** MobileNetV2 (pretrained on ImageNet)
- **Frozen:** Feature extraction layers
- **Trainable:** Final classification layer (6 classes)
- **Input:** 224×224 RGB images
- **Output:** 6-class softmax probabilities

### Clock Detection Pipeline
1. Convert frame to HSV color space
2. Threshold for white pixels (clock center)
3. Morphological operations to clean noise
4. Find contours with area/aspect/circularity filters
5. Check for small red presence (5-20% of region)
6. Extract troop region (170×170 centered on clock, extends upward)

### Inference Speed
- **Clock detection:** ~2ms per frame (OpenCV)
- **CNN classification:** ~10-30ms (CPU), ~2-5ms (GPU)
- **Total pipeline:** ~15-35ms per frame (30-60 FPS capable)
