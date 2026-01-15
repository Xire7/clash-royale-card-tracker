# clash-royale-card-tracker
CNN (MobileNet) & OpenCV based opponent card tracker for clash royale

![Clash Royale Card Tracker Thumbnail](images/cycle_thumbnail.png)

### hard thumbnail by @teka._.ten instagram

## Why does this exist?

If you've ever lost a game of Clash Royale because you got outcycled by an opponent, or at a critical moment you placed your win condition and they had the perfect card to counter it on the right time, this cycle tracker is for you!

This will track where the opponent's cards are in their deck so you know when they'll have it up or not. 

Instead of getting gud, just plug and play this application to a BlueStacks Clash Royale emulator and get infinite LP lol

## Important Disclaimer: MVP Stage

This card tracker only detects Hog 2.6 at the moment ! Clash Royale has 100 deployable troop cards, assuming 200 images per troop class to allow the CNN to learn how the card looks when placed, this would mean I would have to collect 20,000 images by hand. I am not doing allat rn

**Current Accuracy:** ~99% accuracy (6 card classes)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional but recommended for training)
- CUDA 12.8+ (if using GPU)


## Usage

### 1. Train the Model

Trains a MobileNetV2 classifier on collected troop images.
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


## Dataset Collection Notes

### Current Image Requirements

**Per card class:** 200 images total
- **Training:** 160 gold + 40 normal (200 total)
- **Validation:** 14 gold + 6 normal (20 total)

**Timing:** Screenshot during **drop animation** (0.1-0.3s after deployment)
- Troop is elongated/stretched
- Clock is visible but mostly white with small red
- Troop is centered and clearly visible

**Folder structure:**
```
data/train/cannon/
  ├── gold/      # Gold cosmetic troops (160 images)
  └── normal/    # Normal color troops (40 images)
```

## Model Architecture
- **Backbone:** MobileNetV2 (pretrained on ImageNet)
- **Frozen:** Feature extraction layers
- **Trainable:** Final classification layer (6 classes)
- **Input:** 224×224 RGB images
- **Output:** 6-class softmax probabilities