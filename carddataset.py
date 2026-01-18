import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime


# in my dataset, want to include 70 hero musketeers
MUSKETEER_HERO_COUNT = 70

# Ideal Train: 160G/40N/50E OR 185G/65N OR 90G/70H/40N/50E

EVO_COUNT = 50

LIMIT_BOOST = 25 # card with no evos need to sum to 250

class CardDataset(Dataset):
    def __init__(self, root_dir, classes, gold_limit, normal_limit, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)} # assign index to each class
        self.transform = transform
        self.samples = []

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            gold_dir = os.path.join(class_dir, 'gold')
            normal_dir = os.path.join(class_dir, 'normal')
            evo_dir = os.path.join(class_dir, 'evo')
            hero_dir = os.path.join(class_dir, 'hero')

            if class_name == "musketeer":
                hero_images = glob.glob(os.path.join(hero_dir, '*.png'))[:MUSKETEER_HERO_COUNT]
                for img_path in hero_images:
                    self.samples.append((img_path, class_idx))

                evo_images = glob.glob(os.path.join(evo_dir, '*.png'))[:EVO_COUNT]
                for img_path in evo_images:
                    self.samples.append((img_path, class_idx))
                
                modified_gold_limit = gold_limit - MUSKETEER_HERO_COUNT
                gold_images = glob.glob(os.path.join(gold_dir, '*.png'))[:modified_gold_limit] # subtract hero number
                for img_path in gold_images:
                    self.samples.append((img_path, class_idx))

                normal_images = glob.glob(os.path.join(normal_dir, '*.png'))[:normal_limit]
                for img_path in normal_images:
                    self.samples.append((img_path, class_idx))

            elif class_name == "cannon" or class_name == "skeletons":
                evo_images = glob.glob(os.path.join(evo_dir, '*.png'))[:EVO_COUNT]
                for img_path in evo_images:
                    self.samples.append((img_path, class_idx))
                
                gold_images = glob.glob(os.path.join(gold_dir, '*.png'))[:gold_limit]
                for img_path in gold_images:
                    self.samples.append((img_path, class_idx))

                normal_images = glob.glob(os.path.join(normal_dir, '*.png'))[:normal_limit]
                for img_path in normal_images:
                    self.samples.append((img_path, class_idx))
            
            else: # hog rider, ice golem, ice spirit
                gold_images = glob.glob(os.path.join(gold_dir, '*.png'))[:gold_limit + LIMIT_BOOST]
                for img_path in gold_images:
                    self.samples.append((img_path, class_idx))

                normal_images = glob.glob(os.path.join(normal_dir, '*.png'))[:normal_limit + LIMIT_BOOST]
                for img_path in normal_images:
                    self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} samples from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # load image
        image = Image.open(img_path).convert('RGB')

        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution