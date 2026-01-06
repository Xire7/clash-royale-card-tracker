import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime


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

            if os.path.exists(gold_dir):
                gold_images = glob.glob(os.path.join(gold_dir, '*.png'))[:gold_limit] # only gold_limit num of images
                for img_path in gold_images:
                    self.samples.append((img_path, class_idx))
            
            if os.path.exists(normal_dir):
                normal_images = glob.glob(os.path.join(normal_dir, '*.png'))[:normal_limit] # only normal_limit num of images
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