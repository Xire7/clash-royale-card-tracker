import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
from carddataset import CardDataset

CONFIG = {
    'data_dir': 'data',
    'model_save_dir': 'models',
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'image_size': 224,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # image limits per subfolder
    'train_gold_per_class': 160,
    'train_normal_per_class': 40,
    'val_gold_per_class': 14,
    'val_normal_per_class': 6,

    'model_type': 'resnet18' # can be 'resnet18', 'resnet34' or 'mobilenet_v2'
}

# card classes (folder names)
CLASSES = ['cannon', 'hog_rider', 'ice_golem', 'ice_spirit', 'musketeer', 'skeletons'] 


os.makedirs(CONFIG['model_save_dir'], exist_ok=True)


## data loading for training HERE

train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CardDataset(
    root_dir = os.path.join(CONFIG['data_dir'], 'train'),
    classes = CLASSES,
    gold_limit=CONFIG['train_gold_per_class'],
    normal_limit=CONFIG['train_normal_per_class'],
    transform=train_transform
)

val_dataset = CardDataset(
    root_dir = os.path.join(CONFIG['data_dir'], 'val'),
    classes = CLASSES,
    gold_limit = CONFIG['val_gold_per_class'],
    normal_limit = CONFIG['val_normal_per_class'],
    transform = val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = CONFIG['batch_size'],
    shuffle = True,
    num_workers = CONFIG['num_workers']
)

val_loader = DataLoader(
    val_dataset,
    batch_size = CONFIG['batch_size'],
    shuffle = False,
    num_workers = CONFIG['num_workers']
)


print("=========== DATASET INFO (this is why we clash) =============")
print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes: {CLASSES}")
print(f"Number of classes: {len(CLASSES)}")

print("\nTraining set distribution:")
train_dist = train_dataset.get_class_distribution()
for class_name, count in train_dist.items():
    print(f"  {class_name}: {count} images")

print("\nValidation set distribution:")
val_dist = val_dataset.get_class_distribution()
for class_name, count in val_dist.items():
    print(f"  {class_name}: {count} images")

# load pretrained mobilenetv2

model = models.mobilenet_v2(pretrained=True)

for param in model.features.parameters():
    param.requires_grad = False

num_classes = len(CLASSES)

# loading model types

if CONFIG['model_type'] == 'resnet18':
    model = models.resnet18(pretrained=True)

    # freeze early layers

    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze layer4, the last conv block
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # replace final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    trainable_params = list(model.layer4.parameters()) + list(model.fc.parameters())

elif CONFIG['model_type'] == 'resnet34':
    model = models.resnet34(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    trainable_params = list(model.layer4.parameters()) + list(model.fc.parameters())

else: # mobilenet_v2
    model = models.mobilenet_v2(pretrained=True)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    trainable_params = model.classifier.parameters()

# move to device
model = model.to(CONFIG['device'])

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(trainable_params, lr=CONFIG['learning_rate'])

# learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


## training functions HERE

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # stats
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()


        # print progress
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.4f}")
        

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # per class accuracy
    num_classes = len(classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # per class stats
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    # per class acc
    class_acc = {}
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            class_acc[class_name] = 100 * class_correct[i] / class_total[i]
        else:
            class_acc[class_name] = 0.0
    
    return epoch_loss, epoch_acc, class_acc


## training loop HERE

print("=" * 60)
print("Starting training...")
print("=" * 60)

best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_model = {}

model_dir = os.path.join(CONFIG['model_save_dir'], datetime.now().strftime("%Y%m%d_%H%M"))
os.makedirs(model_dir, exist_ok=True)

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
    print("-" * 60)

    # train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, CONFIG['device']
    )

    # validate
    val_loss, val_acc, class_acc = validate(
        model, val_loader, criterion, CONFIG['device'], CLASSES
    )

    scheduler.step()

    # record metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)


    # print results
    print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"\n  Per-class validation accuracy:")
    for class_name, acc in class_acc.items():
        print(f"    {class_name}: {acc:.2f}%")
    

    # save current best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        best_model = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': CLASSES,
            'config': CONFIG,
        }

        print(f"\n new model saved: (VAL ACC: {val_acc:.2f}%)")
    print("\n" + "=" * 60)

best_model_path = os.path.join(model_dir, "best_model.pth")

torch.save(best_model, best_model_path)

print(f"Training complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# loss plot
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# accuracy plot
ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(val_accs, label='Val Acc', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plot_path = os.path.join(CONFIG['model_save_dir'], 'training_curves.png')
plt.savefig(plot_path, dpi=150)
print(f"\nTraining curves saved to: {plot_path}")
plt.show()
