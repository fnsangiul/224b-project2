import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

class ProstateDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

df = pd.read_csv("train.csv")

train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_df, val_df = random_split(df, [train_size, val_size])

train_dataset = ProstateDataset(train_df.dataset.iloc[train_df.indices], '.', transform=train_transform)
val_dataset = ProstateDataset(val_df.dataset.iloc[val_df.indices], '.', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_f1 = 0.0
best_epoch = 0

train_losses, val_f1s, val_accs = [], [], []

for epoch in trange(15, desc="Training Epochs"):
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", leave=False):
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend((probs > 0.5).astype(int))
            targets.extend(labels.numpy())

    val_f1 = f1_score(targets, preds)
    val_acc = accuracy_score(targets, preds)
    val_f1s.append(val_f1)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={val_f1:.4f}, Acc={val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(results_dir, "best_efficientnet_model.pt"))

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_f1s, label="Validation F1")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "efficientnet_training_metrics.png"))
plt.close()

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_f1": val_f1s,
    "val_accuracy": val_accs
})
metrics_df.to_csv(os.path.join(results_dir, "efficientnet_metrics.csv"), index=False)

# Save best epoch separately for report
with open(os.path.join(results_dir, "best_epoch.txt"), "w") as f:
    f.write(f"Best epoch: {best_epoch}\nF1: {best_f1:.4f}\n")

print("Metrics and best epoch saved!")

# Inference
model.load_state_dict(torch.load(os.path.join(results_dir, "best_efficientnet_model.pt")))
model.eval()

submission_df = pd.read_csv("dummyTest.csv")
test_img_paths = submission_df["img_path"].tolist()
preds, probs = [], []

for path in tqdm(test_img_paths, desc="Inference"):
    image = Image.open(os.path.join('.', path)).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).item()
        prob = torch.sigmoid(torch.tensor(output)).item()
        label = 1 if prob > 0.5 else 0
    preds.append(label)
    probs.append(prob)

submission_df["label"] = preds
submission_df["probabilities"] = probs
submission_df.to_csv(os.path.join(results_dir, "efficientnet_submission.csv"), index=False)
