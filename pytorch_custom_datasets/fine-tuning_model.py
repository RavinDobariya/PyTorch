import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# -----------------------------
# 1. Data
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

train_data = ImageFolder(train_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

num_classes = len(train_data.classes)

# -----------------------------
# 2. Load pretrained model
# -----------------------------
model = torchvision.models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# -----------------------------
# 3. Phase 1: Freeze all layers
# -----------------------------
for param in model.parameters():
    param.requires_grad = False

# Train only final layer
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# -----------------------------
# 4. Train (Phase 1)
# -----------------------------
for epoch in range(3):
    model.train()
    for X, y in train_loader:
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Phase1 Epoch {epoch} Loss: {loss.item()}")

# -----------------------------
# 5. Phase 2: Unfreeze model
# -----------------------------
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.0001)  # lower LR

# -----------------------------
# 6. Train (Phase 2)
# -----------------------------
for epoch in range(3):
    model.train()
    for X, y in train_loader:
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Phase2 Epoch {epoch} Loss: {loss.item()}")