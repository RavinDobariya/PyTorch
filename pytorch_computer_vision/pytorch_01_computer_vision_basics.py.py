import torch
from torch import nn
import matplotlib.pyplot as plt

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# torchvision:
# Contains datasets, model architectures, and image transformations
# commonly used for computer vision tasks.

# torchvision.datasets:
# Provides ready-to-use datasets for tasks like image classification,
# object detection, image captioning, video classification, etc.
# Also includes base classes to create custom datasets.

# torchvision.models:
# Contains pre-built and well-performing computer vision model architectures
# that can be used directly or fine-tuned for your own tasks.

# torchvision.transforms:
# Provides image transformations such as converting images to tensors,
# normalization, resizing, and data augmentation.

# torch.utils.data.Dataset:
# Base class used to create custom datasets in PyTorch.

# torch.utils.data.DataLoader:
# Wraps a dataset and provides an iterable for batching,
# shuffling, and loading data efficiently.


# Check versions
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",                        # path where data is stored
    train=True,                         # get training data, if false then get test data
    download=True,                      # download data if it doesn't exist on disk
    transform=ToTensor(),               # images come as PIL format, its turn into Torch tensors
    target_transform=None               # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,                        # get test data
    download=True,
    transform=ToTensor()                # images come as PIL format, its turn into Torch tensors
)


# See classes
class_names = train_data.classes        # class_names = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover",..., 9: "Ankle boot"}
print(f"Class names: {class_names}")

image, label = train_data[0]
print(f"Image shape: {image.shape}, Label: {label}")    # image shape is [1, 28, 28] , label is 0 to 9

# use squeeze Bcuz matplotlib expects [28, 28] but our image shape is [1, 28, 28] 
plt.imshow(image.squeeze(), cmap="gray") # image shape is [1, 28, 28] (colour channels, height, width) 
plt.title(class_names[label])

print(f"Training data length: {len(train_data)}, Test data length: {len(test_data)}") # 60000 training samples, 10000 testing samples

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1): # 16 images - 1 to 16
    random_idx = torch.randint(0, len(train_data), size=[1]).item()     # size=[x] returns x random integer, .item() turns tensors into a python number
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)          # same as plt.subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
    
plt.show()
