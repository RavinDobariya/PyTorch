import torch
from torch import nn
import matplotlib.pyplot as plt

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


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
    

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)  # .shape() => [32,1,28,28]
# dataset to turn into iterable # how many samples per batch?  # shuffle data every epoch?
 
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # .shape() => [32,1,28,28] (last batch will have 16 samples)


# Let's check out what we've created
print(f"Dataloaders: {train_dataloader}, {test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}") # 60000 samples / 32 samples per batch = 1875 batches
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")   # 10000 samples / 32 samples per batch = 312.5 batches (round up to 313 batches)



# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features_batch.shape}") # [32, 1, 28, 28] (batch size, colour channels, height, width
print(f"Label batch shape: {train_labels_batch.shape}") # [32] (batch size,)

# Show a sample
# torch.manual_seed(42)
plt.figure() 
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()       # size=[x] returns x random integer, .item() turns tensors into a python number
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off");

plt.show()

print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")


# Create a flatten layer
flatten_model = nn.Flatten()    # NOTE: all nn modules function as a model (can do a forward pass)

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
y = flatten_model(x) # perform forward pass

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {y.shape} -> [color_channels, height*width]")

# NOTE: Flattening converts multi-dimensional data (like images) into a 1D vector 
# so it can be used by layers (like nn.Linear) that expect flat input.