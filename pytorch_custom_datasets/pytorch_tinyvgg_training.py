# 1. Load and transform data
from torchvision import transforms
import torch
from torchvision import datasets
from torch import nn
from torchinfo import summary
import os
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

      
# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# 2. Turn data into DataLoaders


# Setup batch size and number of workers 
BATCH_SIZE = 32
NUM_WORKERS = 0 # if on Windows, set to 0 Otherwise it will crash due to script rerun multiple times, if on Linux or MacOS, set to 2 or more depending on your CPU
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# Create DataLoader's
train_dataloader_simple = DataLoader(train_data, 
                                     batch_size=BATCH_SIZE, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,    # how big is the square that's going over the image?
                      stride=1,         # default
                      padding=1),       # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # reduce size and keep important features
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

torch.manual_seed(42)

model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) => [batch size, 3, 64, 64]
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)



# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]        # Feeding single image to model, so we need to add an extra dimension to the image (unsqueeze) to make it fit the model's expected input shape  
                                                                                # OR we can also pass whole batch to the model and it will work
print(f"Single image shape: {img_single.shape}\n")

"""
img_batch.shape = [32, 3, 64, 64]  # batch of 32 images

img_batch[0].shape → [3, 64, 64]
img_batch[1].shape → [3, 64, 64]
img_batch[2].shape → [3, 64, 64]
"""

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")

summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size


"""
param => learnable weights + biases in a layer
These are the values the model learns during training

param calculation => (kernel_size * kernel_size * in_features * out_features) + out_features

Params = things model learns

Conv/Linear → learn patterns → need params
ReLU/Pooling → just transform data → no params

"""
