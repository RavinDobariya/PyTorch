from torch import nn
import torch
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from timeit import default_timer as timer 

#____________________________________ 1. Get the data ____________________________________#

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",                        
    train=True,                         
    download=True,                      
    transform=ToTensor(),               
    target_transform=None               
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,                        
    download=True,
    transform=ToTensor()            
)


BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# _____________________________________ 2. Create a model ____________________________________#

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)              # Input → Flatten → Linear → Linear → Output
    
class_names = train_data.classes

torch.manual_seed(42)
# Need to setup model with input parameters
model_0 = FashionMNISTModelV0(input_shape=784, hidden_units=10,  output_shape=len(class_names)) # output_shape = 10 => one for every pixel (28x28)
model_0.to("cpu")  


#_____________________________________ 3. Create a loss function and optimizer ____________________________________#

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time)."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# _____________________________________ 4. Create training and testing loops ____________________________________#

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# start timer before training loop
train_time_start = timer()   

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for index, (X, y) in enumerate(train_dataloader):  # index => 1-1875 , X => [32,1,28,28] data set, y => [32] label for each sample in batch
        model_0.train() 
        
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively sum up the loss per batch per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if index % 400 == 0:                            # X is batchSize 32, so every 400 index is 400*32=12800 samples
            print(f"Looked at {index * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)
           
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))          # NOTE: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names))         
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# end timer after training loop
train_time_end = timer()

# Calculate training time    
total_train_time_model_0 = print_train_time(start=train_time_start, end=train_time_end, device=str(next(model_0.parameters()).device))
print(f"Total training time for model 0: {total_train_time_model_0:.3f} seconds")


# torch.manual_seed(42)
# def eval_model(model: torch.nn.Module, 
#                data_loader: torch.utils.data.DataLoader, 
#                loss_fn: torch.nn.Module, 
#                accuracy_fn):
#     """Returns a dictionary containing the results of model predicting on data_loader.

#     Args:
#         model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
#         data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
#         loss_fn (torch.nn.Module): The loss function of model.
#         accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

#     Returns:
#         (dict): Results of model making predictions on data_loader.
#     """
#     loss, acc = 0, 0
#     model.eval()
#     with torch.inference_mode():
#         for X, y in data_loader:
#             # Make predictions with the model
#             y_pred = model(X)
            
#             # Accumulate the loss and accuracy values per batch
#             loss += loss_fn(y_pred, y)
#             acc += accuracy_fn(y_true=y, 
#                                 y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
#         # Scale loss and acc to find the average loss/acc per batch
#         loss /= len(data_loader)
#         acc /= len(data_loader)
        
#     return {"model_name": model.__class__.__name__, # only works when model is object of a class
#             "model_loss": loss.item(),
#             "model_acc": acc}

# # Calculate model 0 results on test dataset
# model_0_results = eval_model(model=model_0, data_loader=test_dataloader,loss_fn=loss_fn, accuracy_fn=accuracy_fn)
# print(model_0_results)