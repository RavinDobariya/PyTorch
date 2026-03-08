import torch 
from torch import nn 
import matplotlib.pyplot as plt
from tensor_04_pytorch_linear_regression import plot_predictions
from tensor_05_pytorch_model import LinearRegressionModel 


model = LinearRegressionModel()

# Save model parameters (Recommended)
torch.save(model.state_dict(), "model.pth")

model_new = LinearRegressionModel()                 # create model structure
model_new.load_state_dict(torch.load("model.pth"))

print(model_new.state_dict())

# Save entire model
torch.save(model, "full_model.pth")

# Load entire model
model_full = torch.load("full_model.pth", weights_only=False) # Allow loading the full model object instead of only weights

#Saving full models uses Python pickle, which can run malicious code if the file is unsafe.
#So PyTorch now blocks it unless you explicitly allow it.