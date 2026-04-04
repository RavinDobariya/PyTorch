from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from helper_functions import plot_predictions, plot_decision_boundary        


# Linear model + non-linear data => poor learning (random-like output)
# here model doesnt learn anything, its just does random guessing, because the data is non-linear and the model is linear,
# so it cant learn the patterns in the data, it just guesses randomly, which is why we get 50% accuracy, which is what you would expect from random guessing.

#______________________________________________ 1. Create data ______________________________________________

# Make 1000 samples 
n_samples = 1000

# Create circles - non linear dataset
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

print(f"First 5 samples of X: \n{X[:5]}")
print(f"First 5 samples of y: \n{y[:5]}")   

print(f"Shape of X: {X.shape}")     # 1000 rows, 2 columns (x and y coordinates)
print(f"Shape of y: {y.shape}")     # 1000 rows, 1 column (0 or 1 class label) 

plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);
#plt.show()

#______________________________________________ 2. Data Preparation ______________________________________________

# Turn data into tensors
# Otherwise this causes issues with computations later on

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
print(f"First 5 samples of X-non tensor: \n{X[:5]}")
print(f"First 5 samples of y-non tensor: \n{y[:5]}")

# Split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(f"Data sets{len(X_train)}, {len(y_train)},{len(X_test)}, {len(y_test)}")


# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}") # cpu

# ______________________________________________ 3. Building a model ______________________________________________

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)

# _______________________________________________ 4. loss, optimizer, accuracy ______________________________________________

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

torch.manual_seed(42)

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# _______________________________________________ 5. Training and Evaluation Loop ______________________________________________

# Set the number of epochs
epochs = 100

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), y_train) # Using nn.BCELoss you need torch.sigmoid()
   
    loss = loss_fn(y_logits, y_train) # Using nn.BCEWithLogitsLoss works with raw logits
                
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)  

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    #optimizer.state_dict() # shows the state of the optimizer, including the learning rate and momentum
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        # print(f"optimizer state: {optimizer.state_dict()}")           # shows the state of the optimizer, including the learning rate and momentum
        
        
# _______________________________________________ 6. Plot the predictions ______________________________________________

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))

# plt.subplot(rows, columns, index)

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

if __name__ == "__main__":
    plt.show()  
