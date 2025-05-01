# FashionMNIST example as a pytorch introduction
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import time

from sklearn.datasets import load_breast_cancer
import numpy as np




start_time = time.time()


data = load_breast_cancer() # φορτώνω το dataset του καρκίνου του μαστού απο την sklearn.

X = data.data
y = data.target # τα κανω numpy arrays .


# Wrap the datasets around a dataloader

batch_size = 32

X = X.astype(np.float32)
y = y.astype(np.int64)


X_tensor = torch.from_numpy(X) # μετατρέπω τα numpy arrays σε torch tensors.
y_tensor = torch.from_numpy(y)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42,shuffle=True)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test) # τα βάζω σε datasets για να μπορώ να τα χρησιμοποιήσω με το dataloader.


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [batch_size, features]: {X.shape}")

    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

# Select device to use
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu"
          )

print(f"Using device {device}")

# Define a neural network model


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, output_dim=2):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # RelU activation function
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x): 
        return self.net(x) 


model = NeuralNetwork().to(device)
print(model)

# Train the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # to optimizer που χρησιμοποιώ ειναι το SGD με learning rate 1e-3,δοκιμαζω και με αλλα learning rates ομως το 1e-3 δουλεύει καλύτερα.


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]") 


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test error: \n Accuracy: {(100*correct):>7f}, Avg loss: {test_loss:>8f} \n")


epochs = 7
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


# το μοντελο έτρεξε πιο γρήγορα και εβγάλε καλυτέρο accuracy 97% απο το προηγούμενο μοντελο που ειχα φτιάξει, ο χρόνος ειναι 1.068176507949829 second 