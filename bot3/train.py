######################################################################################
#
#       Train the model
#       Updated for GPU usage.
# 
#       NOTE: Run proccess.py first with same batch_size
#
#
######################################################################################
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torch import nn
from model import MyNLPModel
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp

num_epochs = 1000
batch_size = 256
learning_rate = 0.001
hidden_size = 256 # try 32?
input_size = 768  # Size of BERT embeddings
output_size = 768 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()
mp.set_start_method('spawn', force=True)


# Initialize your model and move it to the device (GPU or CPU)
model = MyNLPModel(input_size, hidden_size, output_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss() 

# Load X_train_tensor and y_train_tensor from files
X_train_tensor = torch.load(f'bot3/data/X_train_tensor_{batch_size}.pt')
y_train_tensor = torch.load(f'bot3/data/y_train_tensor_{batch_size}.pt')

# DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)

def train():
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"\rEpoch {epoch+1} of {num_epochs}, Loss: {loss.item()}")
        

if __name__ == '__main__':
    train()

torch.save(model.state_dict(), 'bot3/data/model.pth')
