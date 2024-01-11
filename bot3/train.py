import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torch import nn
from model import MyNLPModel

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
hidden_size = 8 # try 32?
input_size = 768  # Size of BERT embeddings
output_size = 768 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNLPModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss() 

# Load X_train_tensor and y_train_tensor from files
X_train_tensor = torch.load('bot3/data/X_train_tensor.pt')
y_train_tensor = torch.load('bot3/data/y_train_tensor.pt')

# DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 

        # training step
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, "of", num_epochs)

torch.save(model.state_dict(), 'bot3/data/model.pth')
