import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torch import nn

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
hidden_size = 8

def load_lines(filename):
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                lines[parts[0]] = parts[4].strip()
    return lines

lines = load_lines("movie_lines.txt")  # Replace with your file path

def load_conversations(filename, lines):
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                line_ids = eval(parts[3])
                conversation = [lines[line_id] for line_id in line_ids]
                conversations.append(conversation)
    return conversations

conversations = load_conversations("movie_conversations.txt", lines)  # Replace with your file path


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MyNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyNLPModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

bert_model = BertModel.from_pretrained('bert-base-uncased')    

def extract_bert_embedding(sentence, tokenizer, bert_model):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

# Prepare X_train and y_train
X_train = []
y_train = []

for conversation in conversations:
    for i in range(len(conversation) - 1):
        input_embedding = extract_bert_embedding(conversation[i], tokenizer, bert_model)
        target_embedding = extract_bert_embedding(conversation[i + 1], tokenizer, bert_model)
        X_train.append(input_embedding)
        y_train.append(target_embedding)

input_size = 768  # Size of BERT embeddings
output_size = 768 

model = MyNLPModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss() 

# Convert X_train and y_train to PyTorch tensors
X_train_tensor = torch.stack(X_train)  # Use torch.stack to create a tensor from a list of tensors
y_train_tensor = torch.stack(y_train)

# DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pth')
