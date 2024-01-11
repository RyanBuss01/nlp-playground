import torch
from torch import nn
from transformers import BertTokenizer, BertModel


num_epochs = 1000
batch_size = 8
learning_rate = 0.001
hidden_size = 8 # try 32?
input_size = 768  # Size of BERT embeddings
output_size = 768 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)


def load_conversations(filename):
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            # Splitting the line by tab character
            parts = line.strip().split("\t")
            if len(parts) == 2:
                conversations.append((parts[0], parts[1]))
    return conversations

filename = "bot3/movie-corpus/formatted_movie_lines.txt"  # Replace with your actual file name
conversations = load_conversations(filename)


def extract_bert_embedding(sentence, tokenizer, bert_model, device):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input to GPU
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()  # Move the embeddings back to CPU

# Prepare X_train and y_train
X_train = []
y_train = []
j=0

for conversation in conversations:
    for i in range(len(conversation) - 1):
        input_embedding = extract_bert_embedding(conversation[i], tokenizer, bert_model, device)
        target_embedding = extract_bert_embedding(conversation[i + 1], tokenizer, bert_model, device)
        X_train.append(input_embedding)
        y_train.append(target_embedding)
    j+=1
    print(str(j) + " of " + str(len(conversations)))



# Convert X_train and y_train to PyTorch tensors
X_train_tensor = torch.stack(X_train)  # Use torch.stack to create a tensor from a list of tensors
y_train_tensor = torch.stack(y_train)

torch.save(X_train_tensor, 'bot3/data/X_train_tensor.pt')
torch.save(y_train_tensor, 'bot3/data/y_train_tensor.pt')