##############################################################################################################
#
#   This script is used to preproccess the data for the chatbot bot model.
#   It is used to create the X_train_tensor and y_train_tensor files.
#
#   Updated for GPU usage.
#
##############################################################################################################
import torch
from transformers import BertTokenizer, BertModel

batch_size = 256


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

def batch_extract_bert_embedding(sentences, tokenizer, bert_model, device):
    # Process sentences in batches
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input to GPU

        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # Move the embeddings back to CPU
            all_embeddings.extend(embeddings)
        print(f"\r Batch {i+1} of {len(sentences)}", end="")

    return all_embeddings

# Assuming conversations is a list of (input, target) tuples
input_sentences = [conv[0] for conv in conversations]
target_sentences = [conv[1] for conv in conversations]

# Batch extract embeddings
input_embeddings = batch_extract_bert_embedding(input_sentences, tokenizer, bert_model, device)
target_embeddings = batch_extract_bert_embedding(target_sentences, tokenizer, bert_model, device)

# Ensure that input_embeddings and target_embeddings have the same length
assert len(input_embeddings) == len(target_embeddings)

# Prepare X_train and y_train
X_train = input_embeddings
y_train = target_embeddings


# Convert X_train and y_train to PyTorch tensors
X_train_tensor = torch.stack(X_train)  # Use torch.stack to create a tensor from a list of tensors
y_train_tensor = torch.stack(y_train)

torch.save(X_train_tensor, f'bot3/data/X_train_tensor_{batch_size}.pt')
torch.save(y_train_tensor, f'bot3/data/y_train_tensor{batch_size}.pt')

print(f"\rSaved tensors !!")