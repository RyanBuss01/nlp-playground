from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Tuned model.generate() parameters for more coherent responses
    output_ids = model.generate(input_ids, max_length=50, temperature=0.9, 
                                top_k=50, top_p=0.95, num_return_sequences=1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Chat loop
while True:
    input_text = input("You: ")
    if input_text.lower() == 'quit':
        break
    response = generate_response(input_text)
    print(f"Bot: {response}")
