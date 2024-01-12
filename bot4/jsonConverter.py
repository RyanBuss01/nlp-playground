import pandas as pd
import json

# Load BotBuilder data
botbuilder_df = pd.read_csv('bot4/data/friend.tsv', delimiter='\t')

# Initialize the structure similar to your intents.json
converted_data = {"intents": []}

# Initialize a dictionary to keep track of tags for answers
answer_to_tag = {}

int = 0

# Iterate over BotBuilder data and populate the converted_data
for index, row in botbuilder_df.iterrows():
    # Assuming 'Question' and 'Answer' columns exist
    question, answer = row['Question'], row['Answer']
    
    # Check if the answer already has a tag assigned
    if answer in answer_to_tag:
        tag = answer_to_tag[answer]
    else:
        tag = str(int)
        int += 1

        # Store the tag for the answer
        answer_to_tag[answer] = tag

    # Check if the tag already exists
    existing_intent = next((item for item in converted_data["intents"] if item["tag"] == tag), None)
    if existing_intent:
        existing_intent["patterns"].append(question)
        existing_intent["responses"].append(answer)
    else:
        converted_data["intents"].append({
            "tag": tag,
            "patterns": [question],
            "responses": [answer]
        })

# Save converted data to a new JSON file
with open('bot4/data/intents.json', 'w') as json_file:
    json.dump(converted_data, json_file, indent=4)
