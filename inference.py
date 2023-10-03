#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the model and tokenizer
cuda = "cuda:0" if torch.cuda.is_available() else ""
model = AutoModelForCausalLM.from_pretrained("goendalf666/salesGPT_v2", trust_remote_code=True, torch_dtype=torch.float32, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, device_map={"":0})

# Global list to keep track of all conversations and a temporary list for the current conversation
all_conversations = []
current_conversation = []

def save_conversation(all_conversations):
    existing_conversations = []

    # Try to read existing conversations
    try:
        with open("sample_conversations.json", "r") as file:
            existing_conversations = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Append new conversations to the existing ones
    all_conversations_extended = existing_conversations + all_conversations

    # Save all conversations back to file
    with open("sample_conversations.json", "w") as file:
        json.dump(all_conversations_extended, file)

def interact_with_model(user_input):
    global current_conversation
    global all_conversations  # Added to ensure we work on the global variable

    # Check if an end token is provided in user_input
    for end_token in ["<Success>", "<Defeat>"]:
        if end_token in user_input:
            # Remove the end token from the user's message and save both separately
            user_input = user_input.replace(end_token, '').strip()
            current_conversation.append(f"Customer: {user_input}")
            current_conversation.append(f"Customer: {end_token}")

            # Save the current conversation to the global conversation list
            all_conversations.append(current_conversation.copy())
            save_conversation(all_conversations)

            # Reset the current conversation and all_conversations
            current_conversation = []
            all_conversations = []
            return "Conversation ended with token: " + end_token

    # Add user input to the current conversation
    current_conversation.append(f"Customer: {user_input}")

    # Construct conversation text for the model
    conversation_text = "You are in the role of a Salesman. Here is a conversation: " + " ".join(current_conversation) + " Salesman: "

    # Tokenize inputs
    inputs = tokenizer(conversation_text, return_tensors="pt", return_attention_mask=False)
    inputs.to(cuda)

    # Generate response
    outputs = model.generate(**inputs, max_length=512)
    response_text = tokenizer.batch_decode(outputs)[0]

    # Extract only the newly generated text
    new_text_start = len(conversation_text)
    new_generated_text = response_text[new_text_start:].strip()

    # Find where the next "Customer:" is, and truncate the text there
    end_index = new_generated_text.find("Customer:")
    if end_index != -1:
        new_generated_text = new_generated_text[:end_index].strip()

    # Ignore if the model puts "Salesman: " itself at the beginning
    if new_generated_text.startswith("Salesman:"):
        new_generated_text = new_generated_text[len("Salesman:"):].strip()

    # Add model response to the current conversation
    current_conversation.append(f"Salesman: {new_generated_text}")

    # Return the model's response
    return new_generated_text

# Run the CLI interaction
while True:
    user_input = input("You: ")
    model_response = interact_with_model(user_input)
    print(f"Salesman: {model_response}")

    # If conversation ended, break the loop
    if "Conversation ended with token:" in model_response:
        break

