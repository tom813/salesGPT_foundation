#!/usr/bin/env python
# coding: utf-8

# # To-Do's

# In[1]:


#try to remove lora (would be could to start with it)
#adapt the lambda function for mapping the dataset. The chats have to be in another format then the instructions
#chat format should look like the following:
#- all previous answers from both parties in a specific prompt e.g. here are is previous chat: user: ... salesman...
# here is the last answer from the user: ...
# give me a convincing/selling answer: ###answer:

#1. Fine tune with normal textbook
#sequence lenght 2048


# In[26]:


#from huggingface_hub import notebook_login
#notebook_login()


# # Importing Dependencies

# In[1]:


import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os


# # Finetuning

# In[2]:


tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# In[10]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5",
    device_map={"":0},
    trust_remote_code=True,
    quantization_config=bnb_config
)


# In[11]:


#model


# In[12]:


#peft and lora stuff can be left out for training
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


# In[13]:


#model


# In[14]:


#load the new textbook dataset
#make seq len 2048 if possible and train the model for a bit


def tokenize(sample):
    model_inps =  tokenizer(sample["text"], padding=True, truncation=True, max_length=2048)
    return model_inps


# In[15]:


data = load_dataset("goendalf666/sales-textbook_for_convincing_and_selling", "main", split="train")
data_df = data.to_pandas()

data_df["text"] = data_df[["text"]]
data = Dataset.from_pandas(data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
#tokenized_data


# In[16]:


training_arguments = TrainingArguments(
        output_dir="phi-1_5-finetuned-textbook-2",
        per_device_train_batch_size=1, #8
        gradient_accumulation_steps=1, #2
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        #save_strategy="epoch",
        logging_steps=100,
        max_steps=20000,
        num_train_epochs=6,
        #push_to_hub=True
    )


# In[9]:


trainer = Trainer(
    model=model,
    train_dataset=tokenized_data["input_ids"],
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
trainer.push_to_hub()
