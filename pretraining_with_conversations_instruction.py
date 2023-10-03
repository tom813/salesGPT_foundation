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


from huggingface_hub import notebook_login
notebook_login()


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


# In[3]:


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


# In[4]:


#peft and lora stuff can be left out for training
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()


#load the new textbook dataset
#make seq len 2048 if possible and train the model for a bit


def tokenize(sample):
    model_inps =  tokenizer(sample["0"], padding=True, truncation=True, max_length=2048)
    return model_inps


# In[7]:


data = load_dataset("goendalf666/sales-conversations-instruction", split="train")
df = data.to_pandas()
df_dict = df.to_dict(orient='list')
df = df.fillna('')

print(data.column_names)
data = Dataset.from_pandas(df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

training_arguments = TrainingArguments(
        output_dir="salesGPT_v2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        #save_strategy="epoch",
        logging_steps=1000,
        max_steps=40000,
        num_train_epochs=6,
        resume_from_checkpoint="./salesGPT_v1/checkpoint-40000"
        #push_to_hub=True
    )


# In[10]:


trainer = Trainer(
    model=model,
    train_dataset=tokenized_data["input_ids"],
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
trainer.push_to_hub()


# # Saving

# In[18]:


from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(model, "goendalf666/salesGPT_v2", from_transformers=True)
model = peft_model.merge_and_unload()


# In[20]:


model.push_to_hub("goendalf666/salesGPT_v2")

