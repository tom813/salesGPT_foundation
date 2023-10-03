#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datasets import load_dataset, Dataset


# In[2]:


from huggingface_hub import notebook_login
notebook_login()


# In[3]:


data = load_dataset("goendalf666/sales-conversations-2", split="train")
df = data.to_pandas()
df_dict = df.to_dict(orient='list')
df = df.fillna('')


# In[11]:


conversations = []
for i in df.iterrows():
    current_conversation = ""
    try:
        for j in i[1]:
            if "Customer:" in j:
                
                if current_conversation == "" :
                    prompt = f"<StartOfConversation> {j}"
                    conversations.append(prompt)
                    current_conversation += j + " "
                else:
                    prompt = f"""You are a in the role of a Customer. Here is a conversation:
                    {current_conversation}

                    Answer as a Customer to the Salesman.
                    {j}"""

                    conversations.append(prompt)
                
                    current_conversation += j + " "
            elif "Salesman:" in j:
                current_conversation += j + " "
            else:
                break
    except Exception as e:
        print(e)
        


# In[15]:


df = pd.DataFrame(conversations)
ds = Dataset.from_pandas(df)
ds.push_to_hub("goendalf666/sales-conversations-instruction-customer")

