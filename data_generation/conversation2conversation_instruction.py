#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from datasets import load_dataset, Dataset


# In[15]:


from huggingface_hub import notebook_login
notebook_login()


# In[16]:


data = load_dataset("goendalf666/sales-conversations-2", split="train")
df = data.to_pandas()
df_dict = df.to_dict(orient='list')
df = df.fillna('')


# In[17]:


conversations = []
for i in df.iterrows():
    current_conversation = ""
    try:
        for j in i[1]:
            if "Customer:" in j:
                current_conversation += j + " "
            elif "Salesman:" in j:

                prompt = f"""You are a in the role of a Salesman. Here is a conversation:
                {current_conversation}

                Answer as a Salesman to the previous Statement to convince the person to buy the product or service.
                {j}"""

                conversations.append(prompt)
                #print("######################################")
                current_conversation += j + " "
            else:
                break
    except Exception as e:
        print(e)
        

# In[18]:


df = pd.DataFrame(conversations)
ds = Dataset.from_pandas(df)
ds.push_to_hub("goendalf666/sales-conversations-instruction")


