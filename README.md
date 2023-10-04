# salesGPT_foundation

# salesGPT_v2

### Model Description
salesGPT_v2, derived from microsoft/phi-1_5, is specialized in simulating sales conversations, wherein it understands customer requirements, manages objections, and suggests suitable products or services. It was fine-tuned on a variety of sales-related datasets and seems proficient in initiating conversations, asking pertinent questions, and sustaining interactive dialogues with users.

salesGPT_v1 is the base version of v2 and not trained on instructions

### Related Ressources

salesGPT_v2: https://huggingface.co/goendalf666/salesGPT_v2
salesGPT_v1: https://huggingface.co/goendalf666/salesGPT_v1

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63797fcb2cb50dda39d8aec6/re7MmsaYNzTYVH2jEXDDu.png)

### Intended Uses & Limitations
**Intended Uses:**
- Simulating sales conversations for training or evaluation purposes.
- Providing guidelines or suggested dialogues for sales representatives.

**Limitations:**
- The model might repetitively ask questions in certain scenarios.
- May struggle with handling customers who lack specific preferences or knowledge about products.
- The objection handling could be more focused on convincing techniques rather than objective criteria.
- Challenges in providing appropriate suggestions for customers without specific needs.
- Limited effectiveness in handling financial and budgetary conversations or sensitivities.

### Training and Evaluation Data
**Training Data:**
1. **Textbook v1 Dataset**
   - URL: [Dataset](https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling)
   - Content: Textbook content for sales, derived from structural points and detailed subpoints created through API calls.
   
2. **Sales Conversation Dataset**
   - URL: [Dataset](https://huggingface.co/datasets/goendalf666/sales-conversations)
   - Content: Sales conversations, generated based on the chapters of the textbook.

3. **Sales Conversations Instruction Base Dataset**
   - URL: [Dataset](https://huggingface.co/datasets/goendalf666/sales-conversations-instruction-base)
   - Content: Extended sales conversations with structured dialogues.
   
4. **Sales Conversations Instruction Extension Dataset**
   - URL: [Dataset](https://huggingface.co/datasets/goendalf666/sales-conversations-instruction-ext)
   - Content: Updates based on real conversations with the model to improve its proficiency in unconvincing cases.

**Evaluation Data:**
- More information is needed regarding how and where the model was evaluated. If it was assessed on a separate test set, providing access and details to that dataset would be crucial.

### Training Procedure
Fine-tuning of salesGPT_v2 was executed in three phases using the LoRa approach with Rank 64:
1. Training on a textbook for 20k steps.
2. Training on sales conversations for 40k steps, resulting in salesGPT_v1.
3. Training on sales conversations instruction for 40k steps, evolving into salesGPT_v2.

Hyperparameters used during training:
- Learning rate: 0.0002
- Train batch size: 2
- Eval batch size: 8
- Seed: 42
- Optimizer: Adam
- LR scheduler type: Cosine
- Training steps: 40,000
- [Additional parameters...](https://huggingface.co/goendalf666/salesGPT_v2)

Model size details:
- Trainable params: 18,874,368
- All params: 1,437,145,088
- Trainable%: 1.313323766514519

### Results
While salesGPT_v2 showcases notable capabilities like initiating a structured dialogue and asking relevant questions, there are some key areas of improvement and observations:
- The model tends to keep asking questions if a user provides vague information.
- In some cases, it's not assertive enough to suggest a product when a customer lacks preferences.
- It tends to leverage objective criteria over convincing strategies during objections.
- Needs better handling of conversations regarding budgetary or financial concerns.

### Example Usage & Conversations
#### Example 1:
**Customer**: I am interested in buying a new laptop.
**salesGPT_v2**: I understand. Could you please tell me more about your specific requirements and budget?

#### Example 2:
**Customer**: Oh I don't know how I should explain this to my wife. I said we can't go to vacation because we don't have any money.
**salesGPT_v2**: I see. Can you please provide more details about your financial situation and the specific expenses you are concerned about?

These examples underline the model's capability to engage in sales dialogues and also hint at its limitations, such as possibly prying too much into a customer's personal financial situation.

### Future Directions & Improvement
- Enhancement in handling objections by focusing more on persuasive techniques and emotional intelligence.
- Improving suggestion-making capability especially when customers are indecisive.
- Better navigation through the conversation that involves budgetary and financial aspects without seeming insensitive or intrusive.
- Striking a balance between being informative and being too technical in its product descriptions.
- Possible implementation of more ethical and privacy-guided conversation guidelines, especially in discussing customers' financial capacities.

### Ethical Considerations
The model’s tendency to repeatedly ask for specific information, especially related to personal financial details, raises ethical concerns regarding privacy and data sensitivity. Care must be taken to ensure the model respects user privacy and does not persistently probe for personal or sensitive information.

### Conclusion
salesGPT_v2 offers a foundation for simulating sales conversations with potential for future refinement in handling objections, making product suggestions, and managing conversations delicately around financial discussions. Future versions might seek to refine its balance between being convincingly persuasive and remaining ethically and emotionally intelligent within dialogues.

### Inference

```
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the model and tokenizer
cuda = "cuda:0" if torch.cuda.is_available() else ""
model = AutoModelForCausalLM.from_pretrained("goendalf666/salesGPT_v2", trust_remote_code=True, torch_dtype=torch.float32, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, device_map={"":0})

inputs = tokenizer(conversation_text, return_tensors="pt", return_attention_mask=False)
inputs.to(cuda)

# Generate response
outputs = model.generate(**inputs, max_length=512)
response_text = tokenizer.batch_decode(outputs)[0]
```
Or 

Inference script: https://github.com/tom813/salesGPT_foundation/blob/main/inference.py

### Framework versions

- Transformers 4.32.1
- Pytorch 2.1.0.dev20230829+cu121
- Datasets 2.14.5
- Tokenizers 0.13.3
