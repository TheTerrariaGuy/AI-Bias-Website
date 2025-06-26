import torch
import json
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm

# USING RNG SEED
SEED = 69420
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


with open('merged_and_labelled.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# Sample from the data
SAMPLE_SIZE = 15000

if len(data) > SAMPLE_SIZE:
    sampled_data = random.sample(data, SAMPLE_SIZE)
else:
    sampled_data = data


tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')


data = sampled_data


def getScore(text):
    batch = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    # increasing temperature to make graph look better, the model is too confident
    temperature = 2.5 
    
    with torch.no_grad():
        output = model(batch)

    logits = output.logits

    scaled_logits = logits / temperature

    probabilities = F.softmax(scaled_logits, dim=-1)

    return (probabilities[0][1] - probabilities[0][0]).item()

#  wowowow so fancy progress bar
print(f"Processing {len(data)} entries...")
for entry in tqdm(data, desc="Calculating toxicity scores"):
    message = entry['message']
    toxicity_score = getScore(message)
    entry['toxicity_score'] = toxicity_score

# Save the results
with open('merged_with_scores.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("done!")