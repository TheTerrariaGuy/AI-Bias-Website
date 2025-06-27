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


# CITATIONS

# @inproceedings{logacheva-etal-2022-paradetox,
#     title = "{P}ara{D}etox: Detoxification with Parallel Data",
#     author = "Logacheva, Varvara  and
#       Dementieva, Daryna  and
#       Ustyantsev, Sergey  and
#       Moskovskiy, Daniil  and
#       Dale, David  and
#       Krotova, Irina  and
#       Semenov, Nikita  and
#       Panchenko, Alexander",
#     booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
#     month = may,
#     year = "2022",
#     address = "Dublin, Ireland",
#     publisher = "Association for Computational Linguistics",
#     url = "https://aclanthology.org/2022.acl-long.469",
#     pages = "6804--6818",
#     abstract = "We present a novel pipeline for the collection of parallel data for the detoxification task. We collect non-toxic paraphrases for over 10,000 English toxic sentences. We also show that this pipeline can be used to distill a large existing corpus of paraphrases to get toxic-neutral sentence pairs. We release two parallel corpora which can be used for the training of detoxification models. To the best of our knowledge, these are the first parallel datasets for this task.We describe our pipeline in detail to make it fast to set up for a new language or domain, thus contributing to faster and easier development of new parallel resources.We train several detoxification models on the collected data and compare them with several baselines and state-of-the-art unsupervised approaches. We conduct both automatic and manual evaluations. All models trained on parallel data outperform the state-of-the-art unsupervised models by a large margin. This suggests that our novel datasets can boost the performance of detoxification systems.",
# }
