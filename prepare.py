import pandas as pd
import json
import os
from pathlib import Path


df1 = pd.read_csv('data/all_data.csv')
df2 = pd.read_csv('data/annotations.csv')

s1 = json.loads(df1.to_json(orient='records'))
s2 = json.loads(df2.to_json(orient='records'))

valid = {}
for entry in s2:
    empty = True
    for k, v in entry.items():
        if k != 'id' and k != 'worker':
            if v is not None and v != '' and v != 'none':
                empty = False
                break
    if not empty:
        valid[entry['id']] = entry
        
for entry in s1:
    if entry['id'] in valid:
         combined_entry = valid[entry['id']].copy()
         combined_entry['message'] = entry['comment_text']
         valid[entry['id']] = combined_entry
        
with open('merged.json', 'w', encoding='utf-8') as f:
    json.dump(list(valid.values()), f, indent=2, ensure_ascii=False)

print(f"Processed {len(valid)} valid entries with enhanced fields")

# note: I deleted the first entry since it was empty and the formatting was off