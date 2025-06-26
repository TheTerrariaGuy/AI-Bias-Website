import json

with open('merged.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in data:
    message = entry['message']
    
    # Build the prefix first
    prefix = "||| information about the sender "
    
    if entry['sexual_orientation'] != 'none':
        prefix += "||| sexual_orientation: " + entry['sexual_orientation'] + " "
    if entry['religion'] != 'none':
        prefix += "||| religion: " + entry['religion'] + " "
    if entry['race_or_ethnicity'] != 'none':
        prefix += "||| race or ethnicity: " + entry['race_or_ethnicity'] + " "
    if entry['gender'] != 'none':
        prefix += "||| gender: " + entry['gender'] + " "
    if entry['disability'] != 'none':
        prefix += "||| disability: " + entry['disability'] + " "
    
    # Combine prefix with the original message
    entry['message'] = prefix + "||| message sent: " + message

# Fix: data is already a list, don't use .values()
with open('merged_and_labelled.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Processed {len(data)} entries and saved to merged_and_labelled.json")
    
#   {
#     "id": 239579,
#     "worker": 6,
#     "disability": "none",
#     "gender": "male",
#     "race_or_ethnicity": "none",
#     "religion": "none",
#     "sexual_orientation": "none",
#     "message": "This is a great story. Man. I wonder if the person who yelled \"shut the fuck up!\" at him ever heard it."
#   },

