import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datetime import datetime

# Initialize the tokenizer and model
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Function to translate English text to Lithuanian
def translate_to_lithuanian(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["lt_LT"])
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# Read the dataset
df = pd.read_csv('datasets/new/labeled_data.csv')

# Translate the first 100 rows in the 'tweet' column
df.loc[:99, 'tweet'] = df.loc[:99, 'tweet'].apply(translate_to_lithuanian)

# Generate a timestamp
current_time = datetime.now().strftime("%Y%m%d%H%M%S")

# Save the translated dataset with a timestamp appended to the filename
output_file = f'datasets/new/labeled_data_lithuanian_{current_time}.csv'
df.to_csv(output_file, index=False)

print(f"Translation complete. The translated data is saved as '{output_file}'")
