from easynmt import EasyNMT
import pandas as pd
from datetime import datetime
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the EasyNMT model
model = EasyNMT('mbart50_en2m')

# Function for translating text
def translate_text(text):
    translated_text = model.translate(text, target_lang='lt')
    return translated_text

# Read the dataset
df = pd.read_csv('datasets/new/labeled_data.csv')

# Translate the 'tweet' column for the first 100 entries
for i in range(0, 100):
    df.loc[i, 'tweet'] = translate_text(df.loc[i, 'tweet'])
    if i % 5 == 0:
        logging.info(f'Translated {i} rows')

# Generate a timestamp
current_time = datetime.now().strftime("%Y%m%d%H%M%S")

# Save the translated dataset with a timestamp appended to the filename
output_file = f'datasets/new/labeled_data_lithuanian_{current_time}.csv'
df.to_csv(output_file, index=False)

logging.info(f"Translation complete. The translated data is saved as '{output_file}'")
