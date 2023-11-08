from langdetect import detect
import pandas as pd

# Load the dataset
df = pd.read_csv('datasets/new/labeled_data_cleaned_from_txt_utf8.csv')

# Initialize counters for Lithuanian and English tweets
lt_count = 0
en_count = 0

# Go through each row in the DataFrame
for index, row in df.iterrows():
    try:
        # Detect the language of the tweet
        lang = detect(row['tweet'])
        
        # Increment the corresponding counter
        if lang == 'lt':
            lt_count += 1
        elif lang == 'en':
            en_count += 1
    except:
        # Skip the row if language detection fails
        continue

# Print the counts
print(f"Lithuanian tweets: {lt_count}")
print(f"English tweets: {en_count}")
