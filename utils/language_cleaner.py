from langdetect import detect
import pandas as pd

# Load the DataFrame
df = pd.read_csv('datasets/old/train_tweets.csv')

# Initialize counters
lt_count = 0
en_count = 0

# Initialize an empty list to store Lithuanian tweet rows
lt_rows = []

# Check language
for index, row in df.iterrows():
    try:
        lang = detect(row['tweet'])
        if lang == 'lt':
            lt_count += 1
            lt_rows.append(row)
        else:
            en_count += 1
    except:
        continue

# Create a new DataFrame from the list of Lithuanian rows
df_lt = pd.DataFrame(lt_rows)

# Reset the index of the DataFrame
df_lt.reset_index(drop=True, inplace=True)

# Save the Lithuanian tweets to a new CSV file
df_lt.to_csv('datasets/old/train_tweets_lt_.csv', header=True, index=False, encoding='utf-8-sig')

print(f"Lithuanian tweets: {lt_count}")
print(f"English tweets: {en_count}")
