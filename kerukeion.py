import sqlite3
import pandas as pd
import nltk
from nltk.corpus import brown
from rich import print as xp
from rich.markdown import Markdown
import re

# Download NLTK data if not already present
nltk.download('brown', quiet=True)

# Connect to the SQLite database
conn = sqlite3.connect('morpheus.db')

# Load all interactions into a Pandas DataFrame
df_data = pd.read_sql_query("SELECT id, prompt, response, ts FROM interactions where id not like '%asklepios%'", conn)

# Close the connection
conn.close()

# Extract conversation ID and sequence number from the 'id' column
df_data['convo_id'] = df_data['id'].apply(lambda x: x.split('-')[0])
df_data['seq'] = df_data['id'].apply(lambda x: int(x.split('-')[1]))

# Group by conversation ID
grouped = df_data.groupby('convo_id')

# Dictionary to hold concatenated text for each conversation
convo_texts = {}

for convo, group in grouped:
    # Sort the group by sequence number
    sorted_group = group.sort_values('seq')
    
    # Concatenate prompt and response for each interaction in order
    texts = []
    for _, row in sorted_group.iterrows():
        texts.append(row['prompt'] ) # + ' ' + row['response'])
    full_text = ' '.join(texts)
    
    convo_texts[convo] = full_text

# Load the Brown corpus frequency distribution
freq_dist = nltk.FreqDist(w.lower() for w in brown.words())

# Function to extract unique lowercase alphabetic words from text
def get_words(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set(words), words

# Dictionary to hold the 10 rarest words for each conversation
rare_words_dict = {}

for convo, text in convo_texts.items():
    words, lwords = get_words(text)
    
    # Get frequencies from the corpus (default to 0 if not found)
    word_freqs = {w: freq_dist.get(w, 0) for w in words}
    
    # Sort by frequency ascending, then alphabetically for ties
    sorted_words = sorted(word_freqs, key=lambda w: (word_freqs[w], w))
    
    # Take the top 10 rarest
    rare_words_list = sorted_words[:20]

    rare_words = " ".join(sorted(rare_words_list, key=lwords.index)[:10])
    
    rare_words_dict[convo] = rare_words

# Create a Pandas DataFrame from the dictionary
df_rare = pd.DataFrame({'conversation_id': list(rare_words_dict.keys()),
                        'rarest_words': list(rare_words_dict.values())})

# Get the minimum ts for each convo_id
min_ts = df_data.groupby('convo_id')['ts'].min().reset_index(name='start_ts')

# Merge with df_rare
df_rare = df_rare.merge(min_ts, left_on='conversation_id', right_on='convo_id').drop('convo_id', axis=1)

# Sort by start_ts
df_rare = df_rare.sort_values('start_ts')

df_rare.rename(columns = {
    "conversation_id":"ID",
    "rarest_words":"Conversation Title",
    "start_ts":"Start"
}, inplace = True)

# Set Pandas display options for better console output
pd.set_option('display.max_rows', None)  # Show all rows
#pd.set_option('display.width', 1000)     # Wide enough to prevent unnecessary wrapping
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Pretty print the DataFrame to the console
#print(df_rare.to_markdown(index=False))
xp(Markdown(df_rare.to_markdown(index=False), style="markdown"))
