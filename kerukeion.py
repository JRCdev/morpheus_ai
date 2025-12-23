import sqlite3
import pandas as pd
import nltk
from nltk.corpus import brown
from rich import print as xp
from rich.markdown import Markdown
import re


_FREQ_DIST = None


def _get_freq_dist():
    global _FREQ_DIST
    if _FREQ_DIST is None:
        # Download NLTK brown corpus quietly if missing
        try:
            nltk.data.find('corpora/brown')
        except Exception:
            nltk.download('brown', quiet=True)
        _FREQ_DIST = nltk.FreqDist(w.lower() for w in brown.words())
    return _FREQ_DIST


def n_rare_words(text: str, n: int) -> str:
    """Return the n rarest lowercase alphabetic words from text (space-joined).

    Rarity is determined against the Brown corpus frequency distribution.
    """
    if not text or n <= 0:
        return ""
    freq_dist = _get_freq_dist()
    # extract lowercase alpha words and keep original order list
    lwords = re.findall(r"\b[a-z]+\b", text.lower())
    if not lwords:
        return ""
    words_set = set(lwords)
    # frequency lookup (default 0)
    word_freqs = {w: freq_dist.get(w, 0) for w in words_set}
    # sort by frequency ascending, then alphabetically for deterministic tie-break
    sorted_words = sorted(word_freqs, key=lambda w: (word_freqs[w], w))
    # select up to n words, preserving first-seen order from lwords
    rare_candidates = sorted_words[: max(n, len(sorted_words))]
    rare_selected = []
    for w in lwords:
        if w in rare_candidates and w not in rare_selected:
            rare_selected.append(w)
        if len(rare_selected) >= n:
            break
    return " ".join(rare_selected[:n])


def main():
    # Connect to the SQLite database
    conn = sqlite3.connect('morpheus.db')

    # Load all interactions into a Pandas DataFrame
    df_data = pd.read_sql_query("SELECT id, prompt, response, ts FROM interactions where id not like '%asklepios%'", conn)

    # Close the connection
    conn.close()

    # Extract conversation ID and sequence number from the 'id' column
    df_data['convo_id'] = df_data['id'].apply(lambda x: x.split('-')[0])
    df_data['seq'] = df_data['id'].apply(lambda x: int(x.split('-')[1]))

    # Group by conversation ID and concatenate prompts
    grouped = df_data.groupby('convo_id')
    convo_texts = {}
    for convo, group in grouped:
        sorted_group = group.sort_values('seq')
        texts = [row['prompt'] for _, row in sorted_group.iterrows()]
        convo_texts[convo] = ' '.join(texts)

    # Compute rarest words per conversation
    rare_words_dict = {}
    for convo, text in convo_texts.items():
        rare_words_dict[convo] = n_rare_words(text, 10)

    # Create a Pandas DataFrame from the dictionary
    df_rare = pd.DataFrame({'conversation_id': list(rare_words_dict.keys()),
                            'rarest_words': list(rare_words_dict.values())
                            })

    # Get the minimum ts for each convo_id
    min_ts = df_data.groupby('convo_id')['ts'].min().reset_index(name='start_ts')

    # get the count for each convo_id
    counts = df_data['convo_id'].value_counts().reset_index()

    print(counts)

    # Merge with df_rare and sort
    df_rare = df_rare.merge(min_ts, left_on='conversation_id', right_on='convo_id')
    df_rare = df_rare.merge(counts, left_on='conversation_id', right_on='convo_id')
    df_rare = df_rare.sort_values('start_ts').drop('convo_id_x', axis=1).drop('convo_id_y', axis=1)

    df_rare.rename(columns={
        'conversation_id': 'ID',
        'rarest_words': 'Conversation Title',
        'start_ts': 'Start',
        'count': '#'
    }, inplace=True)

    # Set Pandas display options for better console output
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

    # Pretty print the DataFrame to the console
    xp(Markdown(df_rare.to_markdown(index=False), style='markdown'))


if __name__ == '__main__':
    main()
