import sqlite3
import pandas as pd
import nltk
from nltk.corpus import brown
from rich import print as xp
from rich.markdown import Markdown
import re
import argparse
import sys
from typing import Optional


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
    parser = argparse.ArgumentParser(description='Kerukeion conversation summarizer')
    parser.add_argument('--query', '-q', type=str, help='Search words (space-separated). All words must be present in prompt/response to match.')
    parser.add_argument('--min-size', type=int, default=0, help='Only show conversations with more than this many messages')
    parser.add_argument('--min-start', type=str, help='Minimum conversation start date (ISO or unix timestamp)')
    parser.add_argument('--max-start', type=str, help='Maximum conversation start date (ISO or unix timestamp)')
    args = parser.parse_args()

    def _parse_date_to_ts(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        s = str(s).strip()
        if s == "":
            return None
        # try integer epoch
        try:
            return int(s)
        except Exception:
            pass
        try:
            dt = pd.to_datetime(s, utc=True)
            return int(dt.timestamp())
        except Exception:
            raise ValueError(f"Unable to parse date/timestamp: {s}")

    min_start_ts = _parse_date_to_ts(args.min_start)
    max_start_ts = _parse_date_to_ts(args.max_start)

    # Connect to the SQLite database
    conn = sqlite3.connect('morpheus.db')

    # Load all interactions into a Pandas DataFrame
    df_data = pd.read_sql_query("SELECT id, prompt, response, ts FROM interactions where id not like '%asklepios%'", conn)

    # Close the connection
    conn.close()

    # Normalize/convert the 'ts' column to integer unix seconds for robust comparisons
    # Accept existing numeric epochs or ISO datetimes; coerce failures to NaN then drop or set to 0
    df_data['ts_numeric'] = pd.to_numeric(df_data['ts'], errors='coerce')
    if df_data['ts_numeric'].isna().any():
        parsed = pd.to_datetime(df_data['ts'], errors='coerce', utc=True)
        # convert to seconds since epoch (ints)
        parsed_seconds = pd.Series(pd.NA, index=parsed.index, dtype='Float64')
        try:
            # preferred: use astype to avoid deprecated view() usage
            parsed_seconds = (parsed.astype('int64') // 10**9).astype('Int64')
        except Exception:
            # fallback: use dt.floor then astype where possible
            try:
                parsed_seconds = parsed.dt.floor('s').astype('Int64')
            except Exception:
                parsed_seconds = pd.Series([pd.NA]*len(parsed), index=parsed.index, dtype='Float64')

        # where numeric was NaN, fill from parsed_seconds
        mask = df_data['ts_numeric'].isna()
        df_data.loc[mask, 'ts_numeric'] = parsed_seconds[mask]

    # final fallback: fill any remaining NaNs with 0 (or choose to drop)
    df_data['ts_numeric'] = df_data['ts_numeric'].fillna(0).astype('int64')
    # replace original ts with normalized integer seconds
    df_data['ts'] = df_data['ts_numeric']
    df_data.drop(columns=['ts_numeric'], inplace=True)

    # Extract conversation ID and sequence number from the 'id' column
    df_data['convo_id'] = df_data['id'].apply(lambda x: x.split('-')[0])
    df_data['seq'] = df_data['id'].apply(lambda x: int(x.split('-')[1]))

    # Apply query filter if provided (split into words and require all words present)
    if args.query:
        words = re.findall(r"\w+", args.query.lower())
        if len(words) > 0:
            combined = (df_data['prompt'].fillna('') + ' ' + df_data['response'].fillna('')).str.lower()
            mask = pd.Series(True, index=df_data.index)
            for w in words:
                mask &= combined.str.contains(w, regex=False)
            df_data = df_data[mask]
            if df_data.shape[0] == 0:
                print(f"No interactions match query: {args.query}")
                sys.exit(0)

    # Group by conversation ID and concatenate prompts (preserving order)
    grouped = df_data.groupby('convo_id')
    convo_texts = {}
    for convo, group in grouped:
        sorted_group = group.sort_values('seq')
        texts = [row['prompt'] for _, row in sorted_group.iterrows()]
        convo_texts[convo] = ' '.join([t for t in texts if t is not None])

    # Compute rarest words per conversation
    rare_words_dict = {convo: n_rare_words(text, 10) for convo, text in convo_texts.items()}

    # Build base DataFrame
    df_rare = pd.DataFrame({'conversation_id': list(rare_words_dict.keys()),
                            'rarest_words': list(rare_words_dict.values())
                            })

    # Get the minimum ts for each convo_id and counts
    min_ts = grouped['ts'].min().reset_index(name='start_ts')
    counts = grouped.size().reset_index(name='count')

    # Merge with df_rare and sort
    df_rare = df_rare.merge(min_ts, left_on='conversation_id', right_on='convo_id')
    df_rare = df_rare.merge(counts, left_on='conversation_id', right_on='convo_id')
    df_rare = df_rare.sort_values('start_ts').drop(['convo_id_x', 'convo_id_y'], axis=1)

    # Apply min-size filter if requested
    if args.min_size and args.min_size > 0:
        df_rare = df_rare[df_rare['count'] > int(args.min_size)]

    # Apply date window filters if requested
    if min_start_ts is not None:
        df_rare = df_rare[df_rare['start_ts'] >= min_start_ts]
    if max_start_ts is not None:
        df_rare = df_rare[df_rare['start_ts'] <= max_start_ts]

    if df_rare.shape[0] == 0:
        print('No conversations match the supplied filters')
        sys.exit(0)

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
    # Convert Start (seconds) back to readable datetime for display
    try:
        df_rare['Start'] = pd.to_datetime(df_rare['Start'], unit='s', utc=True).dt.tz_convert(None).dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        # best-effort: leave as-is if conversion fails
        pass

    xp(Markdown(df_rare.to_markdown(index=False), style='markdown'))


if __name__ == '__main__':
    main()
