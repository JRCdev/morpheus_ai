# morpheus_ai

A tool for combining Large Language Models with an ElasticSearch document store for document retrieval and summarization
Named for the AI from the original Deus Ex

## Getting Started
 - Install an ElasticSearch Local Instance
 - Download a Summarization Model
 - Set up an LLM API Service (OpenAI and Grok currently are known to work)
 - Set up a `config.json` file, filling out the fields as described in `config.json_example`

## Building and Using

 - trigger `mnemosyne.py` to build a database from your ebook library (pdf and epub)
 - once that's done, use `morpheus.py` as a command line interface tool and ask away!


## Usage of Individual Components

### mnemosyne.py

- `--no-cache` : do not use the persistent SQLite vector cache (recompute embeddings)
- `--clear-cache` : clear the vector cache table before running
- `--vacuum-cache` : run `VACUUM` on the `morpheus.db` SQLite file to reclaim space (useful after `--clear-cache`)
- `--force-reindex` : ignore the `indexed_files` bookkeeping and reprocess all source files
- `--atomic-reindex` : build the new index into a temporary physical index and atomically swap a logical alias to it when complete (prevents partial-index states)
- `--drop-old-index` : when used with `--atomic-reindex`, delete the old physical index(es) after the alias swap

Example (PowerShell)

```powershell
# Build into a temporary index and atomically swap when finished
python .\mnemosyne.py --atomic-reindex

# Rebuild from scratch, clear cache and reclaim DB space
python .\mnemosyne.py --clear-cache --vacuum-cache --force-reindex

# Re-index but avoid using the persistent vector cache
python .\mnemosyne.py --no-cache --force-reindex
```

## Development Roadmap
 - Conversation continuations
 - Branching Conversations
