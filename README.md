# morpheus_ai

A tool for combining Large Language Models with an ElasticSearch document store for document retrieval and summarization
Named for the AI from the original Deus Ex

### Getting Started
 - Install an ElasticSearch Local Instance
 - Download a Summarization Model
 - Set up an LLM API Service (OpenAI and Grok currently are known to work)
 - Set up a `config.json` file, filling out the fields as described in `config.json_example`

### Building and Using

 - trigger `mnemosyne.py` to build a database from your ebook library (pdf and epub)
 - once that's done, use `morpheus.py` as a command line interface tool and ask away!

## Development Roadmap
 - Conversation continuations
 - Branching Conversations
