# Modular Agentic RAG Template

This repository provides a lightweight, containerized starter for building an agentic retrieval-augmented generation (RAG) workflow with Google Agent Development Kit (ADK), ChromaDB, and MongoDB.

The current implementation routes uploaded files through an agent that detects the file type, extracts text, splits it into chunks, embeds the chunks with Gemini, and stores the resulting vectors and metadata in ChromaDB and MongoDB.

## What this project does

- Runs a web-based agent using Google ADK
- Accepts uploaded files and detects their MIME type
- Extracts text from supported text-based inputs
- Splits content into manageable chunks
- Embeds chunks with Gemini
- Stores chunk metadata in MongoDB and vector embeddings in ChromaDB

## Architecture

- [agents/agent.py](agents/agent.py): defines the routing and processing agents
- [services/chroma_service.py](services/chroma_service.py): wraps ChromaDB connection and vector storage
- [services/mongo_service.py](services/mongo_service.py): wraps MongoDB connection, chunk persistence, and deduplication tracking
- [docker-compose.yml](docker-compose.yml): orchestrates the agent, ChromaDB, and MongoDB services

## Project structure

```text
.
├── agents/
│   └── agent.py
├── services/
│   ├── chroma_service.py
│   └── mongo_service.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Prerequisites

Before you start, make sure you have:

- Docker and Docker Compose installed
- A Google API key with access to the Gemini embedding and generation models
- Port 8000 available on your machine (or adjust the compose mapping)

## Quick start with Docker Compose

1. Create a local environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit [.env](.env) and set your Google API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. Build and start the stack:

   ```bash
   docker compose up --build
   ```

4. Open the ADK web interface in your browser:

   - Local: http://localhost:8000
   - In Codespaces: use the forwarded port for port 8000 (from cloudflared)

5. Upload a supported text document and let the agent process it.

## Environment variables

The stack uses the following environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| GOOGLE_API_KEY | Required for Gemini API access | none |


## Local development setup

If you prefer to run the agent outside Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_google_api_key_here
export MONGO_URI=mongodb://localhost:27017
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
adk web --host=0.0.0.0 agents
```

You will still need MongoDB and ChromaDB running locally or in Docker.

## Current implementation notes

The current workflow is focused on text-based ingestion. In the implementation, plain text files are wired through the extraction path. Additional document formats can be added by extending the extractor logic in [agents/agent.py](agents/agent.py).

## Useful commands

```bash
# Start services
docker compose up --build

# View logs
docker compose logs -f agent

# Stop services
docker compose down

# Reset persisted data
docker compose down -v
```

## Troubleshooting

- If the agent cannot connect to MongoDB or Chroma, wait a few seconds and check the container logs.
- If you see authentication errors from Google, verify that your API key is valid and has access to the required Gemini models.
- If port 8000 is already in use, change the published mapping in [docker-compose.yml](docker-compose.yml).

## Next steps

This template is a solid base for building a richer RAG experience. Potential next improvements include:

- Adding a retrieval/query agent
- Supporting PDF and DOCX extraction
- Adding a frontend and API layer
- Improving chunking and ranking logic
