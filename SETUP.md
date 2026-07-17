# Setup Guide

This guide walks you through setting up the project locally or in a containerized development environment.

## 1. Prerequisites

Install the following before beginning:

- Docker Engine or Docker Desktop
- Docker Compose v2
- Python 3.11+ (optional, if you want to run the agent outside containers)
- A Google API key for Gemini

## 2. Prepare environment variables

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

Edit [.env](.env) and set at least:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The Docker Compose file already provides defaults for Chroma and Mongo, but you can override them if needed.

## 3. Start the services

From the project root, run:

```bash
docker compose up --build
```

This will start:

- the ADK agent on port 8000
- ChromaDB on its internal container network
- MongoDB on its internal container network

## 4. Verify the stack

Check that the containers are running:

```bash
docker compose ps
```

If you need to inspect startup behavior:

```bash
docker compose logs -f agent
```

## 5. Access the web UI

Open the forwarded port for 8000 in your environment.

- In a local setup, visit http://localhost:8000
- In GitHub Codespaces, use the forwarded port that corresponds to 8000

## 6. Use the agent

Once the UI is open:

1. Upload a text file
2. Wait for the agent to process it
3. Review the generated status messages

The current implementation is designed around text-based ingestion and chunking.

## 7. Stop or reset the environment

To stop the containers:

```bash
docker compose down
```

To remove the persisted data volumes as well:

```bash
docker compose down -v
```

## 8. Optional: run locally without Docker

If you want to run the application directly on your machine:

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

You will still need MongoDB and ChromaDB available for the services to function.

## 9. Troubleshooting

### Port already in use

If port 8000 is busy, edit [docker-compose.yml](docker-compose.yml) and change the published port mapping.

### MongoDB connection errors

Check whether MongoDB is still starting up and verify your URI in [.env](.env).

### Google API errors

Confirm that your API key is valid and that the associated Google Cloud project allows Gemini API access.

### Agent fails to process a file

The current implementation is focused on plain text files. If you upload other formats, the agent will report that the file type is not yet fully supported by the current workflow.
