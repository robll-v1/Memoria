---
inclusion: always
---

# Memoria Local Setup

When the user wants to set up Memoria, **do NOT jump straight into installation**. First ask key questions to determine the right path.

## Decision Flow

### Question 1: Which AI tool?
Ask: "You're using Kiro, Cursor, or Claude Code? (or multiple?)"
This determines which config files to generate.

### Question 2: MatrixOne database
Ask: "Do you already have a MatrixOne database running? If not, I can help you set one up. You have two options:
1. **Local Docker** (recommended for development) — I'll start one for you with docker-compose
2. **MatrixOne Cloud** (free tier available) — register at https://cloud.matrixorigin.cn, no Docker needed"

Based on the answer:
- **Already have one** → ask for the connection URL (host, port, user, password, database)
- **Local Docker** → follow Docker setup below
- **MatrixOne Cloud** → guide user to register, then get connection URL from console

### Question 3: Embedding provider

**⚠️ CRITICAL: This decision is hard to reverse. Once the MCP server starts and creates tables, changing embedding provider requires data migration and re-embedding all memories.**

Ask: "For memory search quality, Memoria needs an embedding model. Do you already have an OpenAI-compatible embedding endpoint? (OpenAI, SiliconFlow, Ollama, or any custom service)
- **Yes** → use it directly. No download, no cold-start. Best choice.
- **No** → use local model. ⚠️ downloads ~900MB (torch + sentence-transformers) on first install. Avoid on slow/proxied networks."

**If user has an existing embedding service**, ask:
- "What is the API base URL? (e.g. `https://api.siliconflow.cn/v1`, `http://localhost:11434/v1`, or leave blank for OpenAI official)"
- "API key? (leave blank if not required)"
- "Model name? (e.g. `BAAI/bge-m3`, `text-embedding-3-small`)"
- "Embedding dimension? (e.g. 1024 for bge-m3, 1536 for text-embedding-3-small)"

These values get written into the `env` block of `mcp.json` automatically — no manual editing needed.

**If user chooses local embedding, explicitly warn**: "This will download ~900MB. If you're on a slow or proxied network, consider using an OpenAI-compatible service instead. Proceed?"

**After collecting embedding config, remind user**: "Once your AI tool starts, the database tables will be created with this embedding dimension. Changing it later requires running `memoria migrate --force` (which clears existing embeddings) and re-embedding all memories via `memoria governance`. Make sure this configuration is correct before proceeding."

## Execution Rules

**CRITICAL: Execute commands one at a time, never chain unrelated steps.**

- Run each command separately and wait for success before proceeding
- If a command fails, stop and diagnose before continuing
- Never chain install + configure + verify into one shell call

## Execution Paths

### Path A: Local Docker + Local Embedding (most common)

```bash
# Step 1: Start MatrixOne (run alone, check output)
docker compose up -d
# or:
docker run -d --name matrixone -p 6001:6001 -v ./data/matrixone:/mo-data --memory=2g matrixorigin/matrixone:latest
```
Wait for success, then:
```bash
# Step 2: Verify MatrixOne is running
docker ps --filter name=matrixone
```
Wait ~30-60s on first start, then:
```bash
# Step 3: Create virtual environment (run alone)
python3 -m venv .venv
```
```bash
# Step 4: Activate it (run alone)
source .venv/bin/activate
```
```bash
# Step 5: Install Memoria (run alone)
# If using local embedding:
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'memoria-lite[local-embedding]'
# If using an existing embedding service (no 900MB download):
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ memoria-lite
```
```bash
# Step 6: Configure (in user's project directory)
# Add embedding flags based on Question 3 answer — see "Embedding provider flags" section below
cd <user-project>
memoria init  # + embedding flags if applicable

# If user didn't provide embedding config via flags, remind them:
# "The config file has been created with all environment variables (even empty ones).
# If you need to customize (database URL, embedding settings), edit the file now before restarting:
# - Kiro: .kiro/settings/mcp.json
# - Cursor: .cursor/mcp.json
# - Claude Code: .claude/mcp.json"
```

### Path B: MatrixOne Cloud

```bash
# 1. User registers at https://cloud.matrixorigin.cn (free tier)
# 2. Get connection info from cloud console: host, port, user, password

# 3. Virtual environment
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
# 4. Install
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'memoria-lite[local-embedding]'
# or, if using an existing embedding service:
# pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ memoria-lite
```
```bash
# 5. Configure with cloud URL
cd <user-project>
memoria init --db-url 'mysql+pymysql://<user>:<password>@<host>:<port>/<database>'
# + embedding flags if applicable (see "Embedding provider flags" section)
```

### Path C: Existing MatrixOne

```bash
# 1. Virtual environment
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
# 2. Install
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'memoria-lite[local-embedding]'
# or, if using an existing embedding service:
# pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ memoria-lite
```
```bash
# 3. Configure with existing DB
cd <user-project>
memoria init --db-url 'mysql+pymysql://<user>:<password>@<host>:<port>/<database>'
# + embedding flags if applicable (see "Embedding provider flags" section)
```

### Embedding provider flags (for any path)

```bash
# Local (default) — no extra flags needed
memoria init

# OpenAI
memoria init --embedding-provider openai --embedding-api-key sk-...

# Existing service (Ollama, SiliconFlow, custom endpoint, etc.)
# All of these get written into the env block of mcp.json automatically
memoria init \
  --embedding-provider openai \
  --embedding-base-url https://api.siliconflow.cn/v1 \
  --embedding-api-key sk-... \
  --embedding-model BAAI/bge-m3 \
  --embedding-dim 1024
```

The resulting `mcp.json` `env` block will contain all configurable variables (even if empty):
```json
{
  "MEMORIA_DB_URL": "...",
  "EMBEDDING_PROVIDER": "openai",
  "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
  "EMBEDDING_API_KEY": "sk-...",
  "EMBEDDING_MODEL": "BAAI/bge-m3",
  "EMBEDDING_DIM": "1024"
}
```

Empty values (e.g. `""`) are treated as "not set" — the MCP server uses defaults (local embedding, dim=384).

## After any path

```bash
# Verify
memoria status

# Tell user to restart their AI tool
```

## Troubleshooting
- MatrixOne won't start → `docker logs memoria-matrixone` to check errors
- Port 6001 in use → edit `.env` to change `MO_PORT`, then `docker compose up -d`
- Can't connect to DB → MatrixOne needs 30-60s on first start, wait and retry
- Cloud connection refused → check firewall/whitelist settings in cloud console
- **Docker permission denied** → `sudo usermod -aG docker $USER && newgrp docker`
- **Image pull slow/timeout** → configure Docker mirror in `/etc/docker/daemon.json`, add `"registry-mirrors": ["https://docker.1ms.run"]`, then `sudo systemctl restart docker`
- **Docker not installed** → suggest MatrixOne Cloud (https://cloud.matrixorigin.cn) as alternative, no Docker needed
- **Data dir permission error** → `mkdir -p data/matrixone && chmod 777 data/matrixone`
- **First query slow** → expected with local embedding; model loads into memory on first use (~3-5s). Subsequent queries are fast. Use `--embedding-provider openai` to avoid this.
