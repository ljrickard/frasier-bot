# Frasier Bot: RAG Ecosystem ☕️🎙️

An end-to-end Retrieval-Augmented Generation (RAG) architecture dedicated to the television show *Frasier*. 

This repository contains the complete, production-ready stack: a custom Go-based web scraper, an advanced multi-step Go API backend, a fine-tuned Python cross-encoder for semantic reranking, and the Terraform infrastructure required to run it all cost-effectively on Google Kubernetes Engine (GKE).

## 🏗️ Architecture Overview

The system is designed with a microservices approach, utilizing GCP Workload Identity for passwordless authentication between services.

1.  **API Backend (Go):** The core orchestrator. Exposes the `/chat` endpoint and manages the RAG pipeline.
2.  **Cross-Encoder Inference (Python/FastAPI):** A dedicated, GPU-accelerated service running a fine-tuned `ms-marco-MiniLM-L-6-v2` model to mathematically score and rerank transcript chunks.
3.  **Infrastructure (Terraform):** Provisions a GKE Standard cluster with a dual-pool strategy: cheap, ephemeral Spot instances for the Go backend, and an on-demand NVIDIA L4 GPU pool (utilizing `TIME_SHARING`) for the ML inference.

## 🧠 The RAG Pipeline

When a user submits a query, `pipeline_bot.go` executes a strict, six-step retrieval process:

1.  **Query Expansion:** Uses an LLM to rewrite and expand the user's query for better semantic matching.
2.  **Intent Classification (Switchboard):** Determines if the query is "SPECIFIC" or "GENERAL", dynamically scaling the `FetchK` and `FinalK` values to prevent context window dilution.
3.  **Embeddings:** Generates vector embeddings via Vertex AI (`gemini-embedding-001`).
4.  **Vector Search:** Queries the PostgreSQL/pgvector database to fetch the initial wide pool of candidate chunks.
5.  **Cross-Encoder Reranking:** Sends the wide pool to the local Python ML service to be scored and sorted, returning only the most highly relevant chunks.
6.  **Generation:** Passes the reranked context and the user's query to Gemini to generate the final, grounded response.

## 📥 Data Ingestion (Local Scraper)

Transcript data is sourced from `kacl780.net`. 

**Note:** The scraper (`scraper.go`) is explicitly designed to be run **manually from your local machine**. It is not part of the active Kubernetes deployment. 

The scraper crawls the site, sanitizes the raw HTML, strips legal/footer boilerplate, and chunks the text using a Parent-Child strategy (~1500-word parents, ~300-word children) before pushing the embeddings to the remote database.

## 💰 Cost Optimization (Sleep / Wake Ops)

Because the infrastructure relies on an L4 GPU, running this 24/7 for a side project is prohibitively expensive. We manage this via targeted OpenTofu (Terraform) scripts:

* **`./sleep.sh`**: Runs a targeted destroy against the specific GKE Node Pools. This stops all compute billing while safely preserving the static IP, the PostgreSQL database, and the Persistent Volume Claims (PVCs) holding the HuggingFace model weights.
* **`./wake.sh`**: Re-applies the Terraform state, spinning the compute nodes back up and reattaching the workloads within minutes.

## 🚀 Deployment

**Prerequisites:**
* OpenTofu (`tofu`) installed locally.
* Authenticated to GCP (`gcloud auth application-default login`).
* Docker & Helm installed.

**Steps:**
1.  **Infra:** Navigate to the root directory and run `tofu apply` (or `./wake.sh`) to spin up the GKE cluster and databases.
2.  **Database Seeding:** Run the local Go scraper to populate the database with embedded transcripts.
3.  **ML Deployment:** Navigate to the Cross-Encoder directory and run `./deploy.sh` to push the model weights to GCS, build the image, and deploy the Helm chart.
4.  **Backend Deployment:** Deploy the Go API backend to the default Spot node pool.