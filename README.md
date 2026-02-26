# 🚧 Construction Safety AI System

An end-to-end AI pipeline that ingests construction site video footage, detects PPE violations and safety hazards using computer vision, retrieves relevant OSHA/CAL-OSHA regulations via RAG, performs LLM-powered risk assessment, generates formal compliance reports, and fires real-time alerts — all orchestrated through a **LangGraph** multi-agent workflow.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [OSHA Knowledge Base](#osha-knowledge-base)
- [Output](#output)

---

## How It Works

```
Video File
    │
    ▼
[1] VideoProcessor     — Extract frames at 1fps via ffmpeg (up to 10 frames)
    │
    ▼
[2] VisionDetector     — Two-stage object detection per frame:
    │                      Stage 1: Roboflow YOLO (fast, ground-level)
    │                      Stage 2: Claude Vision fallback (aerial / any angle)
    ▼
[3] RAGRetriever       — Hybrid semantic search (dense + sparse / RRF fusion)
    │                    against Qdrant vector DB of CAL-OSHA regulations
    ▼
[4] RiskAssessor       — Claude (via AWS Bedrock) evaluates violations +
    │                    equipment context + regulations → risk score 0-100,
    │                    alert level (LOW / MEDIUM / HIGH / CRITICAL)
    ▼
[5] ReportGenerator    — Claude writes 7-section formal incident report;
    │                    saved as .docx + .pdf (via ReportLab — no LibreOffice needed)
    ▼
[6] AlertAgent         — Publishes alert via AWS SNS (email) if violations found
```

The full pipeline is wired together as a **LangGraph** `StateGraph` in [`graph.py`](graph.py). Each step reads from and writes back to a shared `GraphState` typed dict.

---

## Agent Pipeline

### 1. `VideoProcessor` (`agents/video.py`)
Uses `ffmpeg` and `ffprobe` to extract frames at exactly **1 fps** (up to `max_frames=10`). Each frame is a PIL `Image` stored in a `Frame` schema object with its frame number and timestamp.

### 2. `VisionDetector` (`agents/vision.py`)
Two-stage detection per frame:
- **Stage 1 — Roboflow YOLO** (`construction-site-safety/27`): fast inference against a fine-tuned model. Detects PPE violations (`NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`), heavy machinery, people, and site objects. Filtered at a confidence threshold of `0.30`.
- **Stage 2 — Claude Vision fallback**: if Roboflow returns no detections for a frame, Claude 3.5 Haiku (multimodal, via AWS Bedrock) analyzes the raw image using a structured safety inspector prompt and returns a JSON array of detections. Handles overhead/aerial camera angles that Roboflow struggles with.

### 3. `RAGRetriever` (`agents/rag.py`)
Queries a **Qdrant** vector database containing chunked CAL-OSHA regulations. Each unique detection label is mapped to a targeted safety query string (e.g. `"NO-Hardhat"` → `"hard hat head protection requirement construction site"`). Retrieval uses **hybrid search** — dense (BGE-small-en) + sparse (SPLADE) vectors fused with Reciprocal Rank Fusion (RRF). High-priority labels (PPE violations) retrieve up to 5 regulations; medium (machinery) up to 3; low up to 1.

### 4. `RiskAssessor` (`agents/risk.py`)
Sends detection context (violations, heavy equipment present, worker count) and retrieved OSHA excerpts to **Claude via Bedrock** with a structured JSON prompt. Returns a `RiskAssessment` with:
- `risk_score` (0–100)
- `alert_level` (LOW / MEDIUM / HIGH / CRITICAL)
- Per-violation breakdown with severity and reasoning
- Falls back to a simple rule-based scorer if the LLM call fails.

### 5. `ReportGenerator` (`agents/report.py`)
Combines raw detection statistics (counts + avg confidence per label), risk assessment, and OSHA regulation text into a prompt for Claude to write a **7-section formal incident report**:
1. Executive Summary
2. Site Conditions Overview
3. Violations Detected (per type, with accident scenario)
4. Applicable Regulations (citation, breach analysis, OSHA enforcement)
5. Risk Assessment Analysis
6. Recommended Corrective Actions
7. Compliance & Legal Implications

The report is built as a styled **`.docx`** (color-coded risk banner, detection summary table, violation rows highlighted) and simultaneously as a **`.pdf`** using **ReportLab** — no LibreOffice or system dependencies required.

### 6. `AlertAgent` (`agents/alert.py`)
Publishes a plain-text alert to an **AWS SNS** topic (email subscription) containing the alert level, risk score, and violation summary. Gracefully skips if AWS credentials aren't configured.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` |
| LLM (reasoning + vision) | AWS Bedrock — Claude 3.5 Haiku (`us.anthropic.claude-3-5-haiku-20241022-v1:0`) |
| Object Detection | Roboflow Inference SDK (`construction-site-safety/27` YOLO) |
| Vector DB | [Qdrant](https://qdrant.tech/) (hybrid dense + sparse) |
| Dense Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed |
| Sparse Embeddings | `prithivida/Splade_PP_en_v1` via FastEmbed |
| Video Processing | `ffmpeg` / `ffprobe` |
| Report Generation | `python-docx` (DOCX) + ReportLab (PDF) |
| Alerts | AWS SNS |
| UI | Streamlit |
| Data Schemas | Pydantic v2 |

---

## Project Structure

```
construction_safety/
├── app.py              # Streamlit web UI
├── graph.py            # LangGraph pipeline (SafetyGraph)
├── ingestion.py        # CAL-OSHA PDF → Qdrant ingestion script
├── model.py            # AWS Bedrock wrapper (text + vision JSON)
├── schemas.py          # Pydantic models (Frame, Detection, RiskAssessment, …)
├── config.py           # All settings from .env
├── utils.py            # AWS SNS helpers + email subscription (subscribe_email)
├── docker-compose.yml  # Qdrant container
├── requirements.txt
├── .env.example
└── agents/
    ├── video.py        # VideoProcessor — ffmpeg frame extraction
    ├── vision.py       # VisionDetector — Roboflow + Claude Vision
    ├── rag.py          # RAGRetriever — hybrid Qdrant search
    ├── risk.py         # RiskAssessor — Claude risk scoring
    ├── report.py       # ReportGenerator — .docx + PDF report
    └── alert.py        # AlertAgent — AWS SNS alerting
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- `ffmpeg` installed on the system
- AWS account with Bedrock access (Claude 3.5 Haiku enabled in your region)
- Roboflow API key

### 1. Clone & install

```bash
git clone <repo-url>
cd construction_safety
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys (see Configuration section)
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

### 4. Ingest OSHA regulations

Place your `CAL_OSHA.pdf` in `data/docs/` then run:

```bash
python ingestion.py
```

This chunks the PDF, generates dense + sparse embeddings, and upserts everything into Qdrant. The Streamlit app also does this automatically on first launch if the collection is empty.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Configuration

All settings are loaded from `.env` via `config.py`:

| Variable | Description |
|---|---|
| `AWS_DEFAULT_REGION` | AWS region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS credentials (or use IAM role / profile) |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `BEDROCK_MODEL_ID` | Claude model ID (default: `us.anthropic.claude-3-5-haiku-20241022-v1:0`) |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `QDRANT_HOST` | Qdrant host (default: `localhost`) |
| `QDRANT_PORT` | Qdrant port (default: `6333`) |
| `QDRANT_COLLECTION_NAME` | Collection name (default: `osha_regulations`) |
| `ALERT_EMAIL_DEST` | Email address to receive SNS alerts |
| `S3_REPORTS_BUCKET` | S3 bucket for sharing report PDFs in alerts (optional) |
| `SLACK_WEBHOOK_URL` | Slack webhook for alerts (optional) |

---

## OSHA Knowledge Base

The RAG system is built on CAL-OSHA regulations loaded from a PDF (`data/docs/CAL_OSHA.pdf`). The ingestion pipeline:

1. Loads and cleans the PDF (removes noise pages, fixes ligature artifacts from PDF extraction)
2. Splits into ~1000-character chunks with 150-character overlap
3. Deduplicates and filters low-signal chunks
4. Embeds with both `bge-small-en` (dense) and `SPLADE_PP_en` (sparse)
5. Upserts into Qdrant as a hybrid collection

At query time, hybrid RRF fusion is used to surface the most relevant regulatory passages for each detected violation or equipment type.

---

## Output

For each video analyzed, the system produces:

- **Terminal logs** — per-frame detection breakdown, regulation count, risk score
- **Risk assessment** — score (0–100), alert level, per-violation severity
- **Incident report** — color-coded `.docx` + `.pdf` saved to `data/reports/`
- **SNS alert email** — sent if violations are detected and AWS SNS is configured
- **Streamlit UI** — live pipeline progress, metrics, violation list, OSHA regulation viewer, and report download
- **Alert subscriptions** — any email address can subscribe to alerts directly from the sidebar without editing `.env`
