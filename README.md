# Construction Safety AI System

## Overview
A scalable AI system for analyzing construction site footage to detect safety violations, assess risks, and generate automated reports and alerts.

## Project Structure
- `src/agents/`: Specific logic for each of the 6 agents.
- `src/core/`: Shared utilities, types, and configuration.
- `src/interface/`: Streamlit web application.
- `data/`: Local storage for inputs, intermediate files, and vector DB.

## Setup
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Copy `.env.example` to `.env` and fill in your API keys.
4.  Start Qdrant: `docker-compose up -d`
5.  Run the app: `streamlit run src/interface/app.py`
