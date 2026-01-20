# Backable Risk Engine - New Infrastructure

## Overview
Comprehensive risk assessment engine providing ultra-deep business security analysis with multi-database intelligence integration.

## Architecture
- **Framework**: FastAPI with Uvicorn/Gunicorn
- **AI Model**: Google Gemini 2.5 Pro (Vertex AI primary, API key fallback)
- **Databases**: PostgreSQL (Multi-database integration - 7 engine pools)
- **Storage**: Azure Blob Storage (unified-clients-prod)
- **Deployment**: Azure App Service (Python 3.11)

## Live Deployment
- **Production URL**: https://backable-dream-analyzer-new.azurewebsites.net
- **Resource Group**: BACKABLE-AI-NEW-INFRASTRUCTURE

## Main File
`BACKABLE NEW INFRASTRUCTURE RISK ENGINE.py`

## Deployment
Automatic deployment via GitHub Actions on push to main branch.
