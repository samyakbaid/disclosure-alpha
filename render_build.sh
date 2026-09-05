#!/usr/bin/env bash
# Render build script: install dependencies, then download data from HF dataset
set -e

pip install -r requirements.txt

# Download the pre-built Qdrant index and chunks from HF dataset (if not present)
python -c "
import os
from pathlib import Path

ROOT = Path('.')
qdrant_dir = ROOT / 'data' / 'qdrant'
chunks_file = ROOT / 'data' / 'processed' / 'chunks.parquet'

if qdrant_dir.exists() and chunks_file.exists():
    print('Data already present, skipping download.')
else:
    print('Downloading data from HF dataset...')
    from huggingface_hub import snapshot_download, hf_hub_download

    # Download Qdrant index
    qdrant_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=os.environ.get('HF_DATASET_REPO', 'Samyakbaid11/disclosure-alpha-data'),
        repo_type='dataset',
        local_dir=str(ROOT / 'data'),
        allow_patterns=['qdrant/**', 'processed/chunks.parquet'],
    )
    print('Data downloaded successfully.')
"
