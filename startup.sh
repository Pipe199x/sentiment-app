#!/usr/bin/env bash
set -e pipefail

# 1) instala (otra vez) requisitos si no lo hizo Oryx
pip install -r requirements.txt

# 2) instala el modelo si no está
python -m spacy download es_core_news_sm

# 3) arranca Gunicorn
exec gunicorn \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  application:app
