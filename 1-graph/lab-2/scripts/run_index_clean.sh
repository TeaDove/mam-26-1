#!/bin/bash
set -o pipefail
cd "$(dirname "$0")/../graphrag-clean" || exit 2
read -r GRAPHRAG_API_KEY
export GRAPHRAG_API_KEY
if ! curl -sf http://127.0.0.1:8011/health >/dev/null; then
  nohup ~/ml/.venv/bin/python ~/lab1-combined/embed_server.py >embed_server.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:8011/health >/dev/null && break
    sleep 3
  done
fi
curl -sf http://127.0.0.1:8011/health >/dev/null || exit 5
echo "=== clean start $(date -Is) ==="
~/ml/.venv/bin/graphrag index --root .
code=$?
echo "=== clean exit=$code $(date -Is) ==="
[ $code -eq 0 ] || exit 3
echo "=== ALL OK $(date -Is) ==="
