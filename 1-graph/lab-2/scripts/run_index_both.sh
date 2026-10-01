#!/bin/bash
set -o pipefail
cd "$(dirname "$0")/.." || exit 2
: "${GRAPHRAG_API_BASE:?set GRAPHRAG_API_BASE to the OpenAI-compatible API URL}"
export GRAPHRAG_API_BASE
read -r GRAPHRAG_API_KEY
export GRAPHRAG_API_KEY
graphrag_bin="${GRAPHRAG_BIN:-graphrag}"
embed_python="${EMBED_PYTHON:-python}"
if ! curl -sf http://127.0.0.1:8011/health >/dev/null; then
  nohup "$embed_python" scripts/embed_server.py >embed_server.log 2>&1 &
  for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8011/health >/dev/null && break; sleep 3; done
fi
curl -sf http://127.0.0.1:8011/health >/dev/null || exit 5
for workspace in graphrag-dirty graphrag-clean; do
  echo "=== $workspace start $(date -Is) ==="
  "$graphrag_bin" index --root "$workspace"
  code=$?
  echo "=== $workspace exit=$code $(date -Is) ==="
  [ $code -eq 0 ] || exit 3
done
echo "=== ALL OK $(date -Is) ==="
