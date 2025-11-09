#!/usr/bin/env bash
set -euo pipefail

APP_MODE="${APP_MODE:-worker}"
APP_PORT="${APP_PORT:-8000}"

echo "[entrypoint] starting in mode: ${APP_MODE}"

case "${APP_MODE}" in
  worker)
    exec python -m app.worker
    ;;
  api)
    exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}"
    ;;
  *)
    echo "Unknown APP_MODE: ${APP_MODE}" >&2
    exit 1
    ;;
esac
