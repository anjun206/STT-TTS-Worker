#!/bin/bash

# --- (수정) 스크립트 자신의 위치를 기준으로 '루트 폴더' 찾기 ---
# $BASH_SOURCE[0]는 이 스크립트 파일의 전체 경로를 의미합니다.
# 1. 스크립트 파일이 있는 디렉토리 (예: /opt/.../deploy)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# 2. 그 상위 디렉토리 (압축이 풀린 루트, /opt/.../deployment-archive)
ARCHIVE_ROOT=$( dirname "$SCRIPT_DIR" )
# ---
sudo apt-get update -y
<<<<<<< HEAD
if sudo apt-get install -y python3.11 python3.11-venv jq; then
    PYTHON_BIN=$(command -v python3.11)
    if [ -z "$PYTHON_BIN" ]; then
        echo "ERROR: python3.11 installed but binary not found."
        exit 1
    fi
    echo "Using Python interpreter: $PYTHON_BIN"
else
    echo "WARNING: python3.11 packages unavailable. Falling back to system python3."
    sudo apt-get install -y python3 python3-venv jq
    PYTHON_BIN=$(command -v python3)
    if [ -z "$PYTHON_BIN" ]; then
        echo "ERROR: python3 interpreter not found."
        exit 1
    fi
    echo "Using Python interpreter: $PYTHON_BIN"
fi
=======
sudo apt-get install -y python3.12-venv jq
>>>>>>> main

# 3. venv가 설치될 최종 목적지
APP_DIR="/home/ubuntu/app"
VENV_DIR="$APP_DIR/venv"

# 4. venv 생성
<<<<<<< HEAD
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv: $VENV_DIR"
=======
if [ -d "$APP_DIR" ]; then
    echo "Removing existing APP directory: $APP_DIR"
>>>>>>> main
    rm -rf "$VENV_DIR"
fi
echo "Create APP directory: $APP_DIR"
sudo mkdir -p "$APP_DIR"

echo "Create APP venv: $VENV_DIR..."
<<<<<<< HEAD
"$PYTHON_BIN" -m venv "$VENV_DIR"
=======
python3.12 -m venv "$VENV_DIR"
>>>>>>> main

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

<<<<<<< HEAD
REQ_FILE="$ARCHIVE_ROOT/requirements.txt"
=======
REQ_FILE="requirements.txt"
>>>>>>> main

echo "Installing dependencies from $REQ_FILE..."
if [ -f "$REQ_FILE" ]; then
    echo "SEUCCESS: requirements.txt install"
    pip install -r "$REQ_FILE"
else
    echo "ERROR: requirements.txt not found at $REQ_FILE"
    exit 1
fi

echo "Life Cycle - BeforeInstall: complete."
