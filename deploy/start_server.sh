#!/bin/bash
APP_DIR="/home/ubuntu/app"

# 2. 가상 환경(venv) 경로
VENV_DIR="$APP_DIR/venv/bin/activate"

# 3. (중요) 가상 환경 활성화
#    BeforeInstall 단계에서 생성한 venv를 활성화합니다.
echo "Activating virtual environment at $VENV_DIR..."
if [ ! -f "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR"

# 4. 애플리케이션 코드가 있는 디렉토리로 이동
cd $APP_DIR

# 5. 워커 프로세스를 백그라운드로 실행
LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"

WORKER_LOG="$LOG_DIR/worker.log"
WORKER_ERR_LOG="$LOG_DIR/worker_error.log"

sudo touch "$WORKER_LOG" "$WORKER_ERR_LOG"

echo "Starting worker process: python app/worker.py"
nohup python -u app/worker.py > "$WORKER_LOG" 2> "$WORKER_ERR_LOG" &


echo "Life Cycle - ApplicationStart: complete."
<<<<<<< HEAD
=======
#!/bin/bash


APP_DIR="/home/ubuntu/app"

# 2. 가상 환경(venv) 경로
VENV_DIR="$APP_DIR/venv/bin/activate"

# 3. (중요) 가상 환경 활성화
#    BeforeInstall 단계에서 생성한 venv를 활성화합니다.
echo "Activating virtual environment at $VENV_DIR..."
if [ ! -f "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR"

# 4. 애플리케이션 코드가 있는 디렉토리로 이동
cd $APP_DIR

# 시작 전 ingest 실행 (실패해도 서버는 계속 기동)
# echo "Running glossary ingestion..."
# python script/ingest.py || echo "WARNING: glossary ingestion failed (continuing startup)"

# 5. 워커 프로세스를 백그라운드로 실행
WORKER_LOG="$APP_DIR/worker.log"
WORKER_ERR_LOG="$APP_DIR/worker_error.log"

echo "Starting worker process: python app/worker.py"
nohup python -u app/worker.py > "$WORKER_LOG" 2> "$WORKER_ERR_LOG" &


echo "Life Cycle - ApplicationStart: complete."
>>>>>>> main
