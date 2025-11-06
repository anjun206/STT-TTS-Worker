#!/bin/bash
# start_server.sh에서 실행한 uvicorn 프로세스를 찾아서 종료합니다.
# 'pkill -f'는 전체 명령어 라인('uvicorn main:app ...')을 검색합니다.
echo "Stopping worker process..."
echo "FastAPI (uvicorn) 서버를 중지합니다..."
WORKER_PATTERN="python -u app/worker.py"
# (중요)
# '|| true'를 끝에 붙여줍니다.
# 만약 서버가 이미 중지되어 있거나(첫 배포) 프로세스를 찾지 못하면
# pkill이 오류 코드(1)를 반환하여 CodeDeploy 배포가 실패하게 됩니다.
# '|| true'는 pkill이 실패하더라도 스크립트가 항상 성공(종료 코드 0)으로
# 끝나도록 보장해주는 필수 장치입니다.
# pgrep은 찾은 PID를 줄 단위로 출력하므로 공백 분리 가능
PIDS=$(pgrep -f "$WORKER_PATTERN" || true)
if [ -z "$PIDS" ]; then
    echo "No worker process found (already stopped?)."
else
    echo "Worker PID(s) detected: $PIDS"
    for PID in $PIDS; do
        if kill "$PID" 2>/dev/null; then
            echo "Sent SIGTERM to PID $PID"
        fi
    done
    # 잠시 대기 후 잔존 PID 확인
    sleep 2
    REMAINING=$(pgrep -f "$WORKER_PATTERN" || true)
    if [ -n "$REMAINING" ]; then
        echo "Worker still running (PID: $REMAINING). Sending SIGKILL."
        kill -9 $REMAINING 2>/dev/null || true
    fi
fi
echo "서버 중지 명령이 실행되었습니다."
echo "Life Cycle - ApplicationStop: complete."
echo "Life Cycle - ApplicationStop: complete."