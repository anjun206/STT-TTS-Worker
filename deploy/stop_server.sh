#!/bin/bash

# start_server.sh에서 실행한 uvicorn 프로세스를 찾아서 종료합니다.
# 'pkill -f'는 전체 명령어 라인('uvicorn main:app ...')을 검색합니다.

echo "FastAPI (uvicorn) 서버를 중지합니다..."

# (중요)
# '|| true'를 끝에 붙여줍니다.
# 만약 서버가 이미 중지되어 있거나(첫 배포) 프로세스를 찾지 못하면
# pkill이 오류 코드(1)를 반환하여 CodeDeploy 배포가 실패하게 됩니다.
# '|| true'는 pkill이 실패하더라도 스크립트가 항상 성공(종료 코드 0)으로
# 끝나도록 보장해주는 필수 장치입니다.
pkill -f "uvicorn main:app --host 0.0.0.0 --port 8000" || true

echo "서버 중지 명령이 실행되었습니다."
echo "Life Cycle - ApplicationStop: complete."