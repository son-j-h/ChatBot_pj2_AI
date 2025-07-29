#!/bin/bash

echo "[SYNC] Syncing local Git repositories with upstream and origin..."

# --- ChatBot_pj2 처리 ---
cd ChatBot_pj2

# upstream이 없으면 추가
git remote | grep -q upstream || git remote add upstream https://github.com/Seowon-Park/ChatBot_pj2.git

# upstream에서 최신 변경사항 가져오고 병합
git fetch upstream
git merge upstream/master

cd ..

# --- ChatBot_pj2_AI 처리 ---
cd ChatBot_pj2_AI

git remote | grep -q upstream || git remote add upstream https://github.com/Seowon-Park/ChatBot_pj2_AI.git

git fetch upstream
git merge upstream/master

cd ..

# --- Docker 재시작 ---
echo "[DEPLOY] Rebuilding and restarting Docker services..."

docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "[DONE] Sync, merge, and deployment complete."
