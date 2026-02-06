#!/bin/bash

# ==================================================
# Lottery System Server Deploy Script
# ==================================================
# Usage: ./deploy.sh
# 
# Prereqs on server:
# - Docker & Docker Compose installed
# - Port 3000 open
# ==================================================

# Use env vars: DEPLOY_IP, DEPLOY_USER, DEPLOY_PASS
SERVER_IP="${DEPLOY_IP:?Set DEPLOY_IP env var}"
SERVER_USER="${DEPLOY_USER:-root}"
SERVER_PASS="${DEPLOY_PASS:?Set DEPLOY_PASS env var}"
REMOTE_DIR="/opt/lottery-system"

echo "=========================================="
echo "Lottery System Deployment"
echo "Target: ${SERVER_USER}@${SERVER_IP}"
echo "=========================================="

# Helyi build és package
echo "[1/4] Building locally..."
npm run build

# Package fájlok létrehozása
echo "[2/4] Creating deployment package..."
tar -czf lottery-deploy.tar.gz \
    dist/ \
    prisma/ \
    package.json \
    package-lock.json \
    Dockerfile \
    docker-compose.yml

echo "[3/4] Uploading to server..."
# SCP feltöltés (sshpass szükséges)
# sshpass -p "$SERVER_PASS" scp lottery-deploy.tar.gz ${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/

echo "[4/4] Starting services on server..."
# SSH és indítás
# sshpass -p "$SERVER_PASS" ssh ${SERVER_USER}@${SERVER_IP} << 'EOF'
# cd /opt/lottery-system
# tar -xzf lottery-deploy.tar.gz
# docker-compose down
# docker-compose up -d --build
# docker-compose logs -f lottery-api
# EOF

echo ""
echo "=========================================="
echo "Deploy package created: lottery-deploy.tar.gz"
echo ""
echo "Manuális deploy lépések:"
echo "1. Másold át a lottery-deploy.tar.gz fájlt a szerverre:"
echo "   scp lottery-deploy.tar.gz root@${SERVER_IP}:${REMOTE_DIR}/"
echo ""
echo "2. SSH belépés:"
echo "   ssh root@${SERVER_IP}"
echo "   Jelszó: (see DEPLOY_PASS env var)"
echo ""
echo "3. A szerveren:"
echo "   cd ${REMOTE_DIR}"
echo "   tar -xzf lottery-deploy.tar.gz"
echo "   docker-compose up -d --build"
echo ""
echo "4. API elérhetőség:"
echo "   http://${SERVER_IP}:3000/health"
echo "   http://${SERVER_IP}:3000/api/admin/lottery-types"
echo "=========================================="
