# ==================================================
# Lottery System Server Deploy Script for Windows
# ==================================================

$ServerIP = $env:DEPLOY_IP
$ServerUser = if ($env:DEPLOY_USER) { $env:DEPLOY_USER } else { "root" }
if (-not $ServerIP) { Write-Host "Set DEPLOY_IP env var first!"; exit 1 }
$RemoteDir = "/opt/lottery-system"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Lottery System Deployment" -ForegroundColor Cyan
Write-Host "Target: ${ServerUser}@${ServerIP}" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Build projekt
Write-Host "`n[1/3] Building project..." -ForegroundColor Yellow
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# 2. Fajlok listazasa
Write-Host "`n[2/3] Preparing files for upload..." -ForegroundColor Yellow
$filesToUpload = @(
    "dist",
    "prisma",
    "package.json",
    "package-lock.json",
    "Dockerfile",
    "docker-compose.yml",
    "DEPLOY_GUIDE.md"
)

# 3. Utmutato
Write-Host "`n[3/3] Upload instructions..." -ForegroundColor Yellow
Write-Host "`nKovesse ezeket a lepeseket:" -ForegroundColor Green

Write-Host @"

1. Nyisson meg egy PowerShell ablakot es futtassa:

   scp -r dist prisma package.json package-lock.json Dockerfile docker-compose.yml DEPLOY_GUIDE.md root@${ServerIP}:${RemoteDir}/

   Jelszo: (lasd DEPLOY_PASS kornyezeti valtozo)

2. SSH belepes:

   ssh root@${ServerIP}
   Jelszo: (lasd DEPLOY_PASS kornyezeti valtozo)

3. A szerveren futtassa:

   cd ${RemoteDir}
   docker compose up -d --build

4. Inicializalas:

   curl -X POST http://${ServerIP}:3000/api/admin/init-database

5. Teszt:

   curl http://${ServerIP}:3000/health
   curl http://${ServerIP}:3000/api/admin/lottery-types
   curl http://${ServerIP}:3000/api/admin/predict/otos-lotto?tickets=4

"@ -ForegroundColor White

Write-Host "==========================================" -ForegroundColor Cyan
