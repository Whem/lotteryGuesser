# Lottery System Server - Telepítési útmutató

## Szerver adatok
- **IP**: (set DEPLOY_IP env var)
- **User**: root
- **Port**: 3000

## Környezeti változók beállítása
```bash
export DEPLOY_IP=<szerver_ip>
export DEPLOY_USER=root
export DEPLOY_PASS=<jelszo>
export DEPLOY_DOMAIN=<domain>
```

## Előfeltételek a szerveren

```bash
# Docker telepítése
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Docker Compose telepítése
apt-get update
apt-get install -y docker-compose-plugin
```

## Telepítés lépései

### 1. Projekt feltöltése a szerverre

```bash
# Lokálisan - package készítése
cd LotterySystem/server
npm run build
tar -czf lottery-deploy.tar.gz dist/ prisma/ package*.json Dockerfile docker-compose.yml

# Feltöltés
scp lottery-deploy.tar.gz root@<YOUR_SERVER_IP>:/opt/
```

### 2. SSH belépés és indítás

```bash
ssh root@<YOUR_SERVER_IP>
# Jelszó: (see DEPLOY_PASS env var)

# Mappa létrehozása és kicsomagolás
mkdir -p /opt/lottery-system
cd /opt/lottery-system
tar -xzf /opt/lottery-deploy.tar.gz

# Docker Compose indítás
docker compose up -d --build

# Logok ellenőrzése
docker compose logs -f
```

### 3. Adatbázis inicializálás

Miután a szerver elindult, töltsd le a lottó adatokat:

```bash
# Összes lottó típus inicializálása
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/init-database
```

## API Végpontok

### Health check
```bash
curl http://<YOUR_SERVER_IP>:3000/health
```

### Elérhető lottó típusok
```bash
curl http://<YOUR_SERVER_IP>:3000/api/admin/lottery-types
```

### Lottó adatok letöltése (egyesével)
```bash
# Ötöslottó
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/download/otos-lotto

# Hatoslottó
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/download/hatos-lotto

# Skandináv lottó (7-es)
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/download/skandinav-lotto

# Eurojackpot
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/download/eurojackpot
```

### Összes lottó frissítése
```bash
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/download-all
```

### Predikció generálása
```bash
# Ötöslottó predikció (4 szelvény)
curl http://<YOUR_SERVER_IP>:3000/api/admin/predict/otos-lotto?tickets=4

# Hatoslottó predikció
curl http://<YOUR_SERVER_IP>:3000/api/admin/predict/hatos-lotto?tickets=4

# Eurojackpot predikció
curl http://<YOUR_SERVER_IP>:3000/api/admin/predict/eurojackpot?tickets=4
```

### Statisztikák
```bash
# Ötöslottó statisztikák
curl http://<YOUR_SERVER_IP>:3000/api/admin/stats/otos-lotto
```

## Flutter Mobil App Konfiguráció

A mobil alkalmazásban állítsd be az API URL-t:

```dart
// lib/core/config.dart
const String API_BASE_URL = 'http://<YOUR_SERVER_IP>:3000/api';
```

## Hibaelhárítás

### Konténerek újraindítása
```bash
cd /opt/lottery-system
docker compose down
docker compose up -d --build
```

### Adatbázis reset
```bash
docker compose down -v  # Adatok törlése
docker compose up -d --build
curl -X POST http://<YOUR_SERVER_IP>:3000/api/admin/init-database
```

### Logok ellenőrzése
```bash
docker compose logs lottery-api
docker compose logs postgres
```
