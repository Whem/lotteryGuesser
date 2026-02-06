#!/usr/bin/env python3
"""
Lottery System Server - Automatikus Deploy Script
=================================================
Feltölti és telepíti a szervert automatikusan.
"""

import os
import sys
import time
import subprocess

# UTF-8 kódolás beállítása Windows-on
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Szerver adatok - környezeti változókból olvasva!
# Állítsd be: set DEPLOY_IP=... DEPLOY_USER=... DEPLOY_PASS=...
SERVER_IP = os.environ.get("DEPLOY_IP", "")
SERVER_USER = os.environ.get("DEPLOY_USER", "root")
SERVER_PASS = os.environ.get("DEPLOY_PASS", "")
REMOTE_DIR = "/opt/lottery-system"

if not SERVER_IP or not SERVER_PASS:
    print("[HIBA] Allitsd be a kornyezeti valtozokat:")
    print("  set DEPLOY_IP=<szerver_ip>")
    print("  set DEPLOY_USER=root")
    print("  set DEPLOY_PASS=<jelszo>")
    sys.exit(1)

# Szükséges csomagok telepítése
def install_packages():
    """Telepíti a paramiko csomagot ha nincs."""
    try:
        import paramiko
        import scp
    except ImportError:
        print("[*] Paramiko és SCP csomagok telepítése...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "scp", "-q"])
        print("[OK] Csomagok telepítve!")

install_packages()

import paramiko
from scp import SCPClient

def create_ssh_client():
    """SSH kliens létrehozása."""
    print(f"\n[*] Csatlakozás: {SERVER_USER}@{SERVER_IP}")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(
            hostname=SERVER_IP,
            username=SERVER_USER,
            password=SERVER_PASS,
            timeout=30
        )
        print("[OK] SSH kapcsolat létrejött!")
        return client
    except Exception as e:
        print(f"[HIBA] SSH kapcsolat sikertelen: {e}")
        sys.exit(1)

def run_command(client, command, show_output=True):
    """Parancs futtatása a szerveren."""
    if show_output:
        print(f"\n[CMD] {command}")
    
    stdin, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    
    output = stdout.read().decode('utf-8').strip()
    error = stderr.read().decode('utf-8').strip()
    
    if show_output and output:
        print(output)
    if error and "WARNING" not in error.upper():
        print(f"[STDERR] {error}")
    
    return exit_code, output, error

def upload_files(client):
    """Fájlok feltöltése SCP-vel."""
    print("\n[*] Fájlok feltöltése...")
    
    # Lokális könyvtár
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Feltöltendő fájlok és mappák
    items = [
        ("dist", True),
        ("prisma", True),
        ("public", True),
        ("package.json", False),
        ("package-lock.json", False),
        ("Dockerfile", False),
        ("docker-compose.yml", False),
    ]
    
    with SCPClient(client.get_transport()) as scp_client:
        for item, is_dir in items:
            local_path = os.path.join(local_dir, item)
            if os.path.exists(local_path):
                print(f"  -> {item}...")
                try:
                    if is_dir:
                        scp_client.put(local_path, remote_path=REMOTE_DIR, recursive=True)
                    else:
                        scp_client.put(local_path, remote_path=f"{REMOTE_DIR}/{item}")
                except Exception as e:
                    print(f"     [FIGYELEM] {e}")
            else:
                print(f"  [HIÁNYZIK] {item}")
    
    print("[OK] Fájlok feltöltve!")

def setup_server(client):
    """Szerver beállítása."""
    print("\n" + "="*50)
    print("SZERVER BEÁLLÍTÁSA")
    print("="*50)
    
    # 1. Mappa létrehozása
    print("\n[1/5] Mappa létrehozása...")
    run_command(client, f"mkdir -p {REMOTE_DIR}")
    
    # 2. Docker ellenőrzése/telepítése
    print("\n[2/5] Docker ellenőrzése...")
    exit_code, _, _ = run_command(client, "docker --version", show_output=False)
    
    if exit_code != 0:
        print("  -> Docker telepítése...")
        run_command(client, "curl -fsSL https://get.docker.com | sh")
        run_command(client, "systemctl enable docker")
        run_command(client, "systemctl start docker")
    else:
        print("  -> Docker már telepítve")
    
    # 3. Docker Compose ellenőrzése
    print("\n[3/5] Docker Compose ellenőrzése...")
    exit_code, _, _ = run_command(client, "docker compose version", show_output=False)
    
    if exit_code != 0:
        print("  -> Docker Compose telepítése...")
        run_command(client, "apt-get update && apt-get install -y docker-compose-plugin")
    else:
        print("  -> Docker Compose már telepítve")

def deploy_application(client):
    """Alkalmazás deploy."""
    print("\n" + "="*50)
    print("ALKALMAZÁS TELEPÍTÉSE")
    print("="*50)
    
    # 1. Régi konténerek leállítása
    print("\n[1/3] Régi konténerek leállítása...")
    run_command(client, f"cd {REMOTE_DIR} && docker compose down 2>/dev/null || true")
    
    # 2. Új build és indítás
    print("\n[2/3] Új konténerek építése és indítása...")
    exit_code, output, error = run_command(client, f"cd {REMOTE_DIR} && docker compose up -d --build")
    
    if exit_code != 0:
        print(f"[HIBA] Deploy sikertelen!")
        return False
    
    # 3. Várakozás a szolgáltatásokra
    print("\n[3/3] Várakozás a szolgáltatások indulására (30 mp)...")
    time.sleep(30)
    
    return True

def initialize_database(client):
    """Adatbázis inicializálása."""
    print("\n" + "="*50)
    print("ADATBÁZIS INICIALIZÁLÁSA")
    print("="*50)
    
    # Health check
    print("\n[1/2] Health check...")
    for i in range(5):
        exit_code, output, _ = run_command(client, "curl -s http://localhost:3000/health", show_output=False)
        if exit_code == 0 and "healthy" in output:
            print("  -> Szerver fut!")
            break
        print(f"  -> Várakozás... ({i+1}/5)")
        time.sleep(10)
    
    # Lottó adatok letöltése
    print("\n[2/2] Lottó adatok letöltése...")
    exit_code, output, _ = run_command(client, "curl -s -X POST http://localhost:3000/api/admin/init-database")
    
    if "success" in output.lower():
        print("[OK] Adatbázis inicializálva!")
        return True
    else:
        print(f"[FIGYELEM] Válasz: {output[:200]}")
        return False

def check_logs(client):
    """Docker logok ellenőrzése."""
    print("\n" + "="*50)
    print("DOCKER LOGOK")
    print("="*50)
    
    print("\n[LOG] lottery-api konténer:")
    run_command(client, f"cd {REMOTE_DIR} && docker compose logs --tail=30 lottery-api")
    
    print("\n[LOG] postgres konténer:")
    run_command(client, f"cd {REMOTE_DIR} && docker compose logs --tail=10 postgres")
    
    print("\n[STATUS] Konténerek:")
    run_command(client, f"cd {REMOTE_DIR} && docker compose ps")

def test_api(client):
    """API tesztelése."""
    print("\n" + "="*50)
    print("API TESZTELÉSE")
    print("="*50)
    
    tests = [
        ("Health check", "curl -s http://localhost:3000/health"),
        ("Lottó típusok", "curl -s http://localhost:3000/api/admin/lottery-types"),
        ("Ötöslottó predikció", "curl -s 'http://localhost:3000/api/admin/predict/otos-lotto?tickets=4'"),
    ]
    
    for name, cmd in tests:
        print(f"\n[TEST] {name}")
        exit_code, output, _ = run_command(client, cmd, show_output=False)
        
        if exit_code == 0 and output:
            # JSON formázás
            try:
                import json
                data = json.loads(output)
                print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
                if len(output) > 500:
                    print("...")
            except:
                print(output[:300] if output else "[ÜRES VÁLASZ]")
        else:
            print(f"  [HIBA] {output[:100] if output else '[Nincs válasz]'}")

def main():
    print("="*60)
    print("   LOTTERY SYSTEM - AUTOMATIKUS DEPLOY")
    print("="*60)
    print(f"\nCél szerver: {SERVER_USER}@{SERVER_IP}")
    print(f"Távoli mappa: {REMOTE_DIR}")
    
    # Build ellenőrzés
    local_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(local_dir, "dist")
    
    if not os.path.exists(dist_dir):
        print("\n[!] A 'dist' mappa nem létezik!")
        print("    Futtasd először: npm run build")
        sys.exit(1)
    
    # SSH kapcsolat
    client = create_ssh_client()
    
    try:
        # 1. Szerver beállítása
        setup_server(client)
        
        # 2. Fájlok feltöltése
        upload_files(client)
        
        # 3. Deploy
        if deploy_application(client):
            # 4. Logok ellenőrzése
            check_logs(client)
            
            # 5. Inicializálás
            initialize_database(client)
            
            # 6. Tesztelés
            test_api(client)
            
            print("\n" + "="*60)
            print("   DEPLOY SIKERES!")
            print("="*60)
            print(f"\nAPI elérhetőség:")
            print(f"  - Health: http://{SERVER_IP}:3000/health")
            print(f"  - Lottó típusok: http://{SERVER_IP}:3000/api/admin/lottery-types")
            print(f"  - Predikció: http://{SERVER_IP}:3000/api/admin/predict/otos-lotto?tickets=4")
            print(f"\nMobil app konfiguráció:")
            print(f"  API_BASE_URL = 'http://{SERVER_IP}:3000/api'")
        else:
            print("\n[HIBA] Deploy sikertelen!")
            
    finally:
        client.close()
        print("\n[*] SSH kapcsolat lezárva.")

if __name__ == "__main__":
    main()
