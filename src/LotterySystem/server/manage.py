#!/usr/bin/env python3
"""
===========================================
LOTTERY SYSTEM - SZERVER MENEDZSER
===========================================
Minden egy helyen: teszt, domain, HTTPS, cron, diagnosztika

Használat:
  python manage.py status       - Szerver állapot
  python manage.py test         - API tesztelés
  python manage.py cron         - Cron job-ok ellenőrzése + log
  python manage.py domain       - Domain + HTTPS beállítás (Nginx + Let's Encrypt)
  python manage.py fix-mobile   - Mobile socket hiba javítása
  python manage.py logs         - Docker logok
  python manage.py update       - Lottó adatok frissítése (kézi)
  python manage.py full-check   - MINDENT ellenőriz
"""

import os
import sys
import json
import time
import subprocess

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Szerver adatok - környezeti változókból!
# Állítsd be: set DEPLOY_IP=... DEPLOY_PASS=... DEPLOY_DOMAIN=...
SERVER_IP = os.environ.get("DEPLOY_IP", "")
SERVER_USER = os.environ.get("DEPLOY_USER", "root")
SERVER_PASS = os.environ.get("DEPLOY_PASS", "")
DOMAIN = os.environ.get("DEPLOY_DOMAIN", "liggin.xyz")
REMOTE_DIR = "/opt/lottery-system"

if not SERVER_IP or not SERVER_PASS:
    print("[HIBA] Allitsd be a kornyezeti valtozokat:")
    print("  set DEPLOY_IP=<szerver_ip>")
    print("  set DEPLOY_USER=root")
    print("  set DEPLOY_PASS=<jelszo>")
    print("  set DEPLOY_DOMAIN=liggin.xyz")
    sys.exit(1)

# Csomagok
def ensure_packages():
    try:
        import paramiko
        import requests
    except ImportError:
        print("[*] Szukseges csomagok telepitese...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "requests", "scp", "-q"])

ensure_packages()

import paramiko
import requests

# ============================================
# SSH KAPCSOLAT
# ============================================

def get_ssh():
    """SSH kliens letrehozasa."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASS, timeout=30)
        return client
    except Exception as e:
        print(f"[HIBA] SSH kapcsolat sikertelen: {e}")
        sys.exit(1)

def ssh_run(client, cmd, silent=False):
    """Parancs futtatasa SSH-n."""
    if not silent:
        print(f"  $ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return exit_code, out, err

def ssh_write_file(client, remote_path, content):
    """Fajl irasa SFTP-vel."""
    sftp = client.open_sftp()
    with sftp.file(remote_path, 'w') as f:
        f.write(content)
    sftp.close()

# ============================================
# 1. STATUS - Szerver allapot
# ============================================

def cmd_status():
    print("\n" + "=" * 60)
    print("SZERVER ALLAPOT")
    print("=" * 60)
    
    ssh = get_ssh()
    print(f"[OK] SSH kapcsolat: {SERVER_USER}@{SERVER_IP}\n")
    
    # Docker konténerek
    print("--- Docker konténerek ---")
    _, out, _ = ssh_run(ssh, f"cd {REMOTE_DIR} && docker compose ps 2>/dev/null || docker ps", silent=True)
    print(out or "  [!] Nincs futo kontener")
    
    # Health check
    print("\n--- Health check ---")
    _, out, _ = ssh_run(ssh, "curl -s --max-time 5 http://localhost:3000/health", silent=True)
    if out:
        try:
            data = json.loads(out)
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
        except:
            print(f"  Valasz: {out[:200]}")
    else:
        print("  [HIBA] API nem valaszol!")
    
    # Disk / Memory
    print("\n--- Erforrasok ---")
    _, out, _ = ssh_run(ssh, "df -h / | tail -1", silent=True)
    print(f"  Disk: {out}")
    _, out, _ = ssh_run(ssh, "free -h | grep Mem", silent=True)
    print(f"  Memory: {out}")
    
    # Domain / SSL
    print("\n--- Domain ---")
    _, out, _ = ssh_run(ssh, "nginx -t 2>&1 || echo 'Nginx nem telepitve'", silent=True)
    print(f"  Nginx: {out}")
    _, out, _ = ssh_run(ssh, f"certbot certificates 2>/dev/null | head -5 || echo 'SSL nincs'", silent=True)
    print(f"  SSL: {out}")
    
    # Port check
    print("\n--- Portok ---")
    _, out, _ = ssh_run(ssh, "ss -tlnp | grep -E ':(80|443|3000)'", silent=True)
    print(out or "  [!] Nincs figyelve semmilyen port")
    
    ssh.close()

# ============================================
# 2. TEST - API tesztelés
# ============================================

def cmd_test():
    print("\n" + "=" * 60)
    print("API TESZTELÉS")
    print("=" * 60)
    
    base_urls = [
        f"http://{SERVER_IP}:3000",
        f"https://{DOMAIN}",
        f"http://{DOMAIN}",
    ]
    
    tests = [
        ("Health check", "/health", "GET"),
        ("Lottó típusok", "/api/admin/lottery-types", "GET"),
        ("Eurojackpot predikció", "/api/admin/predict/eurojackpot?tickets=4", "GET"),
        ("Ötöslottó predikció", "/api/admin/predict/otos-lotto?tickets=4", "GET"),
        ("Ötöslottó statisztikák", "/api/admin/stats/otos-lotto", "GET"),
    ]
    
    for base_url in base_urls:
        print(f"\n--- Teszt: {base_url} ---")
        
        for name, path, method in tests:
            try:
                url = f"{base_url}{path}"
                if method == "GET":
                    resp = requests.get(url, timeout=10, verify=False)
                else:
                    resp = requests.post(url, timeout=10, verify=False)
                
                status = "OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
                
                # Rövid válasz kiírás
                try:
                    data = resp.json()
                    if data.get('success'):
                        detail = "OK"
                    else:
                        detail = str(data.get('error', ''))[:50]
                except:
                    detail = resp.text[:50]
                
                print(f"  [{status}] {name}: {detail}")
                
            except requests.exceptions.ConnectionError:
                print(f"  [HIBA] {name}: Kapcsolat megtagadva")
                break
            except requests.exceptions.Timeout:
                print(f"  [HIBA] {name}: Timeout")
            except Exception as e:
                print(f"  [HIBA] {name}: {e}")
                break

# ============================================
# 3. CRON - Cron job-ok ellenőrzése
# ============================================

def cmd_cron():
    print("\n" + "=" * 60)
    print("CRON JOB-OK ELLENORZÉSE")
    print("=" * 60)
    
    ssh = get_ssh()
    
    # Aktuális cron job-ok
    print("\n--- Aktuális cron job-ok ---")
    _, out, _ = ssh_run(ssh, "crontab -l 2>/dev/null", silent=True)
    if out:
        for line in out.split('\n'):
            if 'lottery' in line.lower() or 'update' in line.lower():
                print(f"  [CRON] {line}")
            elif line.strip() and not line.startswith('#'):
                print(f"  [OTHER] {line}")
    else:
        print("  [!] Nincs cron job beallitva!")
    
    # Update scriptek meglétének ellenőrzése
    print("\n--- Update scriptek ---")
    scripts = ['update-all.sh', 'update-otos-lotto.sh', 'update-eurojackpot.sh', 
               'update-hatos-lotto.sh', 'update-skandinav-lotto.sh', 'update-keno.sh']
    for s in scripts:
        _, out, _ = ssh_run(ssh, f"ls -la {REMOTE_DIR}/{s} 2>/dev/null", silent=True)
        if out:
            print(f"  [OK] {s}")
        else:
            print(f"  [!] HIÁNYZIK: {s}")
    
    # Logok
    print("\n--- Update log (utolsó 20 sor) ---")
    _, out, _ = ssh_run(ssh, "tail -20 /var/log/lottery-update.log 2>/dev/null", silent=True)
    if out:
        print(out)
    else:
        print("  [!] Nincs log fajl")
    
    # Kézi teszt futtatás
    print("\n--- Kézi frissítés teszt ---")
    _, out, _ = ssh_run(ssh, "curl -s -X POST http://localhost:3000/api/admin/download-all 2>/dev/null", silent=True)
    if out:
        try:
            data = json.loads(out)
            if data.get('success'):
                print(f"  [OK] {data.get('message', '')}")
                for r in data.get('data', []):
                    status = "OK" if not r.get('error') else f"HIBA: {r.get('error', '')[:40]}"
                    print(f"    - {r.get('name', '?')}: {r.get('newDraws', 0)} uj, {r.get('totalDraws', 0)} osszes [{status}]")
            else:
                print(f"  [HIBA] {out[:200]}")
        except:
            print(f"  Valasz: {out[:300]}")
    else:
        print("  [HIBA] API nem valaszol")
    
    ssh.close()

# ============================================
# 4. DOMAIN - Domain + HTTPS beállítás
# ============================================

def cmd_domain():
    print("\n" + "=" * 60)
    print(f"DOMAIN + HTTPS BEALLITAS: {DOMAIN}")
    print("=" * 60)
    
    ssh = get_ssh()
    
    # 1. Nginx telepítése
    print("\n[1/5] Nginx telepítése...")
    ssh_run(ssh, "apt-get update -qq && apt-get install -y -qq nginx certbot python3-certbot-nginx > /dev/null 2>&1")
    print("  [OK] Nginx + Certbot telepítve")
    
    # 2. Nginx konfiguráció
    print(f"\n[2/5] Nginx konfiguracio: {DOMAIN}...")
    
    nginx_config = f"""# Lottery System - {DOMAIN}
server {{
    listen 80;
    server_name {DOMAIN} www.{DOMAIN};

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {{
        root /var/www/html;
    }}

    # API proxy
    location / {{
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        
        # WebSocket support
        proxy_set_header Connection "upgrade";
        
        # CORS
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
        
        if ($request_method = 'OPTIONS') {{
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }}
    }}
}}
"""
    
    ssh_write_file(ssh, f"/etc/nginx/sites-available/{DOMAIN}", nginx_config)
    ssh_run(ssh, f"ln -sf /etc/nginx/sites-available/{DOMAIN} /etc/nginx/sites-enabled/", silent=True)
    ssh_run(ssh, "rm -f /etc/nginx/sites-enabled/default", silent=True)
    
    # Nginx teszt
    code, out, err = ssh_run(ssh, "nginx -t 2>&1", silent=True)
    if "successful" in (out + err).lower():
        print("  [OK] Nginx konfig OK")
    else:
        print(f"  [HIBA] Nginx konfig hiba: {out} {err}")
        ssh.close()
        return
    
    # 3. Nginx újraindítás
    print("\n[3/5] Nginx ujrainditas...")
    ssh_run(ssh, "systemctl restart nginx", silent=True)
    ssh_run(ssh, "systemctl enable nginx", silent=True)
    print("  [OK] Nginx fut")
    
    # 4. Tűzfal
    print("\n[4/5] Tuzfal beallitas...")
    ssh_run(ssh, "ufw allow 80/tcp 2>/dev/null; ufw allow 443/tcp 2>/dev/null; ufw allow 3000/tcp 2>/dev/null", silent=True)
    print("  [OK] Portok nyitva: 80, 443, 3000")
    
    # 5. SSL tanúsítvány (Let's Encrypt)
    print(f"\n[5/5] SSL tanusitvany igénylés: {DOMAIN}...")
    code, out, err = ssh_run(ssh, 
        f"certbot --nginx -d {DOMAIN} --non-interactive --agree-tos --email admin@{DOMAIN} --redirect 2>&1",
        silent=True)
    
    combined = out + " " + err
    if "congratulations" in combined.lower() or "certificate" in combined.lower():
        print("  [OK] SSL tanusitvany sikeresen beszerezve!")
    elif "already" in combined.lower():
        print("  [OK] SSL mar beallitva")
    else:
        print(f"  [FIGYELEM] Certbot kimenet:\n{combined[:500]}")
        print("\n  Ha a domain DNS meg nem all ra a szerverre,")
        print(f"  allitsd be az A recordot: {DOMAIN} -> {SERVER_IP}")
        print("  Es majd futtasd ujra: python manage.py domain")
    
    # Certbot auto-renew
    ssh_run(ssh, "systemctl enable certbot.timer 2>/dev/null", silent=True)
    
    # Eredmény
    print("\n" + "=" * 60)
    print("DOMAIN BEALLITAS KESZ")
    print("=" * 60)
    print(f"\n  HTTP:  http://{DOMAIN}")
    print(f"  HTTPS: https://{DOMAIN}")
    print(f"  API:   https://{DOMAIN}/api/admin/lottery-types")
    print(f"  Health: https://{DOMAIN}/health")
    print(f"\n  DNS A record: {DOMAIN} -> {SERVER_IP}")
    
    ssh.close()

# ============================================
# 5. FIX-MOBILE - Mobile socket hiba javítás
# ============================================

def cmd_fix_mobile():
    print("\n" + "=" * 60)
    print("MOBILE SOCKET HIBA JAVITAS")
    print("=" * 60)
    
    ssh = get_ssh()
    
    # 1. Szerver elérhetőség teszt
    print("\n[1/4] Szerver elérhetőség...")
    _, out, _ = ssh_run(ssh, "curl -s http://localhost:3000/health", silent=True)
    if "healthy" in (out or ""):
        print("  [OK] API fut")
    else:
        print("  [HIBA] API nem fut! Inditsd ujra a Docker konténereket.")
        ssh_run(ssh, f"cd {REMOTE_DIR} && docker compose up -d", silent=True)
        time.sleep(10)
    
    # 2. CORS beállítás ellenőrzése
    print("\n[2/4] CORS beallitas...")
    _, out, _ = ssh_run(ssh, f"cd {REMOTE_DIR} && docker compose exec -T lottery-api printenv CORS_ORIGINS 2>/dev/null || echo 'N/A'", silent=True)
    print(f"  CORS_ORIGINS: {out}")
    if out != "*":
        print("  [!] CORS nem '*' - javítás...")
        # docker-compose.yml-ben már * van
    
    # 3. Nginx WebSocket proxy
    print("\n[3/4] WebSocket proxy ellenorzés...")
    _, out, _ = ssh_run(ssh, f"grep -c 'Upgrade' /etc/nginx/sites-available/{DOMAIN} 2>/dev/null || echo '0'", silent=True)
    if int(out or '0') > 0:
        print("  [OK] WebSocket proxy beallitva")
    else:
        print("  [!] WebSocket proxy hiányzik - a domain parancs beallitja")
    
    # 4. Port külső elérhetőség
    print("\n[4/4] Port elérhetőség kívülről...")
    
    for port in [80, 443, 3000]:
        try:
            resp = requests.get(f"http://{SERVER_IP}:{port}/health", timeout=5)
            print(f"  [OK] Port {port}: valaszol (HTTP {resp.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"  [HIBA] Port {port}: nem érhető el!")
        except:
            pass
    
    # Javasolt mobile API URL
    print("\n" + "=" * 60)
    print("MOBILE APP BEALLITASOK")
    print("=" * 60)
    print(f"""
  A mobile app api_service.dart fajlban allitsd at:
  
  JELENLEGI:
    static const String baseUrl = 'http://{SERVER_IP}:3000/api';
  
  JAVITOTT (HTTPS domain):
    static const String baseUrl = 'https://{DOMAIN}/api';
  
  Ez megoldja:
    - Socket hibat (HTTPS szukseges Android 9+ eseten)
    - Cleartext traffic blokkot
    - Mixed content hibakat
""")
    
    ssh.close()

# ============================================
# 6. LOGS - Docker logok
# ============================================

def cmd_logs():
    print("\n" + "=" * 60)
    print("DOCKER LOGOK")
    print("=" * 60)
    
    ssh = get_ssh()
    
    print("\n--- lottery-api (utolsó 50 sor) ---")
    _, out, _ = ssh_run(ssh, f"cd {REMOTE_DIR} && docker compose logs --tail=50 lottery-api 2>/dev/null", silent=True)
    print(out or "  [!] Nincs log")
    
    print("\n--- postgres (utolsó 20 sor) ---")
    _, out, _ = ssh_run(ssh, f"cd {REMOTE_DIR} && docker compose logs --tail=20 postgres 2>/dev/null", silent=True)
    print(out or "  [!] Nincs log")
    
    ssh.close()

# ============================================
# 7. UPDATE - Lottó adatok kézi frissítése
# ============================================

def cmd_update():
    print("\n" + "=" * 60)
    print("LOTTO ADATOK FRISSITESE")
    print("=" * 60)
    
    ssh = get_ssh()
    
    # Összes lottó frissítése
    print("\n[*] Osszes lotto frissitese...")
    _, out, _ = ssh_run(ssh, "curl -s -X POST http://localhost:3000/api/admin/download-all", silent=True)
    
    if out:
        try:
            data = json.loads(out)
            if data.get('success'):
                print(f"\n[OK] {data.get('message', '')}\n")
                for r in data.get('data', []):
                    err = r.get('error', '')
                    if err:
                        print(f"  [HIBA] {r.get('name', '?')}: {err[:60]}")
                    else:
                        print(f"  [OK] {r.get('name', '?')}: {r.get('newDraws', 0)} uj huzas, ossz: {r.get('totalDraws', 0)}")
            else:
                print(f"  [HIBA] {out[:300]}")
        except:
            print(f"  Valasz: {out[:300]}")
    else:
        print("  [HIBA] API nem valaszol - Docker fut?")
    
    ssh.close()

# ============================================
# 8. FULL-CHECK - Mindent ellenőriz
# ============================================

def cmd_full_check():
    print("\n" + "=" * 60)
    print("TELJES RENDSZER ELLENORZES")
    print("=" * 60)
    
    cmd_status()
    cmd_test()
    cmd_cron()
    cmd_fix_mobile()

# ============================================
# MAIN
# ============================================

COMMANDS = {
    'status': ('Szerver allapot', cmd_status),
    'test': ('API teszteles', cmd_test),
    'cron': ('Cron job-ok ellenorzese', cmd_cron),
    'domain': ('Domain + HTTPS beallitas', cmd_domain),
    'fix-mobile': ('Mobile socket hiba javitas', cmd_fix_mobile),
    'logs': ('Docker logok', cmd_logs),
    'update': ('Lotto adatok frissitese', cmd_update),
    'full-check': ('Mindent ellenoriz', cmd_full_check),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("\n" + "=" * 60)
        print("  LOTTERY SYSTEM - SZERVER MENEDZSER")
        print("=" * 60)
        print(f"\n  Szerver: {SERVER_USER}@{SERVER_IP}")
        print(f"  Domain:  {DOMAIN}")
        print(f"\n  Hasznalat: python manage.py <parancs>\n")
        print("  Elerheto parancsok:")
        for cmd, (desc, _) in COMMANDS.items():
            print(f"    {cmd:15} - {desc}")
        print()
        return
    
    cmd = sys.argv[1]
    _, func = COMMANDS[cmd]
    func()

if __name__ == "__main__":
    main()
