#!/usr/bin/env python3
"""
Setup automatic lottery data updates via cron
Lottónként külön ütemezéssel a húzások után
"""

import paramiko
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
SERVER_IP = os.environ.get("DEPLOY_IP", "")
SERVER_USER = os.environ.get("DEPLOY_USER", "root")
SERVER_PASS = os.environ.get("DEPLOY_PASS", "")

if not SERVER_IP or not SERVER_PASS:
    print("[HIBA] set DEPLOY_IP=... DEPLOY_PASS=... kornyezeti valtozok szuksegesek!")
    sys.exit(1)

# Húzási időpontok (magyar idő) és frissítési ütemezés
# A cron időzónája UTC, ezért 1 órát vonunk le (téli idő)
LOTTERY_SCHEDULES = {
    'otos-lotto': {
        'name': 'Ötöslottó',
        'draw': 'Szombat 18:45',
        # Szombat 18:45 -> Vasárnap 07:00 UTC (08:00 HU)
        'cron': '0 7 * * 0',
    },
    'hatos-lotto': {
        'name': 'Hatoslottó',
        'draw': 'Csütörtök 20:50, Vasárnap 16:00',
        # Csütörtök 20:50 -> Péntek 07:00 UTC
        # Vasárnap 16:00 -> Hétfő 07:00 UTC
        'cron': '0 7 * * 1,5',
    },
    'skandinav-lotto': {
        'name': 'Skandináv lottó',
        'draw': 'Szerda 20:50',
        # Szerda 20:50 -> Csütörtök 07:00 UTC
        'cron': '0 7 * * 4',
    },
    'eurojackpot': {
        'name': 'Eurojackpot',
        'draw': 'Kedd és Péntek 20:00-21:00',
        # Kedd 21:00 -> Szerda 07:00 UTC
        # Péntek 21:00 -> Szombat 07:00 UTC
        'cron': '0 7 * * 3,6',
    },
    'keno': {
        'name': 'Kenó',
        'draw': 'Naponta többször',
        # Minden nap 07:00 UTC
        'cron': '0 7 * * *',
    },
}

def setup_cron():
    print("=" * 60)
    print("LOTTERY AUTO-UPDATE CRON SETUP")
    print("=" * 60)
    
    print(f"\nCsatlakozás: {SERVER_USER}@{SERVER_IP}")
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASS)
    
    print("[OK] SSH kapcsolat létrejött!\n")
    
    # 1. Frissítő scriptek létrehozása lottónként
    print("=" * 60)
    print("FRISSÍTŐ SCRIPTEK LÉTREHOZÁSA")
    print("=" * 60)
    
    sftp = client.open_sftp()
    
    for lottery_key, schedule in LOTTERY_SCHEDULES.items():
        script_content = f"""#!/bin/bash
# Auto-update script for {schedule['name']}
# Húzás: {schedule['draw']}

LOG_FILE="/var/log/lottery-update.log"
API_URL="http://localhost:3000/api/admin/download/{lottery_key}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [{schedule['name']}] Starting update..." >> $LOG_FILE
RESPONSE=$(curl -s -X POST $API_URL 2>&1)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [{schedule['name']}] Response: $RESPONSE" >> $LOG_FILE
echo "---" >> $LOG_FILE
"""
        
        script_path = f'/opt/lottery-system/update-{lottery_key}.sh'
        with sftp.file(script_path, 'w') as f:
            f.write(script_content)
        
        # Futtathatóvá tétel
        client.exec_command(f'chmod +x {script_path}')
        
        print(f"  [+] {schedule['name']}: {script_path}")
    
    # Master script ami minden lottót frissít
    master_script = """#!/bin/bash
# Master update script - all lotteries
LOG_FILE="/var/log/lottery-update.log"
API_URL="http://localhost:3000/api/admin/download-all"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MASTER] Starting full update..." >> $LOG_FILE
RESPONSE=$(curl -s -X POST $API_URL 2>&1)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MASTER] Response: $RESPONSE" >> $LOG_FILE
echo "---" >> $LOG_FILE
"""
    with sftp.file('/opt/lottery-system/update-all.sh', 'w') as f:
        f.write(master_script)
    client.exec_command('chmod +x /opt/lottery-system/update-all.sh')
    print(f"  [+] Master: /opt/lottery-system/update-all.sh")
    
    sftp.close()
    
    # 2. Cron job-ok beállítása
    print("\n" + "=" * 60)
    print("CRON JOB-OK BEÁLLÍTÁSA")
    print("=" * 60)
    print("\nHúzási időpontok (magyar idő):")
    
    cron_lines = []
    for lottery_key, schedule in LOTTERY_SCHEDULES.items():
        cron_line = f"{schedule['cron']} /opt/lottery-system/update-{lottery_key}.sh"
        cron_lines.append(cron_line)
        print(f"  {schedule['name']:20} | {schedule['draw']:30} | {schedule['cron']}")
    
    # Régi lottery cron törlése és újak hozzáadása
    cron_content = '\n'.join(cron_lines)
    
    stdin, stdout, stderr = client.exec_command(
        f'(crontab -l 2>/dev/null | grep -v "lottery-system/update"; echo "{cron_content}") | crontab -'
    )
    stdout.read()
    
    # 3. Ellenőrzés
    print("\n" + "=" * 60)
    print("AKTUÁLIS CRON JOBS")
    print("=" * 60)
    
    stdin, stdout, stderr = client.exec_command('crontab -l')
    cron_output = stdout.read().decode()
    print(cron_output)
    
    # 4. Log fájl létrehozása
    client.exec_command('touch /var/log/lottery-update.log')
    
    # 5. Teszt futtatás
    print("=" * 60)
    print("TESZT FUTTATÁS")
    print("=" * 60)
    
    print("\nMaster script futtatása...")
    stdin, stdout, stderr = client.exec_command('/opt/lottery-system/update-all.sh')
    stdout.read()
    
    stdin, stdout, stderr = client.exec_command('tail -10 /var/log/lottery-update.log')
    print("\nLog (utolsó 10 sor):")
    print(stdout.read().decode())
    
    client.close()
    
    print("=" * 60)
    print("SETUP KÉSZ!")
    print("=" * 60)
    print("""
Összefoglaló:
  - Ötöslottó:     Vasárnap 08:00 (húzás: Szombat 18:45)
  - Hatoslottó:    Hétfő és Péntek 08:00 (húzás: Csüt 20:50, Vas 16:00)
  - Skandináv:     Csütörtök 08:00 (húzás: Szerda 20:50)
  - Eurojackpot:   Szerda és Szombat 08:00 (húzás: Kedd/Péntek 20:00-21:00)
  - Kenó:          Minden nap 08:00 (húzás: naponta)
  
Log fájl: /var/log/lottery-update.log
    """)

if __name__ == "__main__":
    setup_cron()
