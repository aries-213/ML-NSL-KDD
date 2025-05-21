import os
import shutil
import sqlite3
from datetime import datetime, timedelta

def get_history_windows():
    base_path = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
    history_entries = []

    if not os.path.exists(base_path):
        return []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == "History":
                profile_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                temp_path = os.path.join(os.getenv("TEMP"), f"History_{profile_name}")
                try:
                    shutil.copy2(os.path.join(root, file), temp_path)
                    conn = sqlite3.connect(temp_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT url, title, last_visit_time FROM urls ORDER by last_visit_time DESC LIMIT 5")
                    rows = cursor.fetchall()
                    for row in rows:
                        history_entries.append({
                            "profile": profile_name,
                            "url": row[0],
                            "title": row[1],
                            "time": convert_chrome_time(row[2])
                        })
                except Exception as e:
                    print(f"[!] Error processing {profile_name}: {e}")
                finally:
                    try:
                        conn.close()
                        os.remove(temp_path)
                    except:
                        pass
    return history_entries
def convert_chrome_time(chrome_time):
    if chrome_time == 0:
        return "N/A"
    epoch_start = datetime(1601, 1, 1)
    return (epoch_start + timedelta(microseconds=chrome_time)).strftime("%Y-%m-%d %H:%M:%S")


def get_chrome_history():
    return get_history_windows()