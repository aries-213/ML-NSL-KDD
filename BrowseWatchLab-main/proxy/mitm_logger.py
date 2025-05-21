import json
import redis
import os
import sys
from mitmproxy import http

# Tambahkan folder root BrowseWatchLab ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fungsi ML deteksi mencurigakan
from backend.ml.analyzer import detect_suspicious

# Koneksi ke Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def response(flow: http.HTTPFlow):
    # Tangkap request dari MITMProxy
    log = {
        'url': flow.request.pretty_url,
        'method': flow.request.method,
        'status_code': flow.response.status_code
    }

    # Push log ke Redis list
    r.rpush('browsing_logs', json.dumps(log))

    # Cek apakah aktivitas mencurigakan
    if detect_suspicious(log):
        alert = {
            "message": f"⚠️ Suspicious activity detected: {log['url']}",
            "url": log['url'],
            "method": log['method'],
            "status": log['status_code']
        }
        r.rpush('alerts', json.dumps(alert))
