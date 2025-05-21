import pandas as pd
import requests
import time
import json
import random
import datetime
import platform
import socket
import uuid
import os
from urllib.parse import urljoin

# ----------------- CONFIGURATION -----------------
SERVER_URL = "http://localhost:5000"  # Base URL
PREDICT_ENDPOINT = urljoin(SERVER_URL, "/predict")
REPORT_ENDPOINT = urljoin(SERVER_URL, "/report")
VALIDATE_ENDPOINT = urljoin(SERVER_URL, "/validate")
SEND_INTERVAL = 2  # seconds between data sending
NUM_SAMPLES = 10  # number of samples to send
TIMEOUT = 10  # request timeout in seconds

# ----------------- SAMPLE DATA -----------------
SAMPLE_HOSTNAMES = [
    "google.com", "facebook.com", "youtube.com", "netflix.com", 
    "amazon.com", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "stackoverflow.com", "reddit.com"
]

SAMPLE_OS = [
    "Windows", "MacOS", "Linux", "Android", "iOS"
]

SAMPLE_OS_VERSIONS = [
    "10", "11", "12", "Monterey", "Ventura", "Sonoma", "Ubuntu 22.04", 
    "Debian 11", "Android 12", "iOS 16"
]

SAMPLE_STATUS_CODES = [
    200, 301, 302, 304, 400, 401, 403, 404, 500
]

SAMPLE_METHODS = [
    "GET", "POST", "PUT", "DELETE"
]

# ----------------- HELPER FUNCTIONS -----------------
def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def get_random_mac():
    """Generate random MAC address"""
    return ':'.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(6)])

def get_random_host_info():
    """Generate random host information"""
    hostname = random.choice(SAMPLE_HOSTNAMES)
    url = f"https://{hostname}/{random.choice(['index.html', 'api/data', 'login', 'dashboard', 'products'])}"
    
    return {
        "hostname": hostname,
        "url": url,
        "ip_address": '.'.join([str(random.randint(1, 255)) for _ in range(4)]),
        "os": random.choice(SAMPLE_OS),
        "os_version": random.choice(SAMPLE_OS_VERSIONS),
        "mac": get_random_mac(),
        "method": random.choice(SAMPLE_METHODS),
        "status_code": random.choice(SAMPLE_STATUS_CODES),
        "timestamp": datetime.datetime.now().isoformat()
    }

def validate_server_connection():
    """Check if server is reachable"""
    try:
        print("🔌 Checking server connection...")
        response = requests.get(VALIDATE_ENDPOINT, timeout=TIMEOUT)
        
        if response.status_code == 200:
            status = response.json()
            if status.get('status') == 'success':
                print("✅ Server is ready")
                return True
            else:
                print(f"⚠️ Server error: {status.get('error', 'Unknown error')}")
        else:
            print(f"❌ Server returned HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to server: {str(e)}")
    
    return False

def send_sample_network_data(sample_num):
    """Send sample network data to report endpoint"""
    data = get_random_host_info()
    
    try:
        # Print sample data for debugging
        if sample_num == 1:
            print("\n🔍 Sample network data structure being sent:")
            print(json.dumps(data, indent=2))
        
        response = requests.post(
            REPORT_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            print(f"[{sample_num}] ✅ Network data sent successfully")
            return True
        else:
            print(f"[{sample_num}] ❌ Failed to send network data - HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[{sample_num}] ❌ Exception while sending network data: {str(e)}")
        return False

def prepare_ml_dataset():
    sample_data = {}
    for i in range(40):  # Hanya 40 fitur (0-39)
        if i == 1:
            sample_data[f"feature_{i}"] = random.choice(["tcp", "udp", "icmp"])
        elif i == 2:
            sample_data[f"feature_{i}"] = random.choice(["http", "ftp", "smtp", "ssh"])
        elif i == 3:
            sample_data[f"feature_{i}"] = random.choice(["SF", "REJ", "S0", "RSTR"])
        else:
            sample_data[f"feature_{i}"] = random.uniform(0, 1)
    return sample_data

def send_ml_prediction_data(sample_num):
    """Send sample ML prediction data to predict endpoint"""
    data = prepare_ml_dataset()
    
    try:
        # Print sample data for debugging
        if sample_num == 1:
            print("\n🔍 Sample ML data structure being sent:")
            print(json.dumps(data, indent=2))
        
        response = requests.post(
            PREDICT_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=[data],  # Server expects a list of data points
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[{sample_num}] ✅ ML prediction sent successfully - Result: {result.get('results', 'N/A')}")
            return True
        else:
            print(f"[{sample_num}] ❌ Failed to get ML prediction - HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[{sample_num}] ❌ Exception while sending ML data: {str(e)}")
        return False
    
def get_random_host_info():
    """Generate random host information"""
    hostname = random.choice(SAMPLE_HOSTNAMES)
    url = f"https://{hostname}/{random.choice(['index.html', 'api/data', 'login', 'dashboard', 'products'])}"
    
    return {
        "hostname": hostname,
        "url": url,
        "ip_address": '.'.join([str(random.randint(1, 255)) for _ in range(4)]),
        "os": random.choice(SAMPLE_OS),
        "os_version": random.choice(SAMPLE_OS_VERSIONS),
        "mac": get_random_mac(),
        "method": random.choice(SAMPLE_METHODS),
        "status_code": random.choice(SAMPLE_STATUS_CODES),
        "timestamp": datetime.datetime.now().isoformat()  # Ensure consistent timestamp format
    }

# ----------------- MAIN EXECUTION -----------------
def main():
    print("\n🚀 BrowseWatch - Synchronized Data Sender")
    print("=" * 50)
    
    # First validate server connection
    if not validate_server_connection():
        print("\n⚠️ Please ensure the server is running before continuing")
        choice = input("Do you want to continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    print(f"\n🔄 Preparing to send {NUM_SAMPLES} samples of data to server...")
    print("This will send both network traffic data and ML prediction data.")
    
    network_success = 0
    network_fail = 0
    ml_success = 0
    ml_fail = 0
    
    try:
        for i in range(1, NUM_SAMPLES + 1):
            print(f"\n📤 Sending sample #{i}...")
            
            # Send network traffic data
            if send_sample_network_data(i):
                network_success += 1
            else:
                network_fail += 1
            
            # Send ML prediction data
            if send_ml_prediction_data(i):
                ml_success += 1
            else:
                ml_fail += 1
            
            # Wait before sending next sample (except last one)
            if i < NUM_SAMPLES:
                print(f"⏳ Waiting {SEND_INTERVAL} seconds before next sample...")
                time.sleep(SEND_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Final Results:")
    print(f"✅ Successful network data transmissions: {network_success}")
    print(f"❌ Failed network data transmissions: {network_fail}")
    print(f"✅ Successful ML predictions: {ml_success}")
    print(f"❌ Failed ML predictions: {ml_fail}")
    print(f"📈 Overall success rate: {round((network_success + ml_success)/(NUM_SAMPLES*2)*100, 2)}%")
    
    # Additional debug suggestion
    if network_fail > 0 or ml_fail > 0:
        print("\nℹ️  If you're having failures, try:")
        print("- Check if the Flask server is running")
        print("- Verify the SERVER_URL is correct")
        print("- Examine the server logs for errors")

if __name__ == "__main__":
    main()