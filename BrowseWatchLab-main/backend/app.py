from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import os
import numpy as np
import pandas as pd
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ====================== KONFIGURASI UTAMA ======================

# Konfigurasi File Log
LOG_FILE = "logs/traffic.json"
os.makedirs("logs", exist_ok=True)

# Konfigurasi ML Server
DATA_DIR = "D:/BrowseWatchML"
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

KDD_TRAIN = os.path.join(DATA_DIR, "KDDTrain+.txt")
KDD_TEST = os.path.join(DATA_DIR, "KDDTest+.txt")

# Path Model
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "model_knn.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "model_ensemble.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Path Log
PREDICTION_LOG = os.path.join(RESULTS_DIR, "prediction_log.json")
DETECTION_HISTORY = os.path.join(RESULTS_DIR, "detection_history.json")

# Variabel Global
prediction_log = []
detection_history = {"normal": 0, "attack": 0}
stats_data = {"normal_percent": 0, "attack_percent": 0, "total_predictions": 0}
recent_predictions = []  # Untuk menyimpan prediksi terbaru

# ====================== FUNGSI BANTU ML ======================

def load_and_preprocess():
    """Memuat dan memproses dataset"""
    print("📂 Memuat dan memproses dataset...")
    
    # Definisikan kolom yang digunakan (41 fitur + label + difficulty)
    cols = [f"feature_{i}" for i in range(41)] + ["label", "difficulty"]
    
    # Load data
    df_train = pd.read_csv(KDD_TRAIN, names=cols)
    df_test = pd.read_csv(KDD_TEST, names=cols)

    # Drop kolom yang tidak digunakan
    df_train = df_train.drop(columns=["difficulty", "feature_40"])  # Hapus feature_40
    df_test = df_test.drop(columns=["difficulty", "feature_40"])

    # Encode fitur kategorikal (feature_1, feature_2, feature_3)
    cat_cols = ["feature_1", "feature_2", "feature_3"]
    le_dict = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Gabungkan train+test untuk fitting
        combined = pd.concat([df_train[col], df_test[col]])
        le.fit(combined)
        # Transform data
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        le_dict[col] = le

    # Encode label
    df_train["label"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
    df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Pisahkan features dan label
    X_train = df_train.drop(columns=["label"])
    X_test = df_test.drop(columns=["label"])
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Simpan preprocessing tools
    joblib.dump(le_dict, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_test, y_train, y_test

def update_prediction_history(predictions):
    global prediction_log, recent_predictions, detection_history
    
    # Update prediction log
    for pred in predictions:
        prediction_log.append(pred['prediction'])
        
        # Update recent predictions (max 10)
        recent_predictions.insert(0, pred)
        if len(recent_predictions) > 10:
            recent_predictions.pop()
    
    # Update detection statistics
    normal_count = prediction_log.count(0)
    attack_count = prediction_log.count(1)
    total = len(prediction_log)
    
    detection_history = {
        "normal": normal_count,
        "attack": attack_count
    }
    
    stats_data = {
        "normal_percent": (normal_count / total * 100) if total > 0 else 0,
        "attack_percent": (attack_count / total * 100) if total > 0 else 0,
        "total_predictions": total
    }
    
    # Save to file
    with open(DETECTION_HISTORY, 'w') as f:
        json.dump(detection_history, f)
    
    # Emit real-time update
    socketio.emit('stats_update', {
        'stats': {
            "total_predictions": total,
            "normal_percent": (normal_count / total * 100) if total > 0 else 0,
            "attack_percent": (attack_count / total * 100) if total > 0 else 0,
            "normal_count": normal_count,
            "attack_count": attack_count
        }
    })
    
    # Emit new prediction
    if predictions:
        socketio.emit('new_prediction', {
            'prediction': predictions[0],
            'stats': {
                "total_predictions": total,
                "normal_percent": (normal_count / total * 100) if total > 0 else 0,
                "attack_percent": (attack_count / total * 100) if total > 0 else 0,
                "normal_count": normal_count,
                "attack_count": attack_count
            }
        })
    
    # Save prediction log
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": predictions
    }
    save_prediction_log(log_entry)

def train_models(X_train, y_train):
    """Melatih model machine learning"""
    print("🛠 Melatih model...")
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    ensemble = VotingClassifier(estimators=[("knn", knn), ("rf", rf)], voting="soft")
    ensemble.fit(X_train, y_train)

    joblib.dump(knn, KNN_MODEL_PATH)
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)

    return knn, rf, ensemble

def update_statistics():
    """Memperbarui statistik deteksi"""
    global stats_data
    total = len(prediction_log)
    if total == 0:
        stats_data = {"normal_percent": 0, "attack_percent": 0, "total_predictions": 0}
        return
    
    normal = prediction_log.count(0)
    attack = prediction_log.count(1)

    stats_data["normal_percent"] = (normal / total) * 100
    stats_data["attack_percent"] = (attack / total) * 100
    stats_data["total_predictions"] = total
    
    detection_history["normal"] = normal
    detection_history["attack"] = attack
    
    with open(DETECTION_HISTORY, 'w') as f:
        json.dump(detection_history, f)
    
    # Emit updated stats via websocket
    socketio.emit('stats_update', {'stats': stats_data})

def save_prediction_log(log_entry):
    """Menyimpan log prediksi"""
    try:
        log_data = []
        if os.path.exists(PREDICTION_LOG):
            try:
                with open(PREDICTION_LOG, 'r') as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
                
        log_data.append(log_entry)
        
        with open(PREDICTION_LOG, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Gagal menyimpan log prediksi: {str(e)}")

# Simpan log jaringan
def save_network_log(data):
    """Menyimpan log jaringan dari chrome extension"""
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
        
        # Emit real-time update untuk log baru
        socketio.emit('new_log', data)
        return True
    except Exception as e:
        print(f"⚠️ Gagal menyimpan log jaringan: {str(e)}")
        return False

# ====================== INISIALISASI MODEL ======================

print("🚀 Memulai BrowseWatch Server...")

# Inisialisasi model
if not os.path.exists(ENSEMBLE_MODEL_PATH):
    print("⏳ Model tidak ditemukan, melatih model baru...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    knn_model, rf_model, ensemble_model = train_models(X_train, y_train)
else:
    print("⏳ Memuat model yang sudah ada...")
    knn_model = joblib.load(KNN_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)

# Muat preprocessing tools
scaler = joblib.load(SCALER_PATH)
le_dict = joblib.load(ENCODER_PATH)
cat_cols = ["feature_1", "feature_2", "feature_3"]

# ====================== ROUTES UTAMA ======================

@app.route('/')
def index():
    """Halaman utama dashboard"""
    try:
        with open(LOG_FILE, "r") as f:
            entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        entries = []

    # Tambahkan statistik ML ke context
    return render_template("index.html", 
                         entries=entries,
                         stats=stats_data,
                         detection_history=detection_history)

@app.route('/report', methods=['POST'])
def report():
    """Endpoint untuk menerima laporan traffic"""
    data = request.json
    save_network_log(data)
    return {"status": "Received"}, 200

# ====================== ROUTES ML SERVER ======================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.json:
            return jsonify({"error": "No data provided"}), 400
            
        data = request.json
        if not isinstance(data, list):
            data = [data]

        try:
            input_df = pd.DataFrame(data)
        except Exception as e:
            return jsonify({"error": f"Data conversion failed: {str(e)}"}), 400

        # Pastikan kolom yang diperlukan ada
        expected_features = [f"feature_{i}" for i in range(40)]  # 0-39
        missing_features = [f for f in expected_features if f not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Preprocessing
        try:
            # Encode fitur kategorikal
            cat_cols = ["feature_1", "feature_2", "feature_3"]
            for col in cat_cols:
                if col in input_df.columns:
                    # Handle unknown categories
                    input_df[col] = input_df[col].apply(
                        lambda x: 'unknown' if x not in le_dict[col].classes_ else x)
                    input_df[col] = le_dict[col].transform(input_df[col])
                else:
                    return jsonify({"error": f"Missing categorical feature: {col}"}), 400

            # Urutkan kolom sesuai ekspektasi model
            input_df = input_df[expected_features]
            
            # Scaling
            input_scaled = scaler.transform(input_df)
            
            # Prediksi
            preds = ensemble_model.predict(input_scaled)
            probas = ensemble_model.predict_proba(input_scaled)
            
            results = [{
                "id": i,
                "prediction": int(pred),
                "confidence": float(max(proba)),
                "normal_prob": float(proba[0]),
                "attack_prob": float(proba[1])
            } for i, (pred, proba) in enumerate(zip(preds, probas))]
            
            # Update statistik dan log
            update_prediction_history(results)
            
            return jsonify({"results": results})

        except Exception as e:
            return jsonify({"error": f"Preprocessing/Prediction failed: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/statistics')
def get_statistics():
    """Endpoint untuk mendapatkan statistik"""
    return jsonify({
        "total_predictions": stats_data["total_predictions"],
        "normal_percent": stats_data["normal_percent"],
        "attack_percent": stats_data["attack_percent"],
        "normal_count": detection_history["normal"],
        "attack_count": detection_history["attack"]
    })

# ====================== API ROUTES TAMBAHAN ======================
@app.route('/api/logs/')
def get_logs():
    """Endpoint untuk mendapatkan log jaringan"""
    try:
        with open(LOG_FILE, "r") as f:
            # Read all lines and parse JSON
            entries = []
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue  # Skip invalid lines
            
            # Return the last 20 entries (most recent)
            return jsonify(entries[-20:])
            
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        print(f"Error reading logs: {str(e)}")
        return jsonify([])

@app.route('/api/predictions/')
def get_predictions():
    """Endpoint untuk mendapatkan prediksi terbaru"""
    return jsonify(recent_predictions)

@app.route('/validate')
def validate():
    """Endpoint untuk validasi koneksi server ML"""
    try:
        # Pastikan model dan preprocessor dimuat
        if ensemble_model and scaler and le_dict:
            return jsonify({"status": "success", "message": "Server siap"})
        else:
            return jsonify({"status": "error", "error": "Model tidak dimuat dengan benar"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/api/combined_activities/')
def get_combined_activities():
    try:
        with open(LOG_FILE, "r") as f:
            logs = [json.loads(line) for line in f.readlines()[-20:]]
        
        combined = []
        for log, pred in zip(logs[-len(recent_predictions):], recent_predictions):
            combined.append({
                "activity": log,
                "prediction": pred
            })
        
        return jsonify(combined)
    except Exception as e:
        return jsonify({"error": str(e)})
    
# ====================== SECURITY ENDPOINTS ======================

# Global blocked activities storage
blocked_activities = []

@app.route('/api/block', methods=['POST'])
def block_activity():
    """Block a specific malicious activity"""
    try:
        data = request.json
        pred_id = data.get('prediction_id')
        
        # Find the prediction in recent_predictions
        prediction = next((p for p in recent_predictions if str(p.get('id')) == str(pred_id)), None)
        
        if prediction:
            # Find related log entry
            with open(LOG_FILE, "r") as f:
                logs = [json.loads(line) for line in f.readlines()]
            
            related_log = next((log for log in logs if log.get('request_id') == pred_id), None)
            
            blocked_entry = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': prediction,
                'activity': related_log
            }
            
            blocked_activities.append(blocked_entry)
            
            # Implement actual blocking logic here (e.g., iptables rules)
            # Example: block_ip(related_log.get('ip_address'))
            
            return jsonify({
                'success': True,
                'message': 'Activity blocked successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/block-all', methods=['POST'])
def block_all_attacks():
    """Block all detected attacks"""
    try:
        attack_count = 0
        
        # Get all attack predictions
        attacks = [p for p in recent_predictions if p.get('prediction') == 1]
        
        # Get related logs
        with open(LOG_FILE, "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
        
        # Block each attack
        for attack in attacks:
            related_log = next((log for log in logs if log.get('request_id') == str(attack.get('id'))), None)
            
            if related_log:
                blocked_entry = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction': attack,
                    'activity': related_log
                }
                
                blocked_activities.append(blocked_entry)
                attack_count += 1
                
                # Implement actual blocking logic here
                # Example: block_ip(related_log.get('ip_address'))
        
        return jsonify({
            'success': True,
            'count': attack_count,
            'message': f'Blocked {attack_count} attacks'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/blocked')
def get_blocked_activities():
    """Get list of blocked activities"""
    return jsonify(blocked_activities[-20:])  # Return last 20 blocked items

# ====================== UTILITY FUNCTIONS ======================

def block_ip(ip_address):
    """Actually block an IP address (example implementation)"""
    if not ip_address:
        return False
    
    try:
        # Example for Linux systems
        os.system(f"iptables -A INPUT -s {ip_address} -j DROP")
        print(f"Blocked IP: {ip_address}")
        return True
    except Exception as e:
        print(f"Failed to block IP {ip_address}: {str(e)}")
        return False

# ====================== ROUTES HALAMAN ======================

@app.route('/halaman1')
def halaman1():
    return render_template('hal_1.html')

@app.route('/halaman2')
def halaman2():
    try:
        with open(LOG_FILE, "r") as f:
            entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        entries = []
    return render_template('hal_2.html', entries=entries)

@app.route('/halaman3')
def halaman3():
    return render_template('hal_3.html')

@app.route('/halaman4')
def halaman4():
    return render_template('hal_4.html')

@app.route('/halaman5')
def halaman5():
    try:
        with open(LOG_FILE, "r") as f:
            entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        entries = []
    return render_template('hal_5.html', entries=entries)

# ====================== SOCKET HANDLERS ======================
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Kirim data statistik terkini ke client baru
    emit('stats_update', {
        'stats': {
            "total_predictions": stats_data["total_predictions"],
            "normal_percent": stats_data["normal_percent"],
            "attack_percent": stats_data["attack_percent"],
            "normal_count": detection_history["normal"],
            "attack_count": detection_history["attack"]
        }
    })
    
    # Kirim prediksi terbaru
    emit('initial_predictions', {'predictions': recent_predictions[-10:]})

@socketio.on('request_initial_data')
def handle_initial_data():
    emit('stats_update', {
        'stats': {
            "total_predictions": stats_data["total_predictions"],
            "normal_percent": stats_data["normal_percent"],
            "attack_percent": stats_data["attack_percent"],
            "normal_count": detection_history["normal"],
            "attack_count": detection_history["attack"]
        }
    })
    emit('initial_predictions', {'predictions': recent_predictions[-10:]})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ====================== RUN SERVER ======================

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)