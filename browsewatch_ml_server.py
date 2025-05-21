import os
import json
import pandas as pd
import joblib
import datetime
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# === CONFIGURATION ===
DATA_DIR = "D:/BrowseWatchML"
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

KDD_TRAIN = os.path.join(DATA_DIR, "KDDTrain+.txt")
KDD_TEST = os.path.join(DATA_DIR, "KDDTest+.txt")

# Model Paths
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "model_knn.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "model_ensemble.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Log Paths
PREDICTION_LOG = os.path.join(RESULTS_DIR, "prediction_log.json")
DETECTION_HISTORY = os.path.join(RESULTS_DIR, "detection_history.json")

# Initialize global variables
prediction_log = []
detection_history = {"normal": 0, "attack": 0}
stats_data = {"normal_percent": 0, "attack_percent": 0, "total_predictions": 0}

# === DATA HANDLING ===
def load_and_preprocess():
    print("📂 Loading and preprocessing dataset...")
    cols = [f"feature_{i}" for i in range(41)] + ["label", "difficulty"]
    df_train = pd.read_csv(KDD_TRAIN, names=cols)
    df_test = pd.read_csv(KDD_TEST, names=cols)

    # Gabungkan untuk fit encoder
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    # Drop kolom tidak perlu
    df_train.drop(columns=["difficulty"], inplace=True)
    df_test.drop(columns=["difficulty"], inplace=True)
    df_full.drop(columns=["difficulty"], inplace=True)

    # Encode fitur kategorikal
    cat_cols = ["feature_1", "feature_2", "feature_3"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df_full[col])
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        le_dict[col] = le

    # Label encoding
    df_train["label"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
    df_test["label"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.drop(columns=["label"]))
    X_test = scaler.transform(df_test.drop(columns=["label"]))

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # Save preprocessing tools
    joblib.dump(le_dict, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    print("🛠 Training models...")
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    
    # Train individual models
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Train ensemble model
    ensemble = VotingClassifier(estimators=[("knn", knn), ("rf", rf)], voting="soft")
    ensemble.fit(X_train, y_train)

    # Save models
    joblib.dump(knn, KNN_MODEL_PATH)
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)

    return knn, rf, ensemble

def update_statistics():
    global stats_data
    if not prediction_log:
        stats_data["normal_percent"] = 0
        stats_data["attack_percent"] = 0
        stats_data["total_predictions"] = 0
        return
    
    total = len(prediction_log)
    normal = prediction_log.count(0)
    attack = prediction_log.count(1)

    stats_data["normal_percent"] = (normal / total) * 100
    stats_data["attack_percent"] = (attack / total) * 100
    stats_data["total_predictions"] = total
    
    # Update detection history
    detection_history["normal"] = normal
    detection_history["attack"] = attack
    
    # Save detection history to file
    with open(DETECTION_HISTORY, 'w') as f:
        json.dump(detection_history, f)

# === INITIALIZE SERVER ===
print("🚀 Starting BrowseWatch ML Server...")

# Initialize models
if not os.path.exists(ENSEMBLE_MODEL_PATH):
    X_train, X_test, y_train, y_test = load_and_preprocess()
    knn_model, rf_model, ensemble_model = train_models(X_train, y_train)
else:
    # Load all models
    knn_model = joblib.load(KNN_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)

scaler = joblib.load(SCALER_PATH)
le_dict = joblib.load(ENCODER_PATH)
cat_cols = ["feature_1", "feature_2", "feature_3"]

app = Flask(__name__)

# === ROUTES ===
@app.route('/')
def home():
    return '''
    <h2>BrowseWatch Lab - Machine Learning Server</h2>
    <p>Use <code>/predict</code> to POST network activity JSON data and get predictions.</p>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame(data)

    try:
        # --- PREPROCESSING ---
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = le_dict[col].transform(input_df[col])

        # ❗ Drop feature_40 karena tidak dipakai untuk prediksi
        if "feature_40" in input_df.columns:
            input_df = input_df.drop(columns=["feature_40"])

        # --- NORMALISASI ---
        input_scaled = scaler.transform(input_df)

        # --- PREDIKSI ---
        # Use ensemble model for best results
        preds = ensemble_model.predict(input_scaled)

        # --- UPDATE statistik ---
        prediction_log.extend(preds.tolist())
        update_statistics()

        # Log timestamp and prediction
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "prediction": preds.tolist(),
            "data_points": len(input_df)
        }
        
        # Append to prediction log file
        try:
            if os.path.exists(PREDICTION_LOG):
                with open(PREDICTION_LOG, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
                
            log_data.append(log_entry)
            
            with open(PREDICTION_LOG, 'w') as f:
                json.dump(log_data, f)
        except Exception as e:
            print(f"Error saving prediction log: {str(e)}")

        return jsonify({"prediction": preds.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/statistics')
def get_statistics():
    return jsonify({
        "total_predictions": stats_data["total_predictions"],
        "normal_percent": stats_data["normal_percent"],
        "attack_percent": stats_data["attack_percent"],
        "normal_count": detection_history["normal"],
        "attack_count": detection_history["attack"]
    })

# === RUN APP ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)