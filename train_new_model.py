# ---------- FILE: train_new_model.py ----------

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# === CONFIGURATION ===
DATA_DIR = "D:/BrowseWatchML"
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Output model paths
MODEL_PATH = os.path.join(MODEL_DIR, "model_knn.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# === LOAD AND MERGE DATA ===
print("📂 Loading datasets...")
cols = [f"feature_{i}" for i in range(41)] + ["label", "difficulty"]
df_train = pd.read_csv(TRAIN_FILE, names=cols)
df_test = pd.read_csv(TEST_FILE, names=cols)

# Gabungkan train dan test untuk fitting encoder
df_full = pd.concat([df_train, df_test], ignore_index=True)
df_full.drop(columns=["difficulty"], inplace=True)

print(f"🔎 Data gabungan: {df_full.shape}")

# === ENCODING ===
cat_cols = ["feature_1", "feature_2", "feature_3"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_full[col])
    le_dict[col] = le

print("✅ Label encoding selesai.")

# Apply encoder ke df_train (yang akan dipakai training model)
df_train.drop(columns=["difficulty"], inplace=True)
for col in cat_cols:
    df_train[col] = le_dict[col].transform(df_train[col])

# Label: normal = 0, lainnya = 1
df_train["label"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)

# === SCALING ===
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.drop(columns=["label"]))
y_train = df_train["label"].values

print(f"📊 Data scaled: {X_train.shape}")

# === TRAIN MODEL ===
print("🛠 Training model KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === SAVE ARTIFACTS ===
joblib.dump(knn, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le_dict, ENCODER_PATH)

print("\n✅ Training selesai! Semua model dan tools disimpan ke folder 'models'.")
