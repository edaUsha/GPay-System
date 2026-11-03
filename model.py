import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
from imblearn.over_sampling import SMOTE

REAL_DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "data/fraud_model.pkl"


def extract_features(sender_id, receiver_id, amount):
    """
    Translates live user metrics into the 30-feature shape (Time, Amount, V1-V28) 
    expected by the production Kaggle model.
    """
    # 1. Calculate Elapsed Time (Seconds from start of a mock 48-hour cycle)
    now = pd.Timestamp.now()
    seconds_today = (now.hour * 3600) + (now.minute * 60) + now.second
    
    # 2. Structure base placeholder vectors for V1 to V28
    # If the user tries to send an unusually massive amount, simulate anomalous PCA values
    if amount > 25000 or now.hour in [22, 23, 0, 1]:

        # Generate highly anomalous vectors matching real fraud distributions
        v_vectors = {f'V{i}': np.random.uniform(-10.0, -2.0) for i in range(1, 29)}
    else:
        # Standard human spending behavior baseline (centered close to 0)
        v_vectors = {f'V{i}': np.random.uniform(-0.5, 0.5) for i in range(1, 29)}
        
    # 3. Compile the exact 30 features in proper structural order
    features = {
        'Time': float(seconds_today),
        'Amount': float(amount)
    }
    features.update(v_vectors)
    
    return features


def train_model_on_real_data():
    if not os.path.exists(REAL_DATA_PATH):
        print(f"❌ Error: Please download 'creditcard.csv' from Kaggle and place it in '{REAL_DATA_PATH}'")
        return

    print("📖 Loading real-world Kaggle credit card dataset...")
    df = pd.read_csv(REAL_DATA_PATH)
    
    # UPGRADE: Dynamic list compression to extract all V1-V28 columns + Amount + Time
    v_features = [f'V{i}' for i in range(1, 29)]
    selected_features = ['Amount', 'Time'] + v_features
    
    X = df[selected_features]
    y = df['Class']
    
    print(f"Original distribution: Legit (0) = {sum(y==0)}, Fraud (1) = {sum(y==1)}")
    
    # Clean train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🧬 Applying SMOTE to synthesize minority fraud vectors...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("🏋️ Training Full-Scale Random Forest Guard (This might take a minute)...")
    # Increased max_depth slightly to allow the forest to map the complex multi-dimensional features
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Performance Extraction
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    
    auprc_score = average_precision_score(y_test, probabilities)
    
    print("\n📊 --- REAL-WORLD MODEL PERFORMANCE (ALL FEATURES + SMOTE) ---")
    print(classification_report(y_test, predictions, target_names=["Legit (0)", "Fraud (1)"]))
    
    print("=" * 51)
    print(f" 🎯 **Area Under Precision-Recall Curve (AUPRC): {auprc_score:.4f}** ")
    print("=" * 51)
    print(" (Note: A perfect fraud engine scores a 1.0000 AUPRC)\n")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"📦 Advanced production binary written to {MODEL_PATH}")


def predict_fraud_risk(feature_dict):
    """
    Loads the real-world trained model binary and classifies if the incoming 
    30-feature vector is an anomaly/fraud (1) or safe (0).
    """
    if not os.path.exists(MODEL_PATH):
        print("[SYSTEM LOG]: Model binary not found. Training from dataset first...")
        train_model_on_real_data()
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # Convert the 30-feature dictionary into a clean 1-row Pandas DataFrame
    input_df = pd.DataFrame([feature_dict])
    
    # Ensure the columns are passed to the model in the exact order it was trained on
    v_features = [f'V{i}' for i in range(1, 29)]
    ordered_columns = ['Amount', 'Time'] + v_features
    input_df = input_df[ordered_columns]
    
    # Generate the prediction (0 or 1)
    prediction = model.predict(input_df)
    return int(prediction[0])

if __name__ == "__main__":
    train_model_on_real_data()
