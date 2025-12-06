from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load Model, Scaler, Features, Suspicious Ports
try:
    model = joblib.load('network_security_model.pkl')
    scaler = joblib.load('port_enhanced_scaler.pkl')
    X_columns = joblib.load('X_columns.pkl')
    SUSPICIOUS_PORTS = joblib.load('suspicious_ports.pkl')
    print("Model + scaler loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model, scaler, X_columns, SUSPICIOUS_PORTS = None, None, [], []

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(data_dict):
    if isinstance(data_dict, dict):
        data_dict = [data_dict]

    df = pd.DataFrame(0, index=[0], columns=X_columns)

    for record in data_dict:
        for key, value in record.items():
            if key in df.columns:
                df.at[0, key] = value

    # Boolean → int
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Add suspicious_src_port if missing
    if 'src_port' in df.columns:
        df['suspicious_src_port'] = df['src_port'].isin(SUSPICIOUS_PORTS).astype(int)

    # Ensure all columns exist
    missing_cols = set(X_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    df = df[X_columns]

    # Scale
    scaled = scaler.transform(df)
    return scaled

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        X_scaled = preprocess_input(data)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        result = {
            "prediction": int(pred),
            "probability_benign": float(proba[0]),
            "probability_malicious": float(proba[1]),
            "status": "Benign" if pred == 0 else "Malicious"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    results = []
    for record in data.get("records", []):
        X_scaled = preprocess_input(record)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        results.append({
            "prediction": int(pred),
            "probability_benign": float(proba[0]),
            "probability_malicious": float(proba[1]),
            "status": "Benign" if pred == 0 else "Malicious"
        })
    return jsonify({"results": results})

@app.route('/features')
def get_features():
    return jsonify({"features": X_columns})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
