import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_model():
    df = pd.read_csv('dd_enhanced.csv')

    # Define Suspicious Ports
    SUSPICIOUS_PORTS = [22, 23, 3389, 5900, 8080]
    joblib.dump(SUSPICIOUS_PORTS, "suspicious_ports.pkl")

    # Feature Engineering
    df['suspicious_src_port'] = df['src_port'].apply(lambda x: 1 if x in SUSPICIOUS_PORTS else 0)

    # Separate Features and Target
    X = df.drop('label', axis=1)
    y = df['label']

    # Save Feature Column Order
    X_columns = list(X.columns)
    joblib.dump(X_columns, "X_columns.pkl")  # changed to .pkl

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model and Scaler
    joblib.dump(model, "network_security_model.pkl")
    joblib.dump(scaler, "port_enhanced_scaler.pkl")

    return model, scaler, X_columns, SUSPICIOUS_PORTS

if __name__ == "__main__":
    train_model()
