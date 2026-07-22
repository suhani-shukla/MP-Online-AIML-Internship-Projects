import json
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).parent.parent / "models" / "iris_model.pkl"
METRICS_PATH = Path(__file__).parent.parent / "models" / "metrics.json"

def main():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()